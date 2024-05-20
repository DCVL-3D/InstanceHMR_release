from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from models.base import Base
from models.CoordConv import get_coord_maps
from models.basic_modules import BasicBlock,Bottleneck

import config
from config import args
from loss_funcs import Loss
from maps_utils.result_parser import ResultParser

import numpy as np
from torchvision.utils import draw_bounding_boxes, save_image
import math

BN_MOMENTUM = 0.1

class TransBlock(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, bs, h, w, device):
        ones = torch.ones((bs, h, w), dtype=torch.bool, device=device)
        y_embed = ones.cumsum(1, dtype=torch.float32)
        x_embed = ones.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def build_position_encoding(hidden_dim):
    N_steps = hidden_dim // 2

    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

    return position_embedding


class ROMP(Base):
    def __init__(self, backbone=None,**kwargs):
        super(ROMP, self).__init__()
        print('Using ROMP v1')
        self.backbone = backbone
        self._result_parser = ResultParser()
        self._build_head()
        if args().model_return_loss:
            self._calc_loss = Loss()
        if not args().fine_tune and not args().eval:
            self.init_weights()
            self.backbone.load_pretrain_params()

        self.cossim = nn.CosineSimilarity(dim=1)
        self.centerid_trans1 = TransBlock(self.backbone.backbone_channels+2, 128, 3, 2, 1)
        self.centerid_resblock1 = BasicBlock(128, 128)
        self.centerid_trans2 = TransBlock(128+1+24, 128, 3, 1, 1)
        self.centerid_resblock2 = BasicBlock(128, 128)
        self.centerid_outconv = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1,stride=1,padding=0)

        self.jointid_trans1 = TransBlock(self.backbone.backbone_channels+2, 128, 3, 2, 1)
        self.jointid_resblock1 = BasicBlock(128, 128)
        self.jointid_trans2 = TransBlock(128+1+24, 128, 3, 1, 1)
        self.jointid_resblock2 = BasicBlock(128, 128)
        self.jointid_outconv = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1,stride=1,padding=0)

        self.ct_projector = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False),
        )
        self.jt_projector = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False),
        )

        self.pos_encoding = build_position_encoding(hidden_dim=64)
        
        self.params_trans1 = TransBlock(self.backbone.backbone_channels+2, 64, 3, 2, 1)
        self.params_trans_c = TransBlock(64+1+64, 64, 3, 1, 1)
        self.params_trans_j = TransBlock(64+24+64, 64, 3, 1, 1)
        self.params_resblock1 = BasicBlock(128, 128)
        self.params_resblock2 = BasicBlock(128, 128)
        self.params_outconv = nn.Conv2d(in_channels=128, out_channels=self.output_cfg['NUM_PARAMS_MAP'], kernel_size=1, stride=1, padding=0)

    def head_forward(self,x,kp_coords,center_coords, train_flag):
        torch.cuda.empty_cache()
        x = torch.cat((x, self.coordmaps.to(x.device).repeat(x.shape[0],1,1,1)), 1)

        center_maps = self.final_layers[1](x)
        if args().merge_smpl_camera_head:
            cam_maps, params_maps = params_maps[:,:3], params_maps[:,3:]
        else:
            cam_maps = self.final_layers[2](x)
        # to make sure that scale is always a positive value
        cam_maps[:, 0] = torch.pow(1.1,cam_maps[:, 0])


        joint_maps = self.final_layers[3](x)

        x_cid = self.centerid_trans1(x)
        x_jid = self.jointid_trans1(x)

        x_cid = self.centerid_resblock1(x_cid)
        x_jid = self.jointid_resblock1(x_jid)

        x_cid = self.centerid_trans2(torch.cat([x_cid, center_maps, joint_maps], 1))
        x_jid = self.jointid_trans2(torch.cat([x_jid, center_maps, joint_maps], 1))

        x_cid = self.centerid_resblock2(x_cid)
        centeridmap = self.centerid_outconv(x_cid)
        pos_enc = self.pos_encoding(x.shape[0], 64, 64, x.device)

        centeridmap = centeridmap + pos_enc # spatial positional encoding
        x_jid = self.jointid_resblock2(x_jid)
        jointidmap = self.jointid_outconv(x_jid)

        centerid_loss = torch.zeros(1, requires_grad=True).float().to(x.device)
        jointid_loss = torch.zeros(1, requires_grad=True).float().to(x.device)
        if train_flag:
            kp_coords = torch.round(kp_coords).cpu().numpy()
            center_coords = torch.round((center_coords+1.)/2.*64).cpu().numpy()

            batchsize = kp_coords.shape[0]
            jointset = [[] for i in range(batchsize)]
            centerset = [[] for i in range(batchsize)]

            for b in range(batchsize):
                for n in range(64):
                    can = kp_coords[b][n]   # [24, 2]
                    tmp = can[np.any(can != 0, axis=1)] # [M, 2]
                    if tmp.shape[0] > 0:
                        jointset[b].append(tmp.astype(np.uint8))
                        can = center_coords[b][n]
                        can = np.array([can[1], can[0]])
                        centerset[b].append(can.astype(np.uint8))

            jointidset = [[] for i in range(batchsize)]
            centeridset = [[] for i in range(batchsize)]
            jcnt = 0
            ccnt = 0
            for b in range(batchsize):
                joint = jointset[b]
                center = centerset[b]

                for n in range(len(center)):
                    cx, cy = center[n][0], center[n][1]
                    if cx >= 64 or cy >=64:
                        continue
                    centeridset[b].append(self.ct_projector(centeridmap[b, :, cx, cy].unsqueeze(0)))

                    n_joints = joint[n]
                    tmp = []
                    for j in range(len(n_joints)):
                        tmp.append(self.jt_projector(jointidmap[b, :, n_joints[j][0], n_joints[j][1]].unsqueeze(0)))
                    jointidset[b].append(tmp)

                # ID Similarity loss
                N = len(centeridset[b])
                centerid_loss_temp = torch.zeros(1, requires_grad=True).float().to(x.device)
                jointid_loss_temp = torch.zeros(1, requires_grad=True).float().to(x.device)
                for n in range(N):
                    for i in range(n+1, N):
                        centerid_loss_temp += (1+self.cossim(centeridset[b][n], centeridset[b][i]))

                    J = len(jointidset[b][n])
                    jointid_loss_temp_temp = torch.zeros(1, requires_grad=True).float().to(x.device)
                    for j in range(J):
                        jointid_loss_temp_temp += (1-self.cossim(centeridset[b][n], jointidset[b][n][j]))
                    jointid_loss_temp += jointid_loss_temp_temp / J

                if N==0:
                    continue
                elif N==1:
                    jointid_loss += jointid_loss_temp/N
                    jcnt += 1
                else:
                    centerid_loss += centerid_loss_temp/(N*(N-1)/2)
                    jointid_loss += jointid_loss_temp/N
                    ccnt += 1
                    jcnt += 1

            if ccnt > 0:
                centerid_loss /= ccnt
            if jcnt > 0:
                jointid_loss /= jcnt


        x_params = self.params_trans1(x)
        x_params_c = self.params_trans_c(torch.cat([x_params, center_maps, centeridmap], 1))
        x_params_j = self.params_trans_j(torch.cat([x_params, joint_maps, jointidmap], 1))
        x_params = self.params_resblock1(torch.cat([x_params_c, x_params_j], 1))
        x_params = self.params_resblock2(x_params)
        params_maps = self.params_outconv(x_params)

        params_maps = torch.cat([cam_maps, params_maps], 1)

        output = {'params_maps':params_maps.float(), 'center_map':center_maps.float(), 'joint_maps':joint_maps.float(),\
                    'centerid_loss':centerid_loss.float(), 'jointid_loss':jointid_loss.float()}
        return output

    def _build_head(self):
        self.outmap_size = args().centermap_size
        params_num, cam_dim = self._result_parser.params_map_parser.params_num, 3
        self.head_cfg = {'NUM_HEADS': 1, 'NUM_CHANNELS': 64, 'NUM_BASIC_BLOCKS': args().head_block_num}
        self.output_cfg = {'NUM_PARAMS_MAP':params_num-cam_dim, 'NUM_CENTER_MAP':1, 'NUM_CAM_MAP':cam_dim}

        self.final_layers = self._make_final_layers(self.backbone.backbone_channels)
        self.coordmaps = get_coord_maps(128)

    def _make_final_layers(self, input_channels):
        final_layers = []
        final_layers.append(None)

        input_channels += 2
        if args().merge_smpl_camera_head:
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP']+self.output_cfg['NUM_CAM_MAP']))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))
        else:
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CAM_MAP']))
            final_layers.append(self._make_head_layers(input_channels, 24)) # joint heatmap

        return nn.ModuleList(final_layers)
    
    def _make_head_layers(self, input_channels, output_channels):
        head_layers = []
        num_channels = self.head_cfg['NUM_CHANNELS']

        kernel_sizes, strides, paddings = self._get_trans_cfg()
        for kernel_size, padding, stride in zip(kernel_sizes, paddings, strides):
            head_layers.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding),
                    nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)))
        
        for i in range(self.head_cfg['NUM_HEADS']):
            layers = []
            for _ in range(self.head_cfg['NUM_BASIC_BLOCKS']):
                layers.append(nn.Sequential(BasicBlock(num_channels, num_channels)))
            head_layers.append(nn.Sequential(*layers))

        head_layers.append(nn.Conv2d(in_channels=num_channels,out_channels=output_channels,\
            kernel_size=1,stride=1,padding=0))

        return nn.Sequential(*head_layers)


    def _get_trans_cfg(self):
        if self.outmap_size == 32:
            kernel_sizes = [3,3]
            paddings = [1,1]
            strides = [2,2]
        elif self.outmap_size == 64:
            kernel_sizes = [3]
            paddings = [1]
            strides = [2]
        elif self.outmap_size == 128:
            kernel_sizes = [3]
            paddings = [1]
            strides = [1]

        return kernel_sizes, strides, paddings

if __name__ == '__main__':
    args().configs_yml = 'configs/v1.yml'
    args().model_version=1
    from models.build import build_model
    model = build_model().cuda()
    outputs=model.feed_forward({'image':torch.rand(4,512,512,3).cuda()})
    for key, value in outputs.items():
        if isinstance(value,tuple):
            print(key, value)
        elif isinstance(value,list):
            print(key, value)
        else:
            print(key, value.shape)