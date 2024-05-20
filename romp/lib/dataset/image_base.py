import sys, os
import glob
import numpy as np
import random
import cv2
import json
import h5py
import torch
import shutil
import time
import pickle
import copy
import joblib
import logging
import scipy.io as scio
#import quaternion
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

from config import args
import config
import constants

from smpl_family.smpl_regressor import SMPLR
from utils import estimate_translation, normalize_trans_to_cam_params
from utils.augments import get_image_cut_box, process_image, calc_aabb, flip_kps, rot_imgplane, pose_processing, Synthetic_occlusion, convert_bbox2scale
from utils.projection import perspective_projection, filter_out_incorrect_trans
from maps_utils.centermap import CenterMap
if args().learn_2dpose: 
    from maps_utils.kp_group import HeatmapParser
if args().learn_AE:
    from maps_utils.target_generators import JointsGenerator

from maps_utils.target_generators import HeatmapGenerator, ScaleAwareHeatmapGenerator, JointsGenerator  ####################

from utils.cam_utils import convert_scale_to_depth_level
from utils.center_utils import denormalize_center
from maps_utils.centermap import _calc_radius_


JOINT_21 = {
    'L_Hip':0, 'R_Hip':1, 'Spine_SMPL': 2, 'L_Knee':3, 'R_Knee':4, 'Thorax_SMPL': 5, 'L_Ankle':6, 'R_Ankle':7,'Thorax_up_SMPL':8, \
    'L_Toe_SMPL':9, 'R_Toe_SMPL':10, 'Neck': 11, 'L_Collar':12, 'R_Collar':13, 'Jaw':14, 'L_Shoulder':15, 'R_Shoulder':16,\
    'L_Elbow':17, 'R_Elbow':18, 'L_Wrist': 19, 'R_Wrist': 20
    }


SMPL_24 = {
    'Pelvis_SMPL':0, 'L_Hip_SMPL':1, 'R_Hip_SMPL':2, 'Spine_SMPL': 3, 'L_Knee':4, 'R_Knee':5, 'Thorax_SMPL': 6, 'L_Ankle':7, 'R_Ankle':8,'Thorax_up_SMPL':9, \
    'L_Toe_SMPL':10, 'R_Toe_SMPL':11, 'Neck': 12, 'L_Collar':13, 'R_Collar':14, 'Jaw':15, 'L_Shoulder':16, 'R_Shoulder':17,\
    'L_Elbow':18, 'R_Elbow':19, 'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand':22, 'R_Hand':23
    }
SMPL_EXTRA_30 = {
    'Nose':24, 'R_Eye':25, 'L_Eye':26, 'R_Ear': 27, 'L_Ear':28, \
    'L_BigToe':29, 'L_SmallToe': 30, 'L_Heel':31, 'R_BigToe':32,'R_SmallToe':33, 'R_Heel':34, \
    'L_Hand_thumb':35, 'L_Hand_index': 36, 'L_Hand_middle':37, 'L_Hand_ring':38, 'L_Hand_pinky':39, \
    'R_Hand_thumb':40, 'R_Hand_index':41,'R_Hand_middle':42, 'R_Hand_ring':43, 'R_Hand_pinky': 44, \
    'R_Hip': 45, 'L_Hip':46, 'Neck_LSP':47, 'Head_top':48, 'Pelvis':49, 'Thorax_MPII':50, \
    'Spine_H36M':51, 'Jaw_H36M':52, 'Head':53
    }

H36M = {
    'Unkown_part0':0, 'R_Hip':1, 'R_Knee':2, 'R_Ankle':3, 'R_BigToe':4, 'R_SmallToe':5, 'L_Hip':6, 'L_Knee':7, 'L_Ankle':8, 'L_BigToe':9, 'L_SmallToe':10,\
    'Pelvis':11, 'Spine_H36M':12, 'Unkown_part1':13, 'Jaw_H36M':14, 'Head':15, 'Unkown_part2':16, 'L_Shoulder':17, 'L_Elbow':18, 'L_Wrist':19, 'Unkown_part3':20, \
    'Unkown_part4':21, 'Unkown_part5':22, 'Unkown_part6':23, 'Neck':24, 'R_Shoulder':25, 'R_Elbow':26, 'R_Wrist':27,\
    'Unkown_part7':28, 'Unkown_part8':29, 'Unkown_part9':30, 'Unkown_part10':31
    }

MPIINF = {
    'Spine4_unkown': 0, 'Spine3_unkown': 1, 'Spine2_unkown': 2, 'Spine1_unkown': 3, 'Pelvis': 4,\
    'Neck':5, 'Head':6, 'Head_top':7, 'L_Collar':8, 'L_Shoulder':9, 'L_Elbow':10,\
    'L_Wrist':11, 'L_Hand':12, 'R_Collar_MPI':13, 'R_Shoulder':14, 'R_Elbow':15, 'R_Wrist':16,\
    'R_Hand':17, 'L_Hip':18, 'L_Knee':19, 'L_Ankle':20, 'L_SmallToe':21, 'L_BigToe':22,\
    'R_Hip':23, 'R_Knee':24, 'R_Ankle':25, 'R_SmallToe':26, 'R_BigToe':27
    }

COCO = {
    'Nose':0, 'Neck':1, 'R_Shoulder':2, 'R_Elbow':3, 'R_Wrist':4, 'L_Shoulder':5, 'L_Elbow':6, \
    'L_Wrist':7, 'R_Hip': 8, 'R_Knee':9, 'R_Ankle':10, 'L_Hip':11, 'L_Knee':12, 'L_Ankle':13, \
    'R_Eye':14, 'L_Eye':15, 'R_Ear':16, 'L_Ear':17,
    }

MPII = {
    'R_Ankle':0, 'R_Knee':1, 'R_Hip':2, 'L_Hip':3, 'L_Knee':4, 'L_Ankle':5, 'Pelvis':6, 'Thorax_MPII':7,\
    'Neck':8, 'Head_top':9, 'R_Wrist':10, 'R_Elbow':11, 'R_Shoulder':12, 'L_Shoulder':13, 'L_Elbow':14, 'L_Wrist':15,
    }

LSP = {
    'R_Ankle':0, 'R_Knee':1, 'R_Hip':2, 'L_Hip':3, 'L_Knee':4, 'L_Ankle':5, 'R_Wrist':6, 'R_Elbow':7, \
    'R_Shoulder':8, 'L_Shoulder':9, 'L_Elbow':10, 'L_Wrist':11, 'Neck_LSP':12, 'Head_top':13
    }

MuCo = {
    'Head_top':0, 'Thorax_unkown':1, 'R_Shoulder':2, 'R_Elbow':3, 'R_Wrist':4, 'L_Shoulder':5, 'L_Elbow':6, 'L_Wrist':7, 'R_Hip':8, 'R_Knee':9,\
    'R_Ankle':10, 'L_Hip':11, 'L_Knee':12, 'L_Ankle':13, 'Pelvis':14, 'Spine_unkown':15, 'Head':16, 'R_Hand':17, 'L_Hand':18, 'R_BigToe':19, 'L_BigToe':20
    }

Crowdpose = {"L_Shoulder":0, "R_Shoulder":1, "L_Elbow":2, "R_Elbow":3, "L_Wrist":4, "R_Wrist":5,\
     "L_Hip":6, "R_Hip":7, "L_Knee":8, "R_Knee":9, "L_Ankle":10, "R_Ankle":11, "Head_top":12, "Neck_LSP":13}

PW3D = {
    'Nose':0, 'Neck':1, 'R_Shoulder':2, 'R_Elbow':3, 'R_Wrist':4, 'L_Shoulder':5, 'L_Elbow':6, \
    'L_Wrist':7, 'R_Hip': 8, 'R_Knee':9, 'R_Ankle':10, 'L_Hip':11, 'L_Knee':12, 'L_Ankle':13, \
    'R_Eye':14, 'L_Eye':15, 'R_Ear':16, 'L_Ear':17,
    }

constantnames = {'h36m':H36M, 'mpiinf':MPIINF, 'coco':COCO, 'mpii':MPII, 'lsp':LSP, 'muco':MuCo, 'crowdpose':Crowdpose, 'pw3d':PW3D, 'pw3d_vibe':PW3D} 


def joint_mapping_24(source_format, target_format, template_format, datasetname):
    mapping = np.ones(len(template_format),dtype=np.int)*-1
    # print(source_format)
    # print(target_format)
    for joint_name in target_format:
        # print("target joint_name: ", joint_name)
        if joint_name in template_format:
            # print("source joint_name: ", joint_name)
            mapping[template_format[joint_name]] = source_format[joint_name]
        else:
            if datasetname == 'h36m':
                mapping[template_format['Thorax_SMPL']] = source_format['Spine_H36M']
                mapping[template_format['L_Toe_SMPL']] = source_format['L_BigToe']*100 + source_format['L_SmallToe']
                mapping[template_format['R_Toe_SMPL']] = source_format['R_BigToe']*100 + source_format['R_SmallToe']
                mapping[template_format['Pelvis_SMPL']] = source_format['Pelvis']
            elif datasetname == 'mpiinf':
                mapping[template_format['L_Toe_SMPL']] = source_format['L_BigToe']*100 + source_format['L_SmallToe']
                mapping[template_format['R_Toe_SMPL']] = source_format['R_BigToe']*100 + source_format['R_SmallToe']
                mapping[template_format['Pelvis_SMPL']] = source_format['Pelvis']
                mapping[template_format['L_Hand']] = source_format['L_Hand']
                mapping[template_format['R_Hand']] = source_format['R_Hand']
            elif datasetname == 'mpii':
                mapping[template_format['Pelvis_SMPL']] = source_format['Pelvis']
            #     mapping[template_format['Thorax_SMPL']] = source_format['Thorax_MPII']
            elif datasetname == 'lsp':
                mapping[template_format['Jaw']] = source_format['Neck_LSP']
            elif datasetname == 'muco':
                mapping[template_format['L_Toe_SMPL']] = source_format['L_BigToe']
                mapping[template_format['R_Toe_SMPL']] = source_format['R_BigToe']
                mapping[template_format['Pelvis_SMPL']] = source_format['Pelvis']
                mapping[template_format['L_Hand']] = source_format['L_Hand']
                mapping[template_format['R_Hand']] = source_format['R_Hand']
            elif datasetname == 'crowdpose':
                mapping[template_format['Neck']] = source_format['Neck_LSP']
            # elif datasetname == 
    # print(np.array(mapping))
    return np.array(mapping)


class Image_base(Dataset):
    def __init__(self, train_flag=True, regress_smpl = False, **kwargs):
        super(Image_base,self).__init__()
        self.data_folder = args().dataset_rootdir
        if not os.path.isdir(self.data_folder):
            self.data_folder = '/home/yusun/data_drive/dataset'
        self.scale_range = [1.2,2.4]
        self.half_prob = 0.12
        self.noise = 0.2
        self.vis_thresh = 0.03
        self.channels_mix = False
        self.ID_num = 0
        self.min_vis_pts = 2

        self.max_person = args().max_person
        self.multi_mode = args().multi_person
        self.use_eft = args().use_eft

        self.depth_degree_thresh = [0.36,0.18,0]
        self.regress_smpl = regress_smpl
        self.invalid_cam_ds = ['coco', 'lsp', 'mpii', 'up', 'cmup'] # 

        self.homogenize_pose_space = False
        if train_flag:
            self.homogenize_pose_space = args().homogenize_pose_space
            #if args().Synthetic_occlusion_ratio>0:
            #    self.synthetic_occlusion = Synthetic_occlusion(args().voc_dir)
            if args().color_jittering_ratio>0:
                self.color_jittering = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)

        # only shuffle some 3D pose datasets, such as h36m and mpi-inf-3Dhp
        self.shuffle_mode = args().shuffle_crop_mode
        self.shuffle_ratio = args().shuffle_crop_ratio_2d
        self.train_flag=train_flag

        self.input_shape = [args().input_size, args().input_size]
        self.vis_size = args().input_size
        self.labels, self.images, self.file_paths = [],[],[]

        self.fov_tan_itw = np.tan(np.radians(args().FOV/2.))
        self.focal_length = args().focal_length

        self.root_inds = [constants.SMPL_ALL_54['R_Hip'], constants.SMPL_ALL_54['L_Hip']]
        self.neck_idx, self.pelvis_idx = 1,8
        self.torso_ids = [constants.SMPL_ALL_54[part] for part in ['Neck', 'Neck_LSP', 'R_Shoulder', 'L_Shoulder','Pelvis', 'R_Hip', 'L_Hip']]
        self.phase = 'train' if self.train_flag else 'test'
        self.default_valid_mask_3d = np.array([False for _ in range(6)])
        self.heatmap_res = 64
        self.joint_number = len(list(constants.SMPL_ALL_54.keys()))



        # self.heatmap_mapper = constants.joint_mapping(constants.SMPL_ALL_54, constants.COCO_17)
        # self.heatmap_generator = HeatmapGenerator(self.heatmap_res,len(self.heatmap_mapper))

        if args().learn_2dpose:
            self.heatmap_mapper = constants.joint_mapping(constants.SMPL_ALL_54, constants.COCO_17)
            self.heatmap_generator = HeatmapGenerator(self.heatmap_res,len(self.heatmap_mapper))
        if args().learn_AE:
            self.heatmap_mapper = constants.joint_mapping(constants.SMPL_ALL_54, constants.COCO_17)
            self.joint_generator = JointsGenerator(self.max_person,len(self.heatmap_mapper),128,True)
        self.CM = CenterMap()

    def get_item_single_frame(self,index):
        # valid annotation flags for 
        # 0: 2D pose/bounding box(True/False), # detecting all person/front-view person(True/False)
        # 1: 3D pose, 2: subject id, 3: smpl root rot, 4: smpl pose param, 5: smpl shape param, 6: verts, 7:depth
        valid_masks = np.zeros((self.max_person, 8), dtype=np.bool)
        info = self.get_image_info(index)

        datasetname = info['ds']

        position_augments, pixel_augments = self._calc_augment_confs(info['image'], info['kp2ds'], is_pose2d=info['vmask_2d'][:,0])

        img_info = process_image(info['image'], info['kp2ds'], augments=position_augments, is_pose2d=info['vmask_2d'][:,0])
        image, image_wbg, full_kps, offsets = img_info
        centermap, person_centers, full_kp2ds, used_person_inds, valid_masks[:,0], bboxes_hw_norm, heatmap, AE_joints, kp_coords = \
            self.process_kp2ds_bboxes(full_kps, img_shape=image.shape, is_pose2d=info['vmask_2d'][:,0], datasetname=datasetname)
        # print("############## centermap.shape: ", centermap.shape)
        # print("############## heatmap.shape: ", heatmap.shape)
        # print(full_kp2ds.shape) # [64, 54, 2]
        # tmp = torch.from_numpy(full_kp2ds).reshape(-1)
        # print(tmp)
        # print(len(np.where(tmp==-2)))
        # print(np.max(full_kp2ds))
        # tmp = np.array(full_kps)
        # print(np.shape(tmp)) # [N, 54, 3]
        # print(full_kps)
        # tmp = tmp.reshape(-1)
        # print(tmp)

        all_person_detected_mask = info['vmask_2d'][0,2]
        subject_ids, valid_masks[:,2] = self.process_suject_ids(info['track_ids'], used_person_inds, valid_mask_ids=info['vmask_2d'][:,1])
        image, dst_image, org_image = self.prepare_image(image, image_wbg, augments=pixel_augments)

        # valid mask of 3D pose, smpl root rot, smpl pose param, smpl shape param, global translation
        kp3d, valid_masks[:,1] = self.process_kp3ds(info['kp3ds'], used_person_inds, \
            augments=position_augments, valid_mask_kp3ds=info['vmask_3d'][:, 0])
        params, valid_masks[:,3:6] = self.process_smpl_params(info['params'], used_person_inds, \
            augments=position_augments, valid_mask_smpl=info['vmask_3d'][:, 1:4])
        verts_processed, valid_masks[:,6], root_trans_processed, valid_masks[:,7] = self.process_verts(info['verts'], info['root_trans'], used_person_inds, \
            augments=position_augments, valid_mask_verts=info['vmask_3d'][:, 4], valid_mask_depth=info['vmask_3d'][:, 5])

        # print(np.shape(np.array(kp_coords)))
        # print(torch.from_numpy(np.array(kp_coords)))
        # print(torch.from_numpy(np.array(kp_coords)).float().shape())
        kp_coords = torch.from_numpy(np.array(kp_coords)).float()
        # print(kp_coords.shape)
        # print(torch.ones((64-kp_coords.shape[0], 21, 3)).shape)

        # kp_coords = torch.cat([kp_coords, torch.ones(64-kp_coords.shape[0], 21, 3)*-1], 0)

        rot_flip = np.array([position_augments[0], position_augments[1]]) if position_augments is not None else np.array([0,0])
        input_data = {
            'image': torch.from_numpy(dst_image).float(),
            'image_org': torch.from_numpy(org_image),
            'full_kp2d': torch.from_numpy(full_kp2ds).float(),
            'person_centers':torch.from_numpy(person_centers).float(), 
            'subject_ids':torch.from_numpy(subject_ids).long(),
            'centermap': centermap.float(),
            'kp_3d': torch.from_numpy(kp3d).float(),
            'verts': torch.from_numpy(verts_processed).float(),
            'params': torch.from_numpy(params).float(),
            'valid_masks':torch.from_numpy(valid_masks).bool(),
            'root_trans': torch.from_numpy(root_trans_processed).float(),
            'offsets': torch.from_numpy(offsets).float(),
            'rot_flip': torch.from_numpy(rot_flip).float(),
            'all_person_detected_mask':torch.Tensor([all_person_detected_mask]).bool(),
            'imgpath': info['imgpath'],
            'data_set': info['ds'],
            'heatmap': torch.from_numpy(heatmap).float(),    # joint heatmap
            'kp_coords': kp_coords    # keypoint coordinates for joint sampling [N, 21, 3]
            }
        # print("################ info['ds']: ", info['ds'])
        # print("############## info['imgpath']: ", info['imgpath'])
        # print("########################")
        # print(input_data['kp_3d'].shape)
        # print(input_data['person_centers'].shape)
        # print(input_data['kp_coords'].shape)
        # input_data['kp_coords'] = torch.cat([input_data['kp_coords'], torch.ones((64-input_data['kp_coords'].shape()[0], 21, 3)*-1)], 0)
        input_data = self.add_cam_parameters(input_data, info)

        if self.train_flag:
            img_scale = 1 if position_augments is None else position_augments[3]
            input_data['img_scale'] = torch.Tensor([img_scale]).float()

        if args().learn_2dpose:
            input_data.update({'heatmap':torch.from_numpy(heatmap).float()})
        if args().learn_AE:
            input_data.update({'AE_joints': torch.from_numpy(AE_joints).long()})

        if args().perspective_proj:
            root_trans, cam_params, cam_mask = self._calc_normed_cam_params_(full_kp2ds, kp3d, valid_masks[:, 1], info['ds'])
            centermap_3d, valid_centermap3d_mask = self.generate_centermap_3d(person_centers, cam_params, cam_mask, bboxes_hw_norm, all_person_detected_mask)
            input_data.update({'cams': cam_params,'cam_mask': cam_mask, 'root_trans_cam':root_trans,\
                'centermap_3d':centermap_3d.float(),'valid_centermap3d_mask':torch.Tensor([valid_centermap3d_mask]).bool()})

        if args().learn_relative:
            input_data.update({'depth_info': torch.ones(self.max_person, 4).long()*-1,\
                'kid_shape_offsets': torch.zeros(self.max_person).float()})
        #print_data_shape(input_data)

        if 'seq_info' in info:
            input_data.update({'seq_info':torch.Tensor(info['seq_info'])})
        return input_data
    
    def _calc_augment_confs(self, image, full_kp2ds, is_pose2d=None):
        if not self.train_flag:
            return None, None
        
        color_jitter = True if random.random()<args().color_jittering_ratio else False
        syn_occlusion = True if random.random()<args().Synthetic_occlusion_ratio else False
        
        flip = True if random.random()<0.5 else False
        rot = random.randint(-30,30) if random.random()<args().rotate_prob else 0                 
        
        crop_bbox, img_scale = None, 1
        if random.random()<self.shuffle_ratio:
            scale = np.random.rand() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            
            crop_person_number = np.random.randint(len(is_pose2d)) + 1
            sample_ids = np.array(random.sample(list(range(len(full_kp2ds))), crop_person_number))

            bboxes = []
            if is_pose2d[sample_ids].sum()>0:
                sample_ids = sample_ids[np.where(is_pose2d[sample_ids])[0]]
                xys = np.array([full_kp2ds[ind] for ind in sample_ids])[:,:,:2].reshape(-1,2)
                valid_xys = (xys!=-2.).sum(-1)==2
                xys = xys[valid_xys]
                bboxes.append(xys)
            # if crop_person_number==0 and len(full_kp2ds) and random.random()<0.1
            # TODO: crop image area that doesn't contain person. 
            if (~is_pose2d).sum()>0:
                sample_ids = sample_ids[np.where(~is_pose2d[sample_ids])[0]]
                vboxes = np.array([full_kp2ds[ind] for ind in sample_ids])[2:].reshape(-1,2)
                bboxes.append(vboxes)
            bboxes = np.concatenate(bboxes, 0)
            if len(bboxes)>0:
                height, width = image.shape[0], image.shape[1]
                box = calc_aabb(bboxes)
                leftTop, rightBottom = np.clip(box[0], 0, width), np.clip(box[1], 0, height)
                [l, t], [r, b] = get_image_cut_box(leftTop, rightBottom, scale, force_square=True)
                crop_bbox = (l,t,r,b)
                img_scale = convert_bbox2scale(crop_bbox, self.input_shape)
        
        position_augments = (rot, flip, crop_bbox, img_scale)

        pixel_augments = (color_jitter, syn_occlusion)
        return position_augments, pixel_augments

    def process_kps(self,kps,img_size,set_minus=True):
        # print("before: ", kps)
        kps = kps.astype(np.float32)
        kps[:,0] = kps[:,0]/ float(img_size[1])
        kps[:,1] = kps[:,1]/ float(img_size[0])
        kps[:,:2] = 2.0 * kps[:,:2] - 1.0

        if set_minus:
            if kps.shape[1]>2:
                kps[kps[:,2]<=self.vis_thresh] = -2.
            kps=kps[:,:2]
            for inds, kp in enumerate(kps):
                # set the out-of-region points to -2.
                if not _check_upper_bound_lower_bound_(kp, ub=1, lb=-1):
                    kps[inds] = -2.
        # print("after: ", kps)

        return kps

    def generate_centermap_3d(self, person_centers, cam_params, cam_mask, bboxes_hw_norm, all_person_detected_mask):
        valid_pc_yx, valid_scale = person_centers[cam_mask], cam_params[cam_mask, 0]
        person_num = len(bboxes_hw_norm)
        valid_pc_z = convert_scale_to_depth_level(valid_scale).numpy()
        valid_pc_yx = denormalize_center(valid_pc_yx.copy(), args().centermap_size)
        valid_pc_zyx = np.concatenate([valid_pc_z[:,None], valid_pc_yx], 1)
        centermap_3d, valid_centermap3d_mask = self.CM.generate_centermap_3dheatmap_adaptive_scale(valid_pc_zyx[:,[2,1,0]])
        return centermap_3d, valid_centermap3d_mask

    def map_kps(self,joint_org,maps=None):
        kps = joint_org[maps].copy()
        kps[maps==-1] = -2.
        return kps

    def parse_multiperson_kp2ds(self, full_kps, datasetname, visible_full_kps):
        bboxes_normalized = _calc_bbox_normed(full_kps)
        if self.train_flag:
            heatmap, AE_joints, kp_coords = self.generate_heatmap_AEmap(visible_full_kps, datasetname)    ##############################
        else:
            heatmap, AE_joints, kp_coords = np.zeros((21, 64, 64)), np.zeros((self.max_person, 21, 2)), np.zeros((self.max_person, 21, 2))
        # print("##################################### heatmap, AE_joints: ", heatmap.shape, AE_joints.shape)
        # print("##################################### heatmap.shape: ", heatmap.shape)
        # print("##################################### heatmap, AE_joints: ", heatmap, AE_joints)
        # print("max: ", torch.max(torch.from_numpy(heatmap)))
        # print("min: ", torch.min(torch.from_numpy(heatmap)))

        # if args().learn_2dpose or args().learn_AE:
        #     heatmap, AE_joints = self.generate_heatmap_AEmap(full_kps)
        # else:
            # heatmap, AE_joints = np.zeros((17, 128, 128)), np.zeros((self.max_person, 17, 2))
        person_centers, full_kp2ds, bboxes_hw_norm, used_person_inds = [], [], [], []
        for inds in range(min(len(full_kps),self.max_person)):
            center = self._calc_center_(full_kps[inds])
            if center is None or len(center)==0:
                continue
            person_centers.append(center)
            full_kp2ds.append(full_kps[inds])
            bboxes_hw_norm.append((bboxes_normalized[inds][1]-bboxes_normalized[inds][0])[::-1])
            used_person_inds.append(inds)
        
        person_centers, full_kp2ds = np.array(person_centers), np.array(full_kp2ds)
        occluded_by_who = detect_occluded_person(person_centers,full_kp2ds) if args().collision_aware_centermap else None
        # print(person_centers)
        return person_centers, full_kp2ds, used_person_inds, bboxes_hw_norm, occluded_by_who, heatmap, AE_joints, kp_coords

    def parse_bboxes(self, full_kps, hw_ratio_thresh=0.8):
        person_centers, bboxes_hw_norm, used_person_inds = [], [], []
        fbox, vbox = full_kps[:,:2], full_kps[:,2:]
        fwh, vwh = fbox[:,1]-fbox[:,0], vbox[:,1]-vbox[:,0]
        fhw_ratios, vhw_ratios = fwh[:,1]/(fwh[:,0]+1e-4), vwh[:,1]/(vwh[:,0]+1e-4)
        fb_centers = np.stack([0.5*fbox[:,0,0]+0.5*fbox[:,1,0], 0.7*fbox[:,0,1]+0.3*fbox[:,1,1]], 1)
        vb_centers = np.stack([0.5*vbox[:,0,0]+0.5*vbox[:,1,0], 0.6*vbox[:,0,1]+0.4*vbox[:,1,1]], 1)
        for inds, (fhw_ratio, vhw_ratio) in enumerate(zip(fhw_ratios, vhw_ratios)):
            if len(used_person_inds)>=self.max_person:
                continue
            if vhw_ratio>1. and _check_upper_bound_lower_bound_(fb_centers[inds], ub=1, lb=-1):
                person_centers.append(fb_centers[inds])
                bboxes_hw_norm.append(fwh[inds][::-1])
                used_person_inds.append(inds)
            elif _check_upper_bound_lower_bound_(vb_centers[inds], ub=1, lb=-1):
                person_centers.append(vb_centers[inds])
                bboxes_hw_norm.append(vwh[inds][::-1])
                used_person_inds.append(inds)

        return person_centers, bboxes_hw_norm, used_person_inds

    def process_kp2ds_bboxes(self, full_kps, img_shape=None, is_pose2d=None, datasetname=None):
        person_centers = np.ones((self.max_person, 2))*-2.
        full_kp2ds = np.ones((self.max_person, self.joint_number, 2))*-2.
        valid_mask_kp2ds = np.zeros(self.max_person, dtype=np.bool)
        used_person_inds, bboxes_hw_norm, occluded_by_who = [], [], None

        # print(type(full_kps))
        # print(full_kps)
        visible_full_kps = np.array(full_kps).copy()
        visible_full_kps[visible_full_kps[:,:,2]==1] = [-2,-2,-2]

        
        if is_pose2d.sum()>0:
            # mask out the out-of-range subject because of getting cropped out from image area.
            full_kp2d = [self.process_kps(full_kps[ind], img_shape,set_minus=True) for ind in np.where(is_pose2d)[0]]
            visible_full_kp2d = [self.process_kps(visible_full_kps[ind], img_shape,set_minus=True) for ind in np.where(is_pose2d)[0]]
            # print("#######################")
            # print(full_kps)
            # print(full_kp2d)
            # print("full_kp2d: ", np.shape(full_kp2d))   # N, 54, 2
            person_centers_kp2d, full_kp2ds_kp2d, mask_kp2d, bboxes_hw_norm_kp2d, occluded_by_who, heatmap, AE_joints, kp_coords = self.parse_multiperson_kp2ds(full_kp2d, datasetname, visible_full_kp2d)
            # print("####################################### ", heatmap.shape)
            mask_kp2d = np.where(is_pose2d)[0][np.array(mask_kp2d,dtype=np.int32)].tolist()
            if len(mask_kp2d)>0:
                person_centers[:len(mask_kp2d)], full_kp2ds[:len(mask_kp2d)], valid_mask_kp2ds[:len(mask_kp2d)] = person_centers_kp2d, full_kp2ds_kp2d, True
                used_person_inds += mask_kp2d
                bboxes_hw_norm += bboxes_hw_norm_kp2d
        
        if (~is_pose2d).sum()>0:
            full_bboxes = np.array([self.process_kps(full_kps[ind], img_shape,set_minus=False) for ind in np.where(~is_pose2d)[0]])
            person_centers_bbox, bboxes_hw_norm_bbox, mask_bbox = self.parse_bboxes(full_bboxes)
            mask_bbox = np.where(~is_pose2d)[0][np.array(mask_bbox,dtype=np.int32)].tolist()
            left_num = max(0, min(self.max_person - len(used_person_inds), len(mask_bbox)))
            if left_num != len(mask_bbox):
                person_centers_bbox, bboxes_hw_norm_bbox, mask_bbox = person_centers_bbox[:left_num], bboxes_hw_norm_bbox[:left_num], mask_bbox[:left_num]
            if len(mask_bbox)>0:
                person_centers[len(used_person_inds):len(used_person_inds)+left_num] = person_centers_bbox
                used_person_inds += mask_bbox
                bboxes_hw_norm += bboxes_hw_norm_bbox
        
        if is_pose2d.sum() == 0:
            # heatmap, AE_joints = np.zeros((17, 128, 128)), np.zeros((self.max_person, 17, 2))
            heatmap, AE_joints = np.zeros((21, 128, 128)), np.zeros((self.max_person, 17, 2))   #############################
        # person_centers changed after CAR processing
        centermap = self.CM.generate_centermap(person_centers, bboxes_hw_norm=bboxes_hw_norm, occluded_by_who=occluded_by_who)
        # rectify the x, y order, from x-y to y-x
        person_centers = person_centers[:,::-1].copy()
        return centermap, person_centers, full_kp2ds, used_person_inds, valid_mask_kp2ds, bboxes_hw_norm, heatmap, AE_joints, kp_coords

    def process_suject_ids(self, subject_ids, used_person_inds, valid_mask_ids=False):
        person_ids, valid_id_mask = np.ones(self.max_person)*-1, np.zeros(self.max_person,dtype=np.bool)
        if subject_ids is None:
            return person_ids, valid_id_mask
        for inds, s_inds in enumerate(used_person_inds):
            person_ids[inds] = subject_ids[s_inds]
            valid_id_mask[inds] = valid_mask_ids[s_inds]
        return person_ids, valid_id_mask

    def prepare_image(self, image, image_wbg, augments=None):
        if augments is not None:
            image = self.aug_image(image, augments[0], augments[1])
        try:
            dst_image = cv2.resize(image, tuple(self.input_shape), interpolation = cv2.INTER_CUBIC)
            org_image = cv2.resize(image_wbg, (self.vis_size, self.vis_size), interpolation=cv2.INTER_CUBIC)
        except:
            dst_image = np.zeros((self.input_shape[0], self.input_shape[1], 3))
            org_image = np.zeros((self.vis_size, self.vis_size, 3))
        return image, dst_image, org_image

    def process_kp3ds(self, kp3ds, used_person_inds, augments=None, valid_mask_kp3ds=None):
        kp3d_flag = np.zeros(self.max_person, dtype=np.bool)
        joint_num = self.joint_number if self.train_flag or kp3ds is None else kp3ds[0].shape[0]
        kp3d_processed = np.ones((self.max_person, joint_num, 3), dtype=np.float32)*-2. # -2 serves as an invisible flag

        for inds, used_id in enumerate(used_person_inds):
            if valid_mask_kp3ds[used_id]:
                kp3d, kp3d_flag[inds] = kp3ds[used_id], valid_mask_kp3ds[used_id]
                valid_mask = self._check_kp3d_visible_parts_(kp3d)
                if self.root_inds is not None:
                    kp3d -= kp3d[self.root_inds].mean(0)[None]
                if augments is not None:
                    if augments[0]!=0:
                        kp3d = rot_imgplane(kp3d, augments[0])
                    if augments[1]:
                        kp3d = flip_kps(kp3d,flipped_parts=constants.All54_flip)
                        valid_mask = valid_mask[constants.All54_flip]
                kp3d[~valid_mask] = -2.
                kp3d_processed[inds] = kp3d
             
        return kp3d_processed, kp3d_flag

    def process_smpl_params(self, params, used_person_inds, augments=None, valid_mask_smpl=None):
        params_processed = np.ones((self.max_person,76), dtype=np.float32)*-10
        smpl_flag = np.zeros((self.max_person, 3), dtype=np.bool)

        for inds, used_id in enumerate(used_person_inds):
            if valid_mask_smpl[used_id].sum()>0:
                param, smpl_flag[inds] = params[used_id], valid_mask_smpl[used_id]
                theta, beta = param[:66], param[-10:]
                if augments is not None:
                    theta = pose_processing(theta,augments[0],augments[1],valid_grot=smpl_flag[inds,0],valid_pose=smpl_flag[inds,1])
                params_processed[inds] = np.concatenate([theta, beta])

        return params_processed, smpl_flag

    def process_verts(self, verts, root_trans, used_person_inds, augments=None, valid_mask_verts=None, valid_mask_depth=None):
        verts_processed = np.ones((self.max_person,6890, 3), dtype=np.float32)*-10
        verts_flag = np.zeros(self.max_person, dtype=np.bool)
        root_trans_processed = np.ones((self.max_person, 3), dtype=np.float32)*-2
        depth_flag = np.zeros(self.max_person, dtype=np.bool)
        if root_trans is not None:
            root_trans_processed[:len(used_person_inds)] = root_trans[used_person_inds]
            depth_flag[:len(used_person_inds)] = valid_mask_depth[used_person_inds]
        if verts is not None:
            for inds, used_id in enumerate(used_person_inds):
                if valid_mask_verts[used_id].sum()>0:
                    vert = verts[used_id]# - np.expand_dims(root_trans[used_id], axis=0)
                    if augments is not None:
                        if augments[0]!=0:
                            vert = rot_imgplane(vert, augments[0])
                        if augments[1]:
                            vert[:,0] *= -1
                    verts_processed[inds] = vert
                    verts_flag[inds] = valid_mask_verts[used_id]

        return verts_processed, verts_flag, root_trans_processed, depth_flag

    def _calc_normed_cam_params_(self, full_kp2ds, kp_3ds, kp3d_flag, ds, thresh=50):
        cam_params = torch.ones(self.max_person,3,dtype=torch.float32)*-2
        cam_mask = torch.zeros(self.max_person,dtype=torch.bool)
        root_trans  = torch.ones(self.max_person,3,dtype=torch.float32)*-2

        if kp3d_flag.sum()==0 or ds in self.invalid_cam_ds:
            return root_trans,cam_params, cam_mask

        kp_3ds_2dformat = kp_3ds[kp3d_flag]
        kp2ds = denormalize_kp2ds(full_kp2ds[kp3d_flag])
        
        trans = estimate_translation(kp_3ds_2dformat, kp2ds, \
                focal_length=self.focal_length, img_size=np.array(self.input_shape))
        trans, valid_mask = filter_out_incorrect_trans(kp_3ds_2dformat, trans, kp2ds, thresh=thresh, \
                focal_length=self.focal_length, center_offset=torch.Tensor(self.input_shape)/2.)
        kp3d_flag[kp3d_flag] = valid_mask
        # rectify the root displacement between SMPL 24 and OpenPose 25 body joints
        normed_cams = normalize_trans_to_cam_params(trans)
    
        if valid_mask.sum()>0:
            cam_params[kp3d_flag] = torch.from_numpy(normed_cams).float()
            root_trans[kp3d_flag] = trans.float()
            cam_mask[kp3d_flag] = True
        return root_trans, cam_params, cam_mask

    def regress_kp3d_from_smpl(self, params, maps=None, genders=None):
        kp3ds = None
        if params is not None and self.regress_smpl:
            kp3ds = []
            for inds, param in enumerate(params):
                if param is not None:
                    pose, beta = np.concatenate([param[:-10], np.zeros(6)]),param[-10:]
                    gender = 'n' if genders is None else genders[inds]
                    verts, kp3d = self.smplr(pose, beta, gender=gender)
                    kp3d = kp3d[0]
                    if maps is not None:
                        kp3d = self.map_kps(kp3d,maps=maps)
                    kp3ds.append(kp3d)
                else:
                    kp3ds.append(None)
        return kp3ds

    def generate_heatmap_AEmap(self, full_kps, datasetname):
        heatmaps, AE_joints = None, None
        if args().learn_2dpose or args().learn_AE:
            full_kps_hm = [(kps_i+1.)/2.*self.heatmap_res for kps_i in full_kps]
            full_kps_hm = [np.concatenate([kps_i,(kps_i[:,0]>0)[:,None]],-1) for kps_i in full_kps_hm]
            heatmap_kps = []
            for kps in full_kps_hm:
                heatmap_kps.append(kps[self.heatmap_mapper])


        ################################################################################
        # print("################ len(full_kps): ", len(full_kps))    # 3

        constantname = constantnames[datasetname]
        self.heatmap_mapper = joint_mapping_24(constants.SMPL_ALL_54, constantname, SMPL_24, datasetname) #################################
        # print("################### self.heatmap_mapper: ", self.heatmap_mapper) 
        self.heatmap_generator = HeatmapGenerator(self.heatmap_res,len(self.heatmap_mapper))    #################################
        self.joint_generator = JointsGenerator(self.max_person,len(self.heatmap_mapper),64,False)    #################################

        # print(full_kps)
        full_kps_hm = [(kps_i+1.)/2.*self.heatmap_res for kps_i in full_kps]
        full_kps_hm = [np.concatenate([kps_i,(kps_i[:,0]>0)[:,None]],-1) for kps_i in full_kps_hm]
        heatmap_kps = []
        for kps in full_kps_hm:
            # print("########### kps.shape: ", kps.shape) # 54, 3
            # print("########### kps: ", kps) # 
            # print("########### self.heatmap_mapper: ", self.heatmap_mapper)
            # print("########### kps[self.heatmap_mapper].shape: ", kps[self.heatmap_mapper].shape) # 21, 3

            # heatmap_kps.append(kps[self.heatmap_mapper])

            heatmap_kps_temp = []
            for i in range(len(self.heatmap_mapper)):
                if self.heatmap_mapper[i] == -1:
                    heatmap_kps_temp.append(np.ones(3, dtype=np.int)*-1)
                elif self.heatmap_mapper[i] < 54:
                    heatmap_kps_temp.append(kps[self.heatmap_mapper[i]])
                else:
                    index1 = self.heatmap_mapper[i]//100
                    index2 = self.heatmap_mapper[i]%100
                    heatmap_kps_temp.append((kps[index1]+kps[index2])/2)
            heatmap_kps.append(heatmap_kps_temp)

        # print("###### heatmap_kps: ", heatmap_kps)  # 


        #     print("###### len(heatmap_kps): ", len(heatmap_kps))  #various (maybe number of persons in frame)
            # print("###### np.shape(heatmap_kps): ", np.shape(heatmap_kps))  # N, 21, 3
        # print("################ len(heatmap_kps): ", len(heatmap_kps))  #various (maybe number of persons in frame)
        heatmaps = self.heatmap_generator.single_process(heatmap_kps)
        kp_coords = self.joint_generator.single_process(heatmap_kps)
        # print(np.shape(kp_coords))
        # print(kp_coords)
        ################################################################################

        if args().learn_2dpose:
            heatmaps = self.heatmap_generator.single_process(heatmap_kps)
        if args().learn_AE:
            AE_joints = self.joint_generator.single_process(heatmap_kps)
        return heatmaps, AE_joints, kp_coords

    def _check_kp3d_visible_parts_(self, kp3ds, invisible_flag=-2.):
        visible_parts_mask = (kp3ds!=invisible_flag).sum(-1) == kp3ds.shape[-1]
        return visible_parts_mask

    def parse_cluster_results(self, cluster_results_file, file_paths):
        annots = np.load(cluster_results_file, allow_pickle=True)
        cluster_results, img_names = annots['kp3ds'], annots['img_names']
        cluster_dict = {os.path.basename(img_name): cluster_id for img_name, cluster_id in zip(img_names, cluster_results)}
        cluster_num = max(cluster_results)+1
        cluster_pool = [[] for _ in range(cluster_num)]
        for inds, img_name in enumerate(file_paths):
            cluster_pool[cluster_dict[os.path.basename(img_name)]].append(inds)
        return cluster_pool

    def homogenize_pose_sample(self, index):
        cluster_num = len(self.cluster_pool)
        return random.sample(self.cluster_pool[index%cluster_num],1)[0]

    def get_image_info(self,index):
        raise NotImplementedError

    def resample(self):
        return self.__getitem__(random.randint(0,len(self)))

    def reget_info(self):
        return self.get_image_info(random.randint(0,len(self)))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        return self.get_item_single_frame(index)
        try:
            return self.get_item_single_frame(index)
        except:
            print('eror!!!! dataset length:',len(self))
            index = np.random.randint(len(self))
            return self.get_item_single_frame(index)

    def aug_image(self, image, color_jitter, syn_occlusion):      
        if syn_occlusion:
            image = self.synthetic_occlusion(image)
        if color_jitter:
            image = np.array(self.color_jittering(Image.fromarray(image)))

        return image

    def read_pkl(self,file_path):
        return pickle.load(open(file_path,'rb'),encoding='iso-8859-1')

    def read_json(self,file_path):
        with open(file_path,'r') as f:
            file = json.load(f)
        return file

    def read_npy(self,file_path):
        return np.load(file_path)

    def _calc_center_(self, kps):
        center = None
        if args().center_def_kp:
            vis = kps[self.torso_ids,0]>-1
            if vis.sum()>0:
                center = kps[self.torso_ids][vis].mean(0)
            elif (kps[:,0]>-1).sum()>0:
                center = kps[kps[:,0]>-1].mean(0)
        else:
            vis = kps[:,0]>-1
            if vis.sum()>self.min_vis_pts:
                center = kps[vis].mean(0)
        return center

    def add_cam_parameters(self, input_data, info):
        if 'camMats' in info:
            input_data['camMats'] = torch.from_numpy(info['camMats']).float()
        else:
            input_data['camMats'] = torch.tensor([[args().focal_length, 0, info['img_size'][1]/2.],\
                                                  [0, args().focal_length, info['img_size'][0]/2.],\
                                                  [0, 0, 0]])
        if 'camDists' in info:
            input_data['camDists'] = torch.from_numpy(info['camDists']).float()
        else:
            input_data['camDists'] = torch.zeros(5).float()
        return input_data


def detect_occluded_person(person_centers, full_kp2ds, thresh=2*64/512.):
    person_num = len(person_centers)
    # index of the person at the front of an occlusion
    occluded_by_who = np.ones(person_num)*-1
    if person_num>1:
        for inds, (person_center, kp2d) in enumerate(zip(person_centers, full_kp2ds)):
            dist = np.sqrt(((person_centers-person_center)**2).sum(-1))
            if (dist>0).sum()>0:
                if (dist[dist>0]<thresh).sum()>0:
                    # Comparing the visible keypoint number to justify whether the person is occluded by the others
                    if (full_kp2ds[inds,:,0]>0).sum() >= (kp2d[:,0]>0).sum():
                        closet_idx = np.where(dist==np.min(dist[dist>0]))[0][0]
                        if occluded_by_who[closet_idx]<0:
                            occluded_by_who[inds] = closet_idx

    return occluded_by_who.astype(np.int)

def _calc_bbox_normed(full_kps):
    bboxes = []
    for kps_i in full_kps:
        if (kps_i[:,0]>-2).sum()>0:
            bboxes.append(calc_aabb(kps_i[kps_i[:,0]>-2]))
        else:
            bboxes.append(np.zeros((2,2)))

    return bboxes

        
def _check_minus2_error_(kp3ds, acceptable_list=[-2., 0.]):
    kp3ds_flatten = kp3ds[:,1:].reshape(-1, 3)
    for kp3d in kp3ds_flatten:
        if kp3d[0] in acceptable_list and kp3d[1] in acceptable_list and kp3d[2] in acceptable_list:
            continue

        equal_kp_value = (kp3ds_flatten[:,0] == kp3d[0]).long() + (kp3ds_flatten[:,1] == kp3d[1]).long() + (kp3ds_flatten[:,2] == kp3d[2]).long()
        equal_mask = equal_kp_value==3
        if equal_mask.sum()>3:
            print(torch.where((kp3ds[:,:] == kp3d).sum(-1)==3))
            print('there are incorrect process that may break the state of invisible flag -2., and make the value becomes {}'.format(kp3ds_flatten[equal_mask]))
            #raise Exception

def _check_upper_bound_lower_bound_(kps, ub=1, lb=-1):
    for k in kps:
        if k >= ub or k <= lb:
            return False
    return True

def check_and_mkdir(dir):
    os.makedirs(dir,exist_ok=True)

def test_projection_depth(pred_joints, trans_xyz, depth_pred, fov_tan):
    from utils.projection import batch_persp_depth
    projected_kp2d = batch_persp_depth(pred_joints, trans_xyz, depth_pred, fov_tan)
    return projected_kp2d

def denormalize_kp2ds(mat, img_size=args().input_size):
    return (mat+1)/2*img_size

def print_data_shape(data):
    for key,value in data.items():
        if isinstance(value,torch.Tensor):
            print(key,value.shape)
        elif isinstance(value,list):
            print(key,len(value))
        elif isinstance(value,str):
            print(key,value)

def test_image_dataset(dataset,with_3d=False,with_smpl=False):
    #args().configs_yml='configs/basic_training_resnet.yml'
    print('configs_yml:', args().configs_yml)
    print('model_version:',args().model_version)
    
    from visualization.visualization import Visualizer, draw_skeleton_multiperson
    test_projection_part = True if args().model_version in [4,5,6,7] else False
    print('test_projection_part:',test_projection_part)

    save_dir = os.path.join(config.project_dir,'test')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    print('Initialized dataset')

    batch_size, model_type= 2, 'smpl'
    dataloader = DataLoader(dataset = dataset,batch_size = batch_size,shuffle = True,\
        drop_last = False,pin_memory = True,num_workers = 1)
    visualizer = Visualizer(resolution = (512,512,3), result_img_dir=save_dir,with_renderer=True)

    from visualization.visualization import make_heatmaps
    from utils.cam_utils import denormalize_cam_params_to_trans
    if with_smpl:
        from smpl_family.smpl import SMPL
        smpl = SMPL(args().smpl_model_path, model_type='smpl')
    print('Initialized SMPL models')

    img_size = 512
    bones, cm = constants.All54_connMat, constants.cm_All54

    for _,r in enumerate(dataloader):
        if _%100==0:
            print_data_shape(r)
        _check_minus2_error_(r['kp_3d'])

        for inds in range(2):
            img_bsname = os.path.basename(r['imgpath'][inds])
            image = r['image'][inds].numpy().astype(np.uint8)[:,:,::-1]
            full_kp2d = (r['full_kp2d'][inds].numpy() + 1) * img_size / 2.0
            person_centers = (r['person_centers'][inds].numpy() + 1) * img_size / 2.0
            subject_ids = r['subject_ids'][inds]
            image_kp2d = draw_skeleton_multiperson(image.copy(), full_kp2d, bones=bones, cm=cm)
            
            if test_projection_part and r['cam_mask'][inds].sum()>0:
                cam_mask = r['cam_mask'][inds]
                kp3d_tp = r['kp_3d'][inds][cam_mask].clone()
                kp2d_tp = r['full_kp2d'][inds][cam_mask].clone()
                pred_cam_t = denormalize_cam_params_to_trans(r['cams'][inds][cam_mask].clone(), positive_constrain=True)
                print('pred_cam_t:',pred_cam_t.half())
                pred_keypoints_2d = perspective_projection(kp3d_tp,translation=pred_cam_t,focal_length=args().focal_length, normalize=False)+512//2
                invalid_mask = np.logical_or(kp3d_tp[:,:,-1]==-2., kp2d_tp[:,:,-1]==-2.)
                pred_keypoints_2d[invalid_mask] = -2.
                image_kp2d_projection = draw_skeleton_multiperson(image.copy(), pred_keypoints_2d, bones=bones, cm=cm)
                cv2.imwrite('{}/{}_{}_projection.jpg'.format(save_dir,_,img_bsname), image_kp2d_projection)

            for person_center, subject_id in zip(person_centers,subject_ids):
                y,x = person_center.astype(np.int)
                if y>0 and x>0:
                    image_kp2d[y-10:y+10, x-10:x+10] = [0,0,255]
                    cv2.putText(image_kp2d,'id:{}'.format(subject_id), (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),1)
          
            centermap_color = make_heatmaps(image.copy(), r['centermap'][inds])
            image_vis = np.concatenate([image_kp2d, centermap_color],1)
            cv2.imwrite('{}/{}_{}_centermap.jpg'.format(save_dir,_,img_bsname), image_vis)
            if 'heatmap' in r:
                heatmap_color = make_heatmaps(image.copy(), r['heatmap'][inds])
                cv2.imwrite('{}/{}_{}_heatmap.jpg'.format(save_dir,_,img_bsname), heatmap_color)

            person_centers_onmap = ((r['person_centers'][inds].numpy() + 1)/ 2.0 * (args().centermap_size-1)).astype(np.int)
            positive_position = torch.stack(torch.where(r['centermap'][inds,0]==1)).permute(1,0)

        if with_smpl and r['valid_masks'][0,0,4]:
            params, subject_ids = r['params'][0],  r['subject_ids'][0]
            image = r['image_org'][0].numpy().astype(np.uint8)[:,:,::-1]
            valid_mask = torch.where(r['valid_masks'][0,:,4])[0]
            subject_ids = subject_ids[valid_mask]
            pose, betas = params[valid_mask][:,:66].float(), params[valid_mask][:,-10:].float()
            pose = torch.cat([pose, torch.zeros(len(pose),6)],-1).float()
            print(pose[:,12*3:13*3])
            verts, joints = smpl(poses=pose, betas=betas, get_skin = True)
            if r['valid_masks'][0,0,6]:
                verts = verts[0][r['valid_masks'][0,:,6]]

            if test_projection_part:
                if r['cam_mask'][0][valid_mask].sum()>0:
                    trans = denormalize_cam_params_to_trans(r['cams'][0][valid_mask].clone(), positive_constrain=True)
                else:
                    trans = r['root_trans'][0][valid_mask]
                pred_keypoints_2d = perspective_projection(joints,translation=trans,focal_length=args().focal_length, normalize=False)+512//2
   
                render_img = visualizer.visualize_renderer_verts_list([verts.cuda()], trans=[trans.cuda()], images=image[None])[0]
                image_kp2d_projection_smpl24 = draw_skeleton_multiperson(render_img.copy(), pred_keypoints_2d, bones=bones[:23], cm=cm[:23], label_kp_order=True)
                image_kp2d_projection_extra30 = draw_skeleton_multiperson(render_img.copy(), pred_keypoints_2d, bones=bones[23:], cm=cm[23:], label_kp_order=True)
                #cv2.imwrite('{}/op25_regressed_{}.jpg'.format(save_dir,_), image_kp2d_projection)
                rendered_img_bv = visualizer.visualize_renderer_verts_list([verts.cuda()], trans=[trans.cuda()], bird_view=True, auto_cam=True)[0]
                cv2.imwrite('{}/mesh_{}.png'.format(save_dir,_), np.concatenate([render_img, rendered_img_bv, image_kp2d_projection_smpl24, image_kp2d_projection_extra30],1))
            else:
                verts[:,:,2] += 2
                render_img = visualizer.visualize_renderer_verts_list([verts.cuda()], images=image[None])[0]
                cv2.imwrite('{}/mesh_{}.png'.format(save_dir,_), render_img)
        j3ds = r['kp_3d'][0,0]
        image = r['image_org'][0].numpy().astype(np.uint8)[:,:,::-1]
        if r['valid_masks'][0,0,1]:
            pj2d = (j3ds[:,:2] + 1) * img_size / 2.0
            pj2d[j3ds[:,-1]==-2.] = -2.
            image_pkp3d = visualizer.draw_skeleton(image.copy(), pj2d, bones=bones, cm=cm)
            cv2.imwrite('{}/pkp3d_{}_{}.png'.format(save_dir,_,r['subject_ids'][0, 0]), image_pkp3d)