
from .base import *
from loss_funcs import _calc_MPJAE, calc_mpjpe, calc_pampjpe, align_by_parts
from .eval import val_result,print_results, calc_outputs_evaluation_matrix, _init_error_dict
from visualization.visualization import draw_skeleton_multiperson
import pandas
import pickle

class Demo(Base):
    def __init__(self):
        super(Demo, self).__init__()
        self._build_model_()
        self.test_cfg = {'mode':'parsing', 'calc_loss': False,'with_nms':True,'new_training': args().new_training}
        self.eval_dataset = args().eval_dataset
        self.save_mesh = False
        print('Initialization finished!')

    def test_eval(self):
        if self.eval_dataset == 'pw3d_test':
            data_loader = self._create_single_data_loader(dataset='pw3d', train_flag = False, mode='vibe', split='test')
        elif self.eval_dataset == 'pw3d_oc':
            data_loader = self._create_single_data_loader(dataset='pw3d', train_flag = False, split='all', mode='OC')
        elif self.eval_dataset == 'pw3d_pc':
            data_loader = self._create_single_data_loader(dataset='pw3d', train_flag = False, split='all', mode='PC')
        elif self.eval_dataset == 'pw3d_nc':
            data_loader = self._create_single_data_loader(dataset='pw3d', train_flag = False, split='all', mode='NC')
        MPJPE, PA_MPJPE, eval_results = val_result(self,loader_val=data_loader, evaluation=True)

    def net_forward(self,meta_data,mode='val'):
        if mode=='val':
            cfg_dict = self.test_cfg
        elif mode=='eval':
            cfg_dict = self.eval_cfg
        ds_org, imgpath_org = get_remove_keys(meta_data,keys=['data_set','imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['image']))
        if self.model_precision=='fp16':
            with autocast():
                outputs = self.model(meta_data, **cfg_dict)
        else:
            outputs = self.model(meta_data, **cfg_dict)

        outputs['meta_data']['data_set'], outputs['meta_data']['imgpath'] = reorganize_items([ds_org, imgpath_org], outputs['reorganize_idx'].cpu().numpy())
        return outputs

    def test_cmu_panoptic(self):
        action_name = ['haggling', 'mafia', 'ultimatum', 'pizza']
        mpjpe_cacher = {aname:AverageMeter() for aname in action_name}
        J_regressor_h36m = torch.from_numpy(np.load(args().smpl_J_reg_h37m_path)).float()
        data_loader = self._create_single_data_loader(dataset='cmup', train_flag=False, split='test')
        bias = []
        self.model.eval()
        with torch.no_grad():
            for test_iter,meta_data in enumerate(data_loader):
                outputs = self.net_forward(meta_data,mode='eval')
                meta_data = outputs['meta_data']
                pred_vertices = outputs['verts'].float()
                J_regressor_batch = J_regressor_h36m[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
                pred_kp3ds = torch.matmul(J_regressor_batch, pred_vertices)
                gt_kp3ds = meta_data['kp_3d'].cuda()
                visible_kpts = (gt_kp3ds[:,:,0]>-2.).float()
                pred_kp3ds -= pred_kp3ds[:,[0]]
                gt_kp3ds -= gt_kp3ds[:,[0]]

                # following the code of coherece reconstruction of multiperson Jiang et. al.
                # Brought from https://github.com/JiangWenPL/multiperson/blob/4d3dbae945e22bb1e270521b061a837976699685/mmdetection/mmdet/core/utils/eval_utils.py#L265
                mpjpes = (torch.sqrt(((pred_kp3ds - gt_kp3ds) ** 2).sum(dim=-1)) * visible_kpts) *1000
                mpjpes = mpjpes.mean(-1)

                pampjpes, transform_mat = calc_pampjpe(gt_kp3ds, pred_kp3ds,return_transform_mat=True)#pelvis:0 # rhip:4, lhip:1, rshoulder:14,lshoulder:11
                pampjpes = pampjpes*1000
                #print(transform_mat[2].reshape(-1,3).mean(0))
                bias.append(transform_mat[2].reshape(-1,3).mean(0).cpu().numpy())

                for img_path, mpjpe in zip(meta_data['imgpath'], mpjpes):
                    for aname in action_name:
                        if aname in os.path.basename(img_path):
                            mpjpe_cacher[aname].update(float(mpjpe.item()))
                if test_iter%50==0:
                    print(test_iter,'/',len(data_loader))
                    print('dataset bias: ', np.array(bias).mean(0))
                    for key,value in mpjpe_cacher.items():
                        print('MPJPE results of {}: {}'.format(key, value.avg))
        
        print('-'*30)
        print('Final results:')
        print('dataset bias: ', np.array(bias).mean(0))
        avg_all = []
        for key,value in mpjpe_cacher.items():
            print(key, value.avg)
            avg_all.append(value.avg)
        print('MPJPE results:', np.array(avg_all).mean())

    def test_crowdpose(self, set_name='val'):
        import json
        from crowdposetools.coco import COCO
        from crowdposetools.cocoeval import COCOeval

        predicted_results = []
        test_save_dir = os.path.join(config.project_dir,'results_out/results_crowdpose')
        os.makedirs(test_save_dir,exist_ok=True)
        results_json_name = os.path.join(config.project_dir,'results_out/V{}_crowdpose_{}_{}.json'.format(self.model_version,set_name,self.backbone))

        self.model.eval()
        kp2d_mapper = constants.joint_mapping(constants.SMPL_ALL_54,constants.Crowdpose_14)
        data_loader = self._create_single_data_loader(dataset='crowdpose', train_flag = False, split=set_name)
        vis_dict = {}
        with torch.no_grad():
            for test_iter,meta_data in enumerate(data_loader):
                outputs = self.net_forward(meta_data, mode='eval')
                meta_data = outputs['meta_data']
                pj2ds_onorg = outputs['pj2d_org'][:, kp2d_mapper].detach().contiguous().cpu().numpy()
                
                ####################################################################################################################################################
                show_items = ['mesh', 'joint_sampler', 'pj2d', 'classify']
                vis_ids = np.arange(1), None
                self.visualizer.save_rendered_img_crowdpose(outputs, outputs['meta_data'], show_items=show_items,\
                        vis_cfg={'settings': ['save_img'], 'vids': vis_ids, 'save_dir':self.result_img_dir, 'save_name':test_iter}, index=test_iter, kp3ds=None)
                
                # show_items = ['pj2d']
                # vis_ids = np.arange(1), None
                # self.visualizer.save_skeleton_img_crowdpose(outputs, outputs['meta_data'], show_items=show_items,\
                #         vis_cfg={'settings': ['save_img'], 'vids': vis_ids, 'save_dir':self.result_img_dir, 'save_name':test_iter}, index=test_iter, kp3ds=None)
                ####################################################################################################################################################

                for batch_idx, (pj2d_onorg, imgpath) in enumerate(zip(pj2ds_onorg, meta_data['imgpath'])):
                    image_id = int(os.path.basename(imgpath).split('.')[0])
                    keypoints = np.concatenate([pj2d_onorg, np.ones((pj2d_onorg.shape[0],1))],1).reshape(-1).tolist()
                    predicted_results.append({'image_id': image_id, 'category_id': 1, 'keypoints':keypoints,'score':1})

                    if imgpath not in vis_dict:
                        vis_dict[imgpath] = []
                    vis_dict[imgpath].append(pj2d_onorg)

                if test_iter%50==0:
                    print(test_iter,'/',len(data_loader))
        with open(results_json_name, 'w') as f:
            json.dump(predicted_results, f)

        gt_file = os.path.join(args().dataset_rootdir,'crowdpose/json/crowdpose_{}.json'.format(set_name))

        cocoGt = COCO(gt_file)
        cocoDt = cocoGt.loadRes(results_json_name)
        cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        # for imgpath, pj2d_list in vis_dict.items():
        #     img_results = cv2.imread(imgpath)
        #     for pj2d_onorg in pj2d_list:
        #         img_results = self.visualizer.draw_skeleton(img_results, pj2d_onorg, bones=constants.crowdpose_connMat, cm=constants.cm_body17)
        #     save_path =  os.path.join(test_save_dir, os.path.basename(imgpath))
        #     cv2.imwrite(save_path,img_results)

    def test_ochuman(self, set_name='test'):
        import json
        from ochumantools.coco import COCO
        from ochumantools.cocoeval import COCOeval
        import cv2 as cv

        predicted_results = []
        test_save_dir = os.path.join(config.project_dir,'results_out/results_ochuman')
        os.makedirs(test_save_dir,exist_ok=True)
        results_json_name = os.path.join(config.project_dir,'results_out/V{}_ochuman_{}_{}.json'.format(self.model_version,set_name,self.backbone))

        self.model.eval()
        kp2d_mapper = constants.joint_mapping(constants.SMPL_ALL_54,constants.OCHuman_19)
        data_loader = self._create_single_data_loader(dataset='ochuman', train_flag = False, split=set_name)
        vis_dict = {}
        with torch.no_grad():
            for test_iter,meta_data in enumerate(data_loader):
                outputs = self.net_forward(meta_data, mode='eval')
                meta_data = outputs['meta_data']
                pj2ds_onorg = outputs['pj2d_org'][:, kp2d_mapper].detach().contiguous().cpu().numpy()   # output keypoints
                # print(np.shape(pj2ds_onorg))    # [N, 17, 2]
                # print(meta_data['full_kp2d'].shape) # [1, 64, 54, 2]
                # print(meta_data['full_kp2d'])
                #print(np.shape(meta_data['image'].cpu().numpy().astype(np.uint8)))  # [N, 512, 512, 3]
                # for i in range(meta_data['image'].shape[0]):
                #     cv.imwrite('/home/dcvl/MK/ROMP_Exp05/image/{}.png'.format(i), cv.cvtColor(meta_data['image'][i].cpu().numpy().astype(np.uint8), cv.COLOR_RGB2BGR))

                # img = cv.cvtColor(meta_data['image'][0].cpu().numpy().astype(np.uint8), cv.COLOR_RGB2BGR)
                # for i in range(np.shape(pj2ds_onorg)[0]):
                #     for j in range(np.shape(pj2ds_onorg)[1]):
                #         # print(type(pj2ds_onorg[i, j, 0]))
                #         # print(pj2ds_onorg[i, j, 0], pj2ds_onorg[i, j, 1])
                #         img = cv.circle(img, (int(pj2ds_onorg[i, j, 0]), int(pj2ds_onorg[i, j, 1])), radius=3, color=(0,255,0), thickness=-1)

                # kp2d = meta_data['full_kp2d'][0, :, kp2d_mapper, :].cpu().numpy()
                # print(np.shape(kp2d))
                # idx = np.where(kp2d!=-2)
                # print(idx)
                # for i in range(64):
                #     for j in range(np.shape(kp2d)[1]):
                #         # print(type(pj2ds_onorg[i, j, 0]))
                #         # print(pj2ds_onorg[i, j, 0], pj2ds_onorg[i, j, 1])
                #         x, y = kp2d[i, j, 0], kp2d[i, j, 1]
                #         if x != -2 and y!= -2:
                #             x, y = int(x*512), int(y*512)
                #             img = cv.circle(img, (x, y), radius=3, color=(255,0,0), thickness=-1)
                        # try:
                        #     img = cv.circle(img, (), radius=3, color=(255,0,0), thickness=-1)
                        #     # print(int(kp2d[i, j, 0]), int(kp2d[i, j, 1]))
                        # except:
                        #     continue

                # cv.imwrite('/home/dcvl/MK/ROMP_Exp05/image/{}.png'.format(test_iter), img)
                ####################################################################################################################################################
                # show_items = ['mesh', 'joint_sampler', 'pj2d', 'classify']
                # vis_ids = np.arange(1), None
                # self.visualizer.save_rendered_img(outputs, outputs['meta_data'], show_items=show_items,\
                #         vis_cfg={'settings': ['save_img'], 'vids': vis_ids, 'save_dir':self.result_img_dir, 'save_name':test_iter}, index=test_iter, kp3ds=None)
                # ####################################################################################################################################################

                # ####################################################################################################################################################
                # show_items = ['pj2d']
                # vis_ids = np.arange(1), None
                # self.visualizer.save_skeleton_img(outputs, outputs['meta_data'], show_items=show_items,\
                #         vis_cfg={'settings': ['save_img'], 'vids': vis_ids, 'save_dir':self.result_img_dir, 'save_name':test_iter}, index=test_iter, kp3ds=None)
                ####################################################################################################################################################

                for batch_idx, (pj2d_onorg, imgpath) in enumerate(zip(pj2ds_onorg, meta_data['imgpath'])):
                    # print(imgpath)
                    image_id = int(os.path.basename(imgpath).split('.')[0])
                    # print(image_id)
                    # print(pj2d_onorg.shape) # [17, 2]
                    keypoints = np.concatenate([pj2d_onorg, np.ones((pj2d_onorg.shape[0],1))],1).reshape(-1).tolist()
                    # print(len(keypoints))   # 51
                    # print(meta_data['full_kp2d'].shape) # [1, 64, 54, 2]
                    predicted_results.append({'image_id': image_id, 'category_id': 1, 'keypoints':keypoints,'score':1})


                    # print(np.shape(meta_data['image'].cpu().numpy().astype(np.uint8)))  # [N, 512, 512, 3]

                    if imgpath not in vis_dict:
                        vis_dict[imgpath] = []
                    vis_dict[imgpath].append(pj2d_onorg)

                if test_iter%50==0:
                    print(test_iter,'/',len(data_loader))
        with open(results_json_name, 'w') as f:
            json.dump(predicted_results, f)

        gt_file = os.path.join(args().dataset_rootdir,'OCHuman/ochuman_coco_format_{}_range_0.00_1.00.json'.format(set_name))
        # gt_file = os.path.join("/home/dcvl/MK/ROMP/dataset/OCHuman/annots_test.npz")

        cocoGt = COCO(gt_file)
        cocoDt = cocoGt.loadRes(results_json_name)
        cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

def get_crowdpose_crowd_index():
    root_dir = "/media/yusun/Extreme SSD/dataset/crowdpose/"
    annot_dict_file = os.path.join(root_dir, 'crowdpose_test_personnum_crowd_Index_dict.npz')
    if not os.path.exists(annot_dict_file):
        import json
        from pycocotools.coco import COCO
        annot_file = os.path.join(root_dir, "json/crowdpose_test.json")
        with open(annot_file,'rb') as f:
            annots = json.load(f)
        annot_dict = {}
        for annot_info in annots['annotations']:
            image_id, person_id, crowd_flag = annot_info['image_id'], annot_info['id'], annot_info['iscrowd']
            if image_id not in annot_dict:
                annot_dict[image_id] = []
            annot_dict[image_id].append(person_id)

        for info_dict in annots['images']:
            image_id = int(info_dict['file_name'].replace('.jpg', ''))
            annot_dict[image_id] = [info_dict['crowdIndex']]+annot_dict[image_id]
        np.savez(annot_dict_file, annots=annot_dict)
    else:
        annot_dict = np.load(annot_dict_file,allow_pickle=True)['annots'][()]
    return annot_dict


def _calc_pn_fps(runtime_dict, person_num_crowd_index_dict):
    person_num_runtime_dict = {}
    for img_id, runtime in runtime_dict.items():
        crowd_index = person_num_crowd_index_dict[img_id][0]
        person_num = len(person_num_crowd_index_dict[img_id]) - 1
        if person_num not in person_num_runtime_dict:
            person_num_runtime_dict[person_num] = []
        person_num_runtime_dict[person_num].append(runtime)
    pn_fps = {}
    for person_num, runtime_list in person_num_runtime_dict.items():
        pn_fps[person_num] = 1./np.array(runtime_list).mean(0)
    for pn in sorted(list(pn_fps.keys())):
        print('{} : {:.2f}'.format(pn, pn_fps[pn]))

    return person_num_runtime_dict, pn_fps

def main():
    with ConfigContext(parse_args(sys.argv[1:])):
        demo = Demo()
        if args().eval_dataset == 'crowdpose_val':
            args().eval_2dpose=True
            demo.test_crowdpose(set_name='val')
        elif args().eval_dataset == 'crowdpose_test':
            args().eval_2dpose=True
            demo.test_crowdpose(set_name='test')
        elif args().eval_dataset == 'cmup':
            demo.test_cmu_panoptic()
        elif args().eval_dataset == 'MuPoTs':
            demo.eval_MuPoTs()
        elif args().eval_dataset == 'runtime':
            demo.test_runtime_crowdpose()
        elif args().eval_dataset == 'ochuman_test':
            demo.test_ochuman(set_name='test')
        elif args().eval_dataset == 'ochuman_val':
            demo.test_ochuman(set_name='val')
        else:
            demo.test_eval()

if __name__ == '__main__':
    main()


