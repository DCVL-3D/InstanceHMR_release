import plotly
import plotly.graph_objects as go
from plotly.graph_objs.scatter import Line
from plotly.subplots import make_subplots
import plotly.express as px
import ipywidgets
from ipywidgets.widgets import Layout, HBox, VBox
from ipywidgets.embed import embed_minimal_html
import pandas as pd
import sys, os
import constants
import config
from config import args

def convert_3dpose_to_line_figs(poses, bones, pred_color='goldenrod', gt_color='red'):
    figs = []
    items_name = ["x","y","z",'class','joint_name']
    if bones.max()==13:
        joint_names = constants.LSP_14_names
    elif bones.max()==23:
        joint_names = constants.SMPL_24_names
    for batch_inds, (pred, real) in enumerate(zip(*poses)):
        pose_dict, color_maps = {}, {}
        for bone_inds in bones:
            si, ei = bone_inds
            bone_name = '{}-{}'.format(joint_names[si], joint_names[ei])
            pose_dict['p_'+bone_name+'_s'] = [*pred[si],'p_'+bone_name, joint_names[si]]
            pose_dict['p_'+bone_name+'_e'] = [*pred[ei],'p_'+bone_name, joint_names[ei]]
            color_maps['p_'+bone_name] = pred_color
            pose_dict['r_'+bone_name+'_s'] = [*real[si],'r_'+bone_name, joint_names[si]]
            pose_dict['r_'+bone_name+'_e'] = [*real[ei],'r_'+bone_name, joint_names[ei]]
            color_maps['r_'+bone_name] = gt_color
        pred_real_pose_df = pd.DataFrame.from_dict(pose_dict,orient='index',columns=items_name)
        pose3d_fig = px.line_3d(pred_real_pose_df, x="x", y="y", z="z", color='class', color_discrete_map=color_maps)#, text='joint_name'
        figs.append(pose3d_fig)
    return figs    

def write_to_html(img_names, plot_dict, vis_cfg):
    containers = []
    raw_layout = Layout(overflow_x='scroll',border='2px solid black',width='1900px',height='',
                    flex_direction='row',display='flex')        
    for inds, img_name in enumerate(img_names):
        Hboxes = []
        for item in list(plot_dict.keys()):
            if inds >= len(plot_dict[item]['figs']):
                continue
            fig = plot_dict[item]['figs'][inds]
            fig['layout'] = {"title":{"text":img_name.replace(args().dataset_rootdir, '')}}
            Hboxes.append(go.FigureWidget(fig))
        containers.append(HBox(Hboxes,layout=raw_layout))
    all_figs = VBox(containers)
    save_name = os.path.join(vis_cfg['save_dir'],vis_cfg['save_name']+'.html')
    embed_minimal_html(save_name, views=[all_figs], title=vis_cfg['save_name'], drop_defaults=True)
    ipywidgets.Widget.close_all()
    del all_figs, containers, Hboxes

def convert_image_list(images):
    figs = []
    for img in images:
        figs.append(px.imshow(img))
    return figs

if __name__ == '__main__':
    import numpy as np
    convert_3dpose_to_line_figs([np.random.rand(18).reshape((2,3,3)),np.random.rand(18).reshape((2,3,3))],np.array([[0,1],[1,2]]))