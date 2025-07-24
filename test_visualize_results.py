import sys
import os
from trimesh import PointCloud
sys.path.append(os.getcwd())
from glob import glob
import gen_utils as gu
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import copy
import argparse

parser = argparse.ArgumentParser(description='Inference models')
parser.add_argument('--mesh_path', default="G:/tooth_seg/main/all_datas/chl/3D_scans_per_patient_obj_files/013FHA7K/013FHA7K_lower.obj", type=str)
parser.add_argument('--pred_json_path', type=str, default="test_results/013FHA7K_lower.json")
args = parser.parse_args()


pred_loaded_json = gu.load_json(args.pred_json_path)
pred_labels = np.array(pred_loaded_json['labels']).reshape(-1)

_, mesh = gu.read_txt_obj_ls(args.mesh_path, ret_mesh=True, use_tri_mesh=True)
gu.print_3d(gu.get_colored_mesh(mesh, pred_labels)) # color is random