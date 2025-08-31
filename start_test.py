import sys
import os
sys.path.append(os.getcwd())
from inference_pipelines.inference_pipeline_maker import make_inference_pipeline
import argparse
from predict_utils import ScanSegmentation
import shutil
import trimesh
import gen_utils as gu
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference models')
    parser.add_argument('--input_lower_path', default="data/example/109653606_lower.obj", type=str, help = "input mesh path that contain lower teeth.")
    parser.add_argument('--input_upper_path', default="data/example/109653606_upper.obj", type=str, help = "input mesh path that contain upper teeth.")
    parser.add_argument('--cache_path', type=str, default="cache", help = "save intermediate and final results")
    parser.add_argument('--checkpoint_path', default="checkpoints/tgnet_fps" ,type=str,help = "checkpoint path for tgnet_fps.")
    parser.add_argument('--checkpoint_path_bdl', default="checkpoints/tgnet_bdl" ,type=str,help = "checkpoint path for tgnet_bdl.")
    args = parser.parse_args()
    
    # Check input
    if not os.path.exists(args.input_lower_path) or not os.path.exists(args.input_upper_path):
        print("Invalid input mesh path!")
        sys.exit(0)

    # Prepare cache and input
    if os.path.isdir(args.cache_path):
        shutil.rmtree(args.cache_path)
        print("Old cache removed!")
    os.mkdir(args.cache_path)
    print("Cache initialized!")
    
    shutil.copyfile(args.input_lower_path, os.path.join(args.cache_path, "input_lower.obj"))
    shutil.copyfile(args.input_upper_path, os.path.join(args.cache_path, "raw_upper.obj"))
    
    # flip upper teeth
    mesh = trimesh.load_mesh(os.path.join(args.cache_path, "raw_upper.obj"))
    rot_mat = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    rot_mesh = mesh.apply_transform(rot_mat)
    rot_mesh.export(os.path.join(args.cache_path, "input_upper.obj"))
    
    # Prepare model
    model = ScanSegmentation(make_inference_pipeline("tgnet", [args.checkpoint_path+".h5", args.checkpoint_path_bdl+".h5"]))

    # Inference
    stl_path_ls = [os.path.join(args.cache_path, "input_lower.obj"), 
                   os.path.join(args.cache_path, "input_upper.obj")]
    
    for i in range(len(stl_path_ls)):
        print(f"Processing: ", i,":",stl_path_ls[i])
        base_name = os.path.basename(stl_path_ls[i]).split(".")[0]
        model.process(stl_path_ls[i], os.path.join(args.cache_path, os.path.basename(stl_path_ls[i]).replace(".obj", ".json")))
        
    # Visualize lower results
    _, lower_mesh = gu.read_txt_obj_ls(os.path.join(args.cache_path, "input_lower.obj"), ret_mesh=True, use_tri_mesh=True)
    lower_loaded_json = gu.load_json(os.path.join(args.cache_path, "input_lower.json"))
    lower_labels = np.array(lower_loaded_json['labels']).reshape(-1)
    gu.print_3d(gu.get_colored_mesh(lower_mesh, lower_labels)) # color is random
    
    # Visualize upper results
    _, upper_mesh = gu.read_txt_obj_ls(os.path.join(args.cache_path, "raw_upper.obj"), ret_mesh=True, use_tri_mesh=True)
    upper_loaded_json = gu.load_json(os.path.join(args.cache_path, "input_upper.json"))
    upper_labels = np.array(upper_loaded_json['labels']).reshape(-1)
    gu.print_3d(gu.get_colored_mesh(upper_mesh, upper_labels)) # color is random
