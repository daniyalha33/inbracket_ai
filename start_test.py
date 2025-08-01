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
import open3d as o3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference models')
    parser.add_argument('--input_lower_path', default="data/example/109653606_lower.obj", type=str, help="input mesh path that contains lower teeth.")
    parser.add_argument('--input_upper_path', default="data/example/109653606_upper.obj", type=str, help="input mesh path that contains upper teeth.")
    parser.add_argument('--cache_path', type=str, default="cache", help="save intermediate and final results")
    parser.add_argument('--output_path', type=str, default="output", help="path to save final output .obj files")
    parser.add_argument('--checkpoint_path', default="checkpoints/tgnet_fps", type=str, help="checkpoint path for tgnet_fps.")
    parser.add_argument('--checkpoint_path_bdl', default="checkpoints/tgnet_bdl", type=str, help="checkpoint path for tgnet_bdl.")
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
    
    # Flip upper teeth
    mesh = trimesh.load_mesh(os.path.join(args.cache_path, "raw_upper.obj"))
    rot_mat = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    rot_mesh = mesh.apply_transform(rot_mat)
    rot_mesh.export(os.path.join(args.cache_path, "input_upper.obj"))
    
    # Prepare model
    model = ScanSegmentation(make_inference_pipeline("tgnet", [args.checkpoint_path + ".h5", args.checkpoint_path_bdl + ".h5"]))

    # Inference
    stl_path_ls = [os.path.join(args.cache_path, "input_lower.obj"), 
                   os.path.join(args.cache_path, "input_upper.obj")]
    
    for i in range(len(stl_path_ls)):
        print(f"Processing: ", i, ":", stl_path_ls[i])
        base_name = os.path.basename(stl_path_ls[i]).split(".")[0]
        model.process(stl_path_ls[i], os.path.join(args.cache_path, os.path.basename(stl_path_ls[i]).replace(".obj", ".json")))
        
    # Save the results for lower jaw
    _, lower_mesh = gu.read_txt_obj_ls(os.path.join(args.cache_path, "input_lower.obj"), ret_mesh=True, use_tri_mesh=True)
    lower_loaded_json = gu.load_json(os.path.join(args.cache_path, "input_lower.json"))
    lower_labels = np.array(lower_loaded_json['labels']).reshape(-1)
    
    # Save the colored lower mesh
    colored_lower_mesh = gu.get_colored_mesh(lower_mesh, lower_labels)
    output_lower_path = os.path.join(args.output_path, "colored_input_lower.obj")
    os.makedirs(args.output_path, exist_ok=True)
    o3d.io.write_triangle_mesh(output_lower_path, colored_lower_mesh)
    print(f"Lower jaw output saved to {output_lower_path}")
    
    # Save the results for upper jaw
    _, upper_mesh = gu.read_txt_obj_ls(os.path.join(args.cache_path, "raw_upper.obj"), ret_mesh=True, use_tri_mesh=True)
    upper_loaded_json = gu.load_json(os.path.join(args.cache_path, "input_upper.json"))
    upper_labels = np.array(upper_loaded_json['labels']).reshape(-1)
    
    # Save the colored upper mesh
    colored_upper_mesh = gu.get_colored_mesh(upper_mesh, upper_labels)
    output_upper_path = os.path.join(args.output_path, "colored_input_upper.obj")
    o3d.io.write_triangle_mesh(output_upper_path, colored_upper_mesh)
    print(f"Upper jaw output saved to {output_upper_path}")
