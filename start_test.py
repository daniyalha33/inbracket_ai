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
    parser.add_argument('--checkpoint_path', default="checkpoints/tgnet_fps", type=str, help="checkpoint path for tgnet_fps.")
    parser.add_argument('--checkpoint_path_bdl', default="checkpoints/tgnet_bdl", type=str, help="checkpoint path for tgnet_bdl.")
    args = parser.parse_args()

    # Check input paths
    if not os.path.exists(args.input_lower_path) or not os.path.exists(args.input_upper_path):
        print("Invalid input mesh path!")
        sys.exit(0)

    # Prepare cache and input
    if os.path.isdir(args.cache_path):
        shutil.rmtree(args.cache_path)
        print("Old cache removed!")
    os.mkdir(args.cache_path)
    print("Cache initialized!")

    # Copy input meshes to cache
    shutil.copyfile(args.input_lower_path, os.path.join(args.cache_path, "input_lower.obj"))
    shutil.copyfile(args.input_upper_path, os.path.join(args.cache_path, "raw_upper.obj"))
    
    # Ensure both files are correctly copied
    if not os.path.exists(os.path.join(args.cache_path, "input_lower.obj")):
        print("Error: Lower jaw mesh not copied correctly!")
        sys.exit(1)
    if not os.path.exists(os.path.join(args.cache_path, "raw_upper.obj")):
        print("Error: Upper jaw mesh not copied correctly!")
        sys.exit(1)

    print("Meshes copied successfully!")

    # Prepare model
    model = ScanSegmentation(make_inference_pipeline("tgnet", [args.checkpoint_path + ".h5", args.checkpoint_path_bdl + ".h5"]))

    # Inference paths for both jaws
    stl_path_ls = [os.path.join(args.cache_path, "input_lower.obj"),
                   os.path.join(args.cache_path, "raw_upper.obj")]

    # Debugging: Print the exact paths being used
    print(f"Paths to process: {stl_path_ls}")

    # Process lower jaw
    print("Processing lower jaw...")
    model.process(stl_path_ls[0], os.path.join(args.cache_path, "input_lower.json"))

    # Process upper jaw
    print("Processing upper jaw...")
    model.process(stl_path_ls[1], os.path.join(args.cache_path, "input_upper.json"))

    # Check if the upper jaw JSON file is generated correctly
    if os.path.exists(os.path.join(args.cache_path, "input_upper.json")):
        print("Upper jaw JSON generated!")
    else:
        print("Error: Upper jaw JSON not generated!")
        sys.exit(1)

    # Visualize lower results and save as colored .obj
    _, lower_mesh = gu.read_txt_obj_ls(os.path.join(args.cache_path, "input_lower.obj"), ret_mesh=True, use_tri_mesh=True)
    lower_loaded_json = gu.load_json(os.path.join(args.cache_path, "input_lower.json"))
    lower_labels = np.array(lower_loaded_json['labels']).reshape(-1)
    colored_lower_mesh = gu.get_colored_mesh(lower_mesh, lower_labels)

    # Save the colored mesh to an .obj file
    o3d.io.write_triangle_mesh(os.path.join(args.cache_path, "colored_input_lower.obj"), colored_lower_mesh)

    # Visualize upper results and save as colored .obj
    _, upper_mesh = gu.read_txt_obj_ls(os.path.join(args.cache_path, "raw_upper.obj"), ret_mesh=True, use_tri_mesh=True)
    upper_loaded_json = gu.load_json(os.path.join(args.cache_path, "input_upper.json"))
    if upper_loaded_json:  # Check if the JSON file is populated
        upper_labels = np.array(upper_loaded_json['labels']).reshape(-1)
        colored_upper_mesh = gu.get_colored_mesh(upper_mesh, upper_labels)

        # Save the colored mesh to an .obj file
        o3d.io.write_triangle_mesh(os.path.join(args.cache_path, "colored_input_upper.obj"), colored_upper_mesh)
    else:
        print("Error: Upper jaw JSON is empty!")
        sys.exit(1)
