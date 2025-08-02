



## Official Dataset

- Dataset consists of dental mesh obj files and corresponding ground truth json files.
- One can also download the training split data in [google drive](https://drive.google.com/drive/u/1/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby)(their models are based on this data).

  - After download and unzip these zip files, merge `3D_scans_per_patient_obj_files.zip` and `3D_scans_per_patient_obj_files_b2.zip`. The parent path of these obj files is `data_obj_parent_directory`.
  - Apply the same to the ground truth json files(`ground-truth_labels_instances.zip` and `ground-truth_labels_instances_b2.zip`. The parent path of these json files is `data_json_parent_directory`).
  - The directory structure of the data should look like below..

    ```
    --data_obj_parent_directory
    ----00OMSZGW
    ------00OMSZGW_lower.obj
    ------00OMSZGW_upper.obj
    ----0EAKT1CU
    ------0EAKT1CU_lower.obj
    ------0EAKT1CU_upper.obj
    and so on..

    --data_json_parent_directory
    ----00OMSZGW
    ------00OMSZGW_lower.json
    ------00OMSZGW_upper.jsno
    ----0EAKT1CU
    ------0EAKT1CU_lower.json
    ------0EAKT1CU_upper.json
    and so on..
    ```

## Custom Dataset

- If using dental mesh data: and want to use custom dataset use diffusion based augmentation which will create mesh data and increase you dataset.

  - Adhere to the data name format(casename_upper.obj or casename_lower.obj).

# Inference

- All of the checkpoint files for each model are private please contact through repo issue place if you need any help regarding to checkpoints or how to train the model.
- Inference with tgnet

  ```
  python start_test.py \
   --input_lower_path obj_path \
   --input_upper_path obj_path \
   --cache_path folder_path \
   --checkpoint_path your/tgnet_fps_checkpoint_path \
   --checkpoint_path_bdl your/tgnet_bdl_checkpoint_path
  ```

- Predicted results are saved in cache path like below. It has the same format as the ground truth json file.
  ```
  --cache_path
  ----input_lower.obj
  ----raw_upper.obj
  ----input_upper.obj
  ----input_lower.json
  ----input_upper.json
  ```

  #For libraray please use the offical library page and use how to install that.

# Installation

- Installation tested with CUDA >= 12.5, PyTorch >=2.0 and Python 3.9.19.
- Create virtual environment using conda

  ```
  conda create -n 3dsegment python=3.9
  conda activate 3dsegment
  ```


- Install following packages in the environment
  ```
  pip install --ignore-installed PyYAML
  pip install open3d
  pip install multimethod
  pip install termcolor
  pip install trimesh
  pip install easydict
  ```
