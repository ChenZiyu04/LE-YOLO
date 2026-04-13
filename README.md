# LE-YOLO: Task-driven curve enhancement with attention-guided ghostnet for low-light object detection
### 1. Description
This repository contains the official implementation and data associated with the paper "LE-YOLO: Task-driven curve enhancement with attention-guided ghostnet for low-light object detection". This project is categorized as an AI Application.

The model integrates a task-driven curve enhancement module (based on Zero-DCE++) with an attention-guided GhostNet backbone to optimize object detection in low-light conditions.

Zenodo Repository: [https://doi.org/10.5281/zenodo.19034332](https://doi.org/10.5281/zenodo.19354824)

### 2. Dataset Information
The research utilizes two primary low-light datasets:

- ExDark Dataset:

  - Source: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

  - Classes: Contains 12 categories: Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table.

Config: Defined in exdark.yaml and final_split_base.yaml.

- DarkFace Dataset:

    - Description: A specialized dataset for face detection in low-light environments.

    - Source: https://github.com/dataset-ninja/dark-face

    - Classes: Single class: "face".

    - Config: Defined in final_darkface.yaml.

### 3. Methodology & Code Information
- Core Modules
    - zerodce_module.py: Implements the Zero-DCE++ architecture, focusing on depthwise separable convolutions to estimate enhancement curves.

    - custom_modules.py: Implements C3Ghost_CBAM, a custom module that integrates the C3Ghost block with CBAM to refine feature maps.

    - new_custom_models.py: Acts as a bridge to integrate the Zero-DCE enhancement layer directly into the YOLO model pipeline.

- Network Architectures
    - leyolo_cbam.yaml: The main proposed architecture, featuring Zero-DCE++ curve enhancement, PSA, and C3Ghost_CBAM modules.

- Baseline/Comparison Models: We provide several YOLO configuration files used for performance benchmarking:

    - yolov5.yaml: Standard YOLOv5 architecture.

    - yoloe-v8.yaml: YOLOv8-based architecture using C2f modules and YOLOEDetect head.

    - yolov10n.yaml: YOLOv10-Nano implementation with PSA and C2fCIB modules.

    - yolov10n_ghost.yaml: A modified YOLOv10 using GhostConv and C3Ghost in the detection head.

    - yolo11.yaml: YOLO11 configuration using C3k2 and C2PSA modules.

    - yolo12.yaml: YOLO12 configuration utilizing A2C2f attention modules.

### 4. Requirements
Ensure the following dependencies are installed:

- Python 3.x

- PyTorch & torchvision

- ultralytics

- pathlib, argparse, shutil

### 5. Usage Instructions
Data Splitting
Before training, use train_val_split.py to organize your raw images and labels:
```
Bash
python train_val_split.py --datapath /path/to/dataset --train_pct 0.8
```
This script randomly allocates files into train and validation folders.

Training & Inference
To train the proposed LE-YOLO model:

Configure the path in exdark.yaml or final_darkface.yaml.

Execute the training command :
```
Bash
yolo train model=leyolo_cbam.yaml data=exdark.yaml
```

### 6. Citation
Citation: If you use this code or the LE-YOLO model in your research, please cite our PeerJ Computer Science paper.
