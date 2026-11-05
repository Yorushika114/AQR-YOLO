# AQR-YOLO: Battery Surface Defect Detection

This repository contains the local research implementation based on Ultralytics YOLO11 and the extended CoatingVision dataset used in the manuscript. The implementation uses the historical names MCSR and SA-TAL in some code and figure files; the manuscript refers to these components as AMEM and QC-TAL. RAP denotes route-aware pruning. Consult the configuration and training scripts for the implemented behavior.

## Dataset

`dataset/data.yaml` describes three classes: Pinhole, Surface_Crack and Delamination. The split contains 7,520 training, 775 validation and 775 test images. The validation and test sets were split from the previous validation set; they should not be described as independent newly collected samples. Labels follow YOLO detection format. `split_manifest.csv` records file hashes and the original split.

The source dataset is CoatingVision: Sampath et al., A defect dataset for electrode coating manufacturing, Scientific Data, DOI: https://doi.org/10.1038/s41597-025-06419-1. Please consult its original terms and the applicable Ultralytics AGPL-3.0 license before redistributing or using the data or code.

## Setup and code

Install a compatible Python/PyTorch environment, then install this project with `pip install -e .`. The model configuration is `ultralytics/cfg/models/11/yolo11n-adaptive-defect.yaml`; model, assignment, and pruning implementations are in `ultralytics/nn/modules/battery_next.py`, `ultralytics/utils/tal.py`, and `tools/prune_route_aware.py`. Training and evaluation utilities are in `tools/`. Some historical scripts contain local paths and must be configured for your environment.

Model weights and training outputs are not included. The dataset YAML resolves relative to its own folder. For example: `yolo detect train model=ultralytics/cfg/models/11/yolo11n-adaptive-defect.yaml data=dataset/data.yaml epochs=100 imgsz=640`. This example starts a training run; it is not a claim of full manuscript reproducibility.
