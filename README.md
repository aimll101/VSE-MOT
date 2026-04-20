# VSE-MOT: Multi-Object Tracking in Low-Quality Video Scenes Guided by Visual Semantic Enhancement

## Introduction

**Abstract.** Current multi-object tracking (MOT) algorithms typically overlook issues inherent in low-quality videos, leading to significant degradation in tracking performance when confronted with real-world image deterioration. Therefore, advancing the application of MOT algorithms in real-world low-quality video scenarios represents a critical and meaningful endeavor. To address the challenges posed by low-quality scenarios, inspired by vision-language models, this paper proposes a Visual Semantic Enhancement-guided Multi-Object Tracking framework (VSE-MOT). Specifically, we first design a tri-branch architecture that leverages a vision-language model to extract global visual semantic information from images and fuse it with query vectors. Subsequently, to further enhance the utilization of visual semantic information, we introduce the Multi-Object Tracking Adapter (MOT-Adapter) and the Visual Semantic Fusion Module (VSFM). The MOT-Adapter adapts the extracted global visual semantic information to suit multi-object tracking tasks, while the VSFM improves the efficacy of feature fusion. Through extensive experiments, we validate the effectiveness and superiority of the proposed method in low-quality video scenarios, while maintaining robust performance in conventional scenarios.


## Main Results

### LQDanceTrack

| **HOTA** | **DetA** | **AssA** | **MOTA** | **IDF1** |                                           **URL**                                           |
| :------: | :------: | :------: | :------: | :------: | :-----------------------------------------------------------------------------------------: |
|   59.9   |   73.7    |   48.9   |   84.1   |   62.3   | [model](https://pan.baidu.com/s/1O_C7IeLl-JHLi2H47nzFNA?pwd=twpk) |

### DanceTrack

| **HOTA** | **DetA** | **AssA** | **MOTA** | **IDF1** |                                           **URL**                                           |
| :------: | :------: | :------: | :------: | :------: | :-----------------------------------------------------------------------------------------: |
|   64.1   |   77.2   |   53.4   |   86.4   |   65.9   | [model](https://pan.baidu.com/s/1O_C7IeLl-JHLi2H47nzFNA?pwd=twpk) |


### LQDanceTrack

| **HOTA** | **DetA** | **AssA** | **MOTA** | **IDF1** |                                           **URL**                                           |
| :------: | :------: | :------: | :------: | :------: | :-----------------------------------------------------------------------------------------: |
|   58.5   |   62.3    |   54.9   |   68.2   |   64.1   | [model](https://pan.baidu.com/s/1yiVU__CdQF51vKmJJmIvsg?pwd=qxbt) |

### DanceTrack

| **HOTA** | **DetA** | **AssA** | **MOTA** | **IDF1** |                                           **URL**                                           |
| :------: | :------: | :------: | :------: | :------: | :-----------------------------------------------------------------------------------------: |
|   67.8   |   69.6   |   66.0   |   73.2   |   71.7   | [model](https://pan.baidu.com/s/1yiVU__CdQF51vKmJJmIvsg?pwd=qxbt) |

### Requirements

* Install pytorch using conda (optional)

    ```bash
    conda create -n motrv2 python=3.7
    conda activate motrv2
    conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.2 -c pytorch
    ```

* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

## Usage

### Dataset preparation

1. Download YOLOX detection from [here](https://drive.google.com/file/d/1cdhtztG4dbj7vzWSVSehLL6s0oPalEJo/view?usp=share_link).
2. Please download [DanceTrack](https://dancetrack.github.io/) and [CrowdHuman](https://www.crowdhuman.org/) and unzip them as follows:

```
/data/Dataset/mot
├── crowdhuman
│   ├── annotation_train.odgt
│   ├── annotation_trainval.odgt
│   ├── annotation_val.odgt
│   └── Images
├── DanceTrack
│   ├── test
│   ├── train
│   └── val
├── det_db_motrv2.json
```

You may use the following command for generating crowdhuman trainval annotation:

```bash
cat annotation_train.odgt annotation_val.odgt > annotation_trainval.odgt
```

### Training

You may download the coco pretrained weight from [Deformable DETR (+ iterative bounding box refinement)](https://github.com/fundamentalvision/Deformable-DETR#:~:text=config%0Alog-,model,-%2B%2B%20two%2Dstage%20Deformable), and modify the `--pretrained` argument to the path of the weight. Then training MOTR on 8 GPUs as following:

```bash 
./tools/train.sh configs/motrv2.args
```

### Inference on DanceTrack Test Set

```bash
# run a simple inference on our pretrained weights
./tools/simple_inference.sh ./motrv2_dancetrack.pth

# Or evaluate an experiment run
# ./tools/eval.sh exps/motrv2/run1

# then zip the results
zip motrv2.zip tracker/ -r
```

## Acknowledgements

- [MOTRv2](https://github.com/megvii-research/MOTRv2)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [DanceTrack](https://github.com/DanceTrack/DanceTrack)
