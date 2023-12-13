# Integrating Tracking and SLAM with RGB-D Images


**Authors:** [Ziyi Zhou](), [Chih-Chun Yang](), [Cheng-Yen Hsieh]()


Our project addresses limitations in conventional Visual Simultaneous Localization and Mapping (SLAM) algorithms, commonly constrained by assumptions of scene rigidity. Focused on enhancing real-world applicability, particularly in densely populated environments, we propose a Visual SLAM system for RGB-D setups integrating object tracking. Our approach combines segmentation model, like Segment Anything, for precise dynamic object segmentation. To reconstruct occluded static backgrounds, we employ inpainting techniques. The system also incorporates a low-cost tracking module, strategically searching for dynamic moving objects. Extensive quantitative results show that our model significantly outperforms existing visual SLAM system in dynamic scenes.



![image](https://hackmd.io/_uploads/HkPENsIIp.png)

# 0. System Overview

Given multiple RGB-D frames, our system outputs the camera trajectory map of the static part of the scene and inpaint the background occluded by the dynamic objects, allowing the system to effectively address the influence of dynamic elements throughout the process.

![image](https://hackmd.io/_uploads/r11GriUUT.png)
![image](https://hackmd.io/_uploads/SkxQSoU8T.png)



# 1. Prerequisites
We have tested the library in **Ubuntu 16.04**, but it should be easy to compile in other platforms. A powerful computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results.

## C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

## Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at leat 2.4.3. Tested with OpenCV 2.4.11 and OpenCV 3.2**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## DBoW2 and g2o (Included in Thirdparty folder)
We use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

## ROS (optional)
We provide some examples to process the live input of a monocular, stereo or RGB-D camera using [ROS](ros.org). Building these examples is optional. In case you want to use ROS, a version Hydro or newer is needed.

# 2. Segment Anything
<!-- ![image](https://hackmd.io/_uploads/Skc3Bj8Ua.png) -->
<!-- ![image](https://hackmd.io/_uploads/rJ6lLoIL6.png) -->
![image](https://hackmd.io/_uploads/SJ77IsL86.png)



The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or clone the repository locally and install with

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```


# 3. Background Inpainting
After finding moving objects with the segmentation and tracking module, we utilize DDPM to inpaint the occluded background. This procedure produces synthesized frames with only static parts of the scene.
![image](https://hackmd.io/_uploads/BJgjnrjLLa.png)

We used latent diffusion for the inpainting part.
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
git clone https://github.com/CompVis/latent-diffusion.git # or cd latent-diffusion
conda env create -f environment.yaml
conda activate ldm
```


Download the pre-trained weights
```
wget -O models/ldm/inpainting_big/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
```
and sample with
```
python scripts/inpaint.py --indir data/inpainting_examples/ --outdir outputs/inpainting_results
indir should contain images *.png and masks <image_fname>_mask.png like the examples provided in data/inpainting_examples.
```

# 4. Building ORB-SLAM2 library and examples

Clone the repository:
```
git clone https://github.com/raulmur/ORB_SLAM2.git ORB_SLAM2
```

We provide a script `build.sh` to build the *Thirdparty* libraries and *ORB-SLAM2*. Please make sure you have installed all required dependencies (see section 2). Execute:
```
cd ORB_SLAM2
chmod +x build.sh
./build.sh
```

This will create **libORB_SLAM2.so**  at *lib* folder and the executables **mono_tum**, **mono_kitti**, **rgbd_tum**, **stereo_kitti**, **mono_euroc** and **stereo_euroc** in *Examples* folder.


# 5. RGB-D Example

## TUM Dataset

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

2. Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools). We already provide associations for some of the sequences in *Examples/RGB-D/associations/*. You can generate your own associations file executing:

  ```
  python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
  ```

3. Execute the following command. Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder. Change `ASSOCIATIONS_FILE` to the path to the corresponding associations file.

  ```
  ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUMX.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
  ```

