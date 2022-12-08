# D3VO

16-833 Robot Localization and Mapping course project

Implementation of D3VO: Deep Depth, Deep Pose and Deep Uncertainty for Monocular Visual Odometry (https://arxiv.org/pdf/2003.01060.pdf)



## Setup 

Use Python 3.8.10 or earlier.

Compile and install g2opy.
```
cd g2opy
mkdir build
cd build
cmake ..
make -j8
cd ..
python setup.py install
cd ..
```

Download trained DepthNet and PoseNet weights.

https://drive.google.com/drive/folders/176fuEVP1BVQlKQNXCp3wQE_kBK_ogOCT?usp=sharing


Install Python packages.

```
python3 -m pip install torch torchvision numpy matplotlib
```

(Optional) Download and install the KITTI dataset following the instructions: 


Run D3VO on an input video (.mp4). 
```
python3 main.py video_path.mp4 weights_directory_path --gt optional_ground_truth_txt_path --out output_dir_path
```




## Resources

