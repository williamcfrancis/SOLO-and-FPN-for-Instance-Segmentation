# SOLO and FPN for Instance Segmentation
## Overview
Instance segmentation can be thought of as a combination of object detection and semantic segmentation, the former of which I have implemented in [this repository](https://github.com/williamcfrancis/YOLOv3-from-scratch-for-Object-Detection). A visulization of this relationship can be seen in the following figure.

<div><img src="https://github.com/LukasZhornyak/CIS680_files/raw/main/HW3/fig1.png" width=500/></div>

Here I implement an instance segmentation framework known as SOLO (Segmenting Object by LOcations). In a similar manner to YOLO, SOLO produces mask predictions on a dense grid. This means that, unlike many other segmenation frameworks (e.g. Mask-RCNN), SOLO directly predicts the segmentation mask without proposing bounding box locations. An visual summary of SOLO can be seen in the following figures

<div><img src="https://github.com/LukasZhornyak/CIS680_files/raw/main/HW3/fig2.png" width=200/></div>

<div><img src="https://github.com/LukasZhornyak/CIS680_files/raw/main/HW3/fig3.png" width=600/></div>

These dense predictions are produced at several different scales using a Feature Pyramid Network (FPN). Using the last few layers of the backbone, I pass the higher level features from the deeper layers back up to larger features scales using lateral connections, shown in the figure below.

<div><img src="https://github.com/LukasZhornyak/CIS680_files/raw/main/HW3/fig4.png" width=300/></div>
