# Image Stitching for Drone Mapping
This project contains code for stitching large number of images using classical computer vision techniques for application in drone mapping.

## Setup python environment
Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and prepare a python environment using the following steps:

Build a new conda environment
```bash
conda create -n image_stitching python=3.8
```

Install additional libraries using pip:

```bash
python3 -m pip install -r requirements.txt
```

Activate the conda environment:
```bash
conda activate image_stitching
```

## Approach
### Assumptions
* There is atleast 70% overlap (both front and sideways) between the images while the data was collected
* The images are numbered sequentially in the dataset

### Methodology
Below is a high-level overview of the image stitching algorithm:

1. Feature detection using SIFT
1. Feature matching using FLANN-basd matcher
1. Homography estimation using DLT and RANSAC
1. Image warping


## Stitched Mosaics
The algorithm is tested on two datasets: agricultural farm and an urban area. Below images shows two examples mosaics:
### Farm Mosaic
![Farm Mosaic](https://github.com/adityajain07/Drone-Images-Mosaicing/blob/master/Display_FarmM.jpg)

### City Mosaic
![City Mosaic](https://github.com/adityajain07/Drone-Images-Mosaicing/blob/master/Display_CityM.jpg)

## Future Work
Two possible improvements over our work is to use bundle adjustment to reduce the cumulative error issue and blending to remove noticeable seams at the stitching
location.

