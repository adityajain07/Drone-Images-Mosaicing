# Image Stitching for Drone mapping
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
* The images are stored sequentially in the dataset

### Methodology
Below is a higher-level overview of the mosaicing technique implemented:

1. Choose first two images in the dataset
1. Find matching keypoints between the two images
1. Calculate the homography using the matched key- points
1. Using this homography, one image is warped to be in the same frame as the other and a new image of all black pixels is created which can fit both images in the new frame
1. Repeat step 2 with the current mosaic and the next image, until all the images in the dataset are covered


### Stitched Mosaics
The algorithm was tested on two datasets: agricultural farm and of Chandigarh city (the dataset has been personally collected). Below shows some of the mosaics (the mosaics below have been resized for easy viewing. Please see the original mosaics in their respective folders):
## Farm Mosaic
![Farm Mosaic](https://github.com/adityajain07/Drone-Images-Mosaicing/blob/master/Display_FarmM.jpg)

## City Mosaic
![City Mosaic](https://github.com/adityajain07/Drone-Images-Mosaicing/blob/master/Display_CityM.jpg)

# Future Work
As it is evident that the quality of mosaic is deteriorating with increase in the number of stitched images. That is because the homography error accumulates with each stitch. Need to implement Bundle Adjustment to reduce the propagating error.

