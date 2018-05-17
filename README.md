# About

In recent times, Unmanned Aerial Vehicles (UAVs), more commonly known as drones, have been finding applications in a range of fields. They are used to capture aerial imagery in sectors like precision agriculture, infrastructure, disaster management, city planning etc. The first crucial step after collecting aerial imagery is to generate a mosaic (or orthomosaic) from the imagery dataset or in a layman’s language a digital map. <br/>

This project contains the code written from scratch to generate mosaics out of drone imagery. This code can also be used for panorama generation.  <br/>

A single-clicked image has a limited field of view (FOV), we need to stitch together several image stills to form a mosaic to increase the FOV. Image mosaicing is a very popular way to obtain a wide FOV image of a scene. The basic idea is to capture images as a camera moves and stitch these images together to obtain a single larger image. The moving camera on the drone can be used to capture more slices of an area. These multiple image slices can be mosaiced together to give an entire view of a scene.

# Approach
## Assumptions
* There is atleast 70% overlap (both front and sideways) between the images while the data was collected
* The images are stored sequentially in the dataset

## Methodology
Below is a higher-level overview of the mosaicing technique implemented:

1. Choose first two images in the dataset
1. Find matching keypoints between the two images
1. Calculate the homography using the matched key- points
1. Using this homography, one image is warped to be in the same frame as the other and a new image of all black pixels is created which can fit both images in the new frame
1. Repeat step 2 with the current mosaic and the next image, until all the images in the dataset are covered


# Stitched Mosaics
The algorithm was tested on two datasets: agricultural farm and of Chandigarh city (the dataset has been personally collected). Below shows some of the mosaics (the mosaics below have been resized for easy viewing. Please see the original mosaics in their respective folders):
## Farm Mosaic
![Farm Mosaic](https://github.com/adityajain07/Drone-Images-Mosaicing/blob/master/Display_FarmM.jpg)

## City Mosaic
![City Mosaic](https://github.com/adityajain07/Drone-Images-Mosaicing/blob/master/Display_CityM.jpg)

# Future Work
As it is evident that the quality of mosaic is deteriorating with increase in the number of stitched images. That is because the homography error accumulates with each stitch. Need to implement Bundle Adjustment to reduce the propagating error.

