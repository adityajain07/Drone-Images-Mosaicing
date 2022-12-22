#!/usr/bin/env python
# coding: utf-8

"""
Authors      : Aditya Jain & Bhavishey Thapar
Date started : November 22, 2022
About        : AER1515 Project; image stitching for drone imagery
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import pickle

def draw_keypoints(img, kp):
    """
    function to draw keypoints on an image

    Args:
        img : image
        kp  : keypoints

    Returns:
        img : image with keypoints drawn on it
    """
    img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

    cv2.imshow('Keypoints', img_kp)
    cv2.waitKey(0)
    
# Function to draw matches on an image
def draw_matches (img1, kp1, img2, kp2, matches):
    """
    function to draw matches on an image

    Args:
        img1    : first image
        kp1     : keypoints in first image
        img2    : second image
        kp2     : keypoints in second image
        matches : matches between keypoints in first and second image

    Returns:
        img3    : image with matches drawn on it
    """
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    # cv2.imshow('Matches', img3)


def find_reprojection_error(src_pts, dst_pts, M):
    """
    function to find reprojection error

    Args:
        src_pts : source points
        dst_pts : destination points
        M       : homography matrix

    Returns:
        reprojection_error : average reprojection error"""

    error = 0
    for i in range(len(src_pts)):
        src_pt = src_pts[i]
        dst_pt = dst_pts[i]
        src_pt = np.append(src_pt, 1)
        transformed_pt = M @ src_pt
        transformed_pt = transformed_pt/transformed_pt[2]
        transformed_pt = transformed_pt[0:2]
        error += np.linalg.norm(transformed_pt - dst_pt)
    reprojection_error = error/len(src_pts)

    return reprojection_error

def stitch_images(img1, img2, H):
    """
    function for warping/stitching two images using the homography matrix H
    
    Args:
        img1  : first image, or source image
        img2  : second image, that needs to be mapped to the frame of img1
        H     : homography matrix that maps img2 to img1        
        
    Returns:
        output_img: img2 warped to img1 using H
    """
    
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    points_1      = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    temp_points   = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    points_2      = cv2.perspectiveTransform(temp_points, H)
    points_concat = np.concatenate((points_1, points_2), axis=0)    

    [x_min, y_min]   = np.int32(points_concat.min(axis=0).ravel() - 0.5)
    [x_max, y_max]   = np.int32(points_concat.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min,-y_min]
    H_translation    = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    frame_size = output_img.shape
    new_image  = img2.shape
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
    
    origin_r = int(points_2[0][0][1])
    origin_c = int(points_2[0][0][0])
    
    # if the origin of projected image is out of bounds, then mapping to ()
    if origin_r < 0:
        origin_r = 0
    if origin_c < 0:
        origin_c = 0
        
    # Clipping the new image, if it's size is more than the frame    
    if new_image[0] > frame_size[0]-origin_r:
        img2 = img2[0:frame_size[0]-origin_r,:]
        
    if new_image[1] > frame_size[1]-origin_c:
        img2 = img2[:,0:frame_size[1]-origin_c]    
            
    output_img[origin_r:new_image[0]+origin_r, origin_c:new_image[1]+origin_c] = img2    
    
    return output_img


def build_mosaic(raw_image_list, save_mosaic_dir, mosaic_base_name, 
                 num_features=10000, reproj_thresh=2.0, flag_save=False):
    """
    main function for image stitching 
    
    Args:
        raw_image_list   : sorted list of raw image paths to be stitched
        save_mosaic_dir  : directory to save the mosaic
        mosaic_base_name : base name of the stitched mosaic 
        num_featues      : number of keypoints to extract from feature detector; optional; default=1000
        reproj_thresh    : reprojection error threshold in pixels; optional; default=5.0
        flag_save        : flag to check whether to save mosaic on the disk
        
        
    Returns:
        avg_repro_error : list of average reprojection error
    """
    
    avg_repro_error = []  # list of average reprojection error
    matches_list    = []  # list of number of good matches at every stage
    sift            = cv2.SIFT_create(num_features)
    img_count       = 0
    
    # starting out with first image
    first_image     = cv2.imread(raw_image_list.pop(0))
    height, width   = first_image.shape[:2]
    first_image     = cv2.resize(first_image, (int(width/4), int(height/4)))
    stitched_mosaic = first_image
    img_count       += 1
    if flag_save:
        cv2.imwrite(save_mosaic_dir + mosaic_base_name + str(img_count) + '.png', stitched_mosaic)
    
    
    while raw_image_list:
        image_name = raw_image_list.pop(0)
        img_count  += 1
        print(f'Stitching for image {image_name}')
        image = cv2.imread(image_name)          
        height, width = image.shape[:2]        
        image = cv2.resize(image, (int(width/4), int(height/4)))

        # Find the features
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(stitched_mosaic,cv2.COLOR_BGR2GRAY),None)  
        kp2, des2 = sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),None)

        # # Feature matching      
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        # Store all good matches as per Lowe's ratio test
        good       = []
        all_points = []
        for m,n in matches:
            if m.distance < 0.3*n.distance:
                good.append(m)                
            all_points.append(m)        
        matches_list.append(len(good))

        # draw_matches(final_mosaic, kp1, image, kp2, good)

        # Find homography
        dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)               
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
        
        # Apply homography to current image and obtain the resultant mosaic
        stitched_mosaic = stitch_images(stitched_mosaic, image, H)
        if flag_save:
            cv2.imwrite(save_mosaic_dir + mosaic_base_name + str(img_count) + '.png', stitched_mosaic)        
 
        # Find reprojection error
        cur_error = find_reprojection_error(src_pts, dst_pts, H)
        print(f'Current reprojection error {cur_error}')
        avg_repro_error.append(cur_error)
        
    return avg_repro_error

## User Input ##
num_imgs_to_use  = 4
raw_img_dir      = '/Users/adityajain/Downloads/raw_images/*.JPG'
save_mosaic_dir  = '/Users/adityajain/Downloads/mosaics/'
error_data_dir   = '/Users/adityajain/Downloads/error_data/'
mosaic_base_name = 'city_imgs-'
flag_save        = False                     # whether to save mosaic on disc
dataset          = 'city'
ftr_detector     = 'sift'
num_keypoints    = 10000
error_filename   = dataset+'_'+ftr_detector+'_'+str(num_keypoints)+'.pkl'
################

raw_image_list   = sorted(glob.glob(raw_img_dir))[:num_imgs_to_use]

if __name__=='__main__':
    error = build_mosaic(raw_image_list, save_mosaic_dir, mosaic_base_name, num_features=num_keypoints, flag_save=flag_save)
    print(f'Error list {error}')
    with open(error_data_dir + error_filename, 'wb') as f:
        pickle.dump(error, f)

print(error)
# plt.plot(error)
# plt.xlabel('Number of images')
# plt.ylabel('Average reprojection error')
# plt.title('Average reprojection error vs Number of images')
# plt.show()
# cv2.imshow('mosaic', cv2.imread(mosaic_name))
# cv2.waitKey(0)