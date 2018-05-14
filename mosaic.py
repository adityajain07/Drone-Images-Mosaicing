
# coding: utf-8

# In[122]:


### Authors: Aditya Jain and Taneea S Agrawaal (IIIT-Delhi) #####
### Topic: Mosaicing of Drone Imagery ###
### Start Date: 10th March, 2018 ###

import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import time
import os
# import math

FilepathFirst = os.path.abspath("/Users/adityaj/Downloads/dji/DJI_0001.JPG")
FilepathAll = os.path.abspath("/Users/adityaj/Downloads/dji/*.JPG")
img1 = cv2.imread(FilepathFirst)
tic = time.clock()
test = []


def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)
    

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min,-y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])
    
    print "Min and Max:", x_min, y_min, x_max, y_max
    print "Translation Distance:", translation_dist    

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    FrameSize = output_img.shape
    NewImage = img2.shape
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
    
    OriginR = int(list_of_points_2[0][0][1])
    OriginC = int(list_of_points_2[0][0][0])
    
    # if the origin of projected image is out of bounds, then mapping to ()
    if OriginR < 0:
        OriginR = 0
    if OriginC < 0:
        OriginC = 0
        
    # Clipping the new image, if it's size is more than the frame    
    if NewImage[0] > FrameSize[0]-OriginR:
        img2 = img2[0:FrameSize[0]-OriginR,:]
        
    if NewImage[1] > FrameSize[1]-OriginC:
        img2 = img2[:,0:FrameSize[1]-OriginC]    
            
    print "Image 2 Magic size:", img2.shape
    output_img[OriginR:NewImage[0]+OriginR, OriginC:NewImage[1]+OriginC] = img2    
    
    return output_img




# In[123]:


### Function to give entire mosaic

images = sorted(glob.glob(FilepathAll))    # for reading images
n = 10000;          # no of features to extract

def giveMosaic(FirstImage, no):
    
    EList = []      # this stores the average reprojection error
    ImgList = []          # No of images stitched
    Matches = []    # this stores the number of good matches at every stage
    i = 1
    
    heightM, widthM = FirstImage.shape[:2]
    FirstImage = cv2.resize(FirstImage, (widthM / 4, heightM / 4))
    RecMosaic = FirstImage
    
    for name in images[1:]:
        
        print name
        image = cv2.imread(name) 
        
        
        sift = cv2.SIFT(no)
        
        ######## Resize them (they are too big)
        # Get their dimensions        
        height, width = image.shape[:2]
#         print heightM, widthM, height, width        
        
        image = cv2.resize(image, (width / 4, height / 4))
        ###########################


        # Find the features
        kp1, des1 = sift.detectAndCompute(RecMosaic,None)   # kp are the keypoints, des are the descriptors
        kp2, des2 = sift.detectAndCompute(image,None)
        
        ########### FLANN Matcher  ##########      
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        #############
        
        
        # store all the good matches as per Lowe's ratio test.
        good = []
        allPoints = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
                
            allPoints.append(m)
        
        Matches.append(len(good))
        print "Good_Matches:", len(good)
        ##################################
        
        
        #### Finding the homography #########
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        all_src_pts = np.float32([ kp1[m.queryIdx].pt for m in allPoints ]).reshape(-1,1,2)
        all_dst_pts = np.float32([ kp2[m.trainIdx].pt for m in allPoints ]).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,6.0)
        ###################################
        
        #### Finding the euclidean distance error ####
        list1 = np.array(src_pts)    
        list2 = np.array(dst_pts)
        list2 = np.reshape(list2, (len(list2), 2))
        ones = np.ones(len(list1))    
        TestPoints = np.transpose(np.reshape(list1, (len(list1), 2)))
        print "Length:", np.shape(TestPoints), np.shape(ones)
        TestPointsHom = np.vstack((TestPoints, ones))  
        print "Homogenous Points:", np.shape(TestPointsHom)
    
        projectedPointsH = np.matmul(M, TestPointsHom)  # projecting the points in test image to collage image using homography matrix    
        projectedPointsNH = np.transpose(np.array([np.true_divide(projectedPointsH[0,:], projectedPointsH[2,:]), np.true_divide(projectedPointsH[1,:], projectedPointsH[2,:])]))
        
        print "list2 shape:", np.shape(list2)
        print "NH Points shape:", np.shape(projectedPointsNH)
        print "Raw Error Vector:", np.shape(np.linalg.norm(projectedPointsNH-list2, axis=1))
        Error = int(np.sum(np.linalg.norm(projectedPointsNH-list2, axis=1)))
        print "Total Error:", Error
        AvgError = np.divide(np.array(Error), np.array(len(list1)))
        print "Average Error:", AvgError
        
        ##################       
        
        i+=1

        RecMosaic = warpImages(RecMosaic, image, M)
        cv2.imwrite("FinalMosaicTemp.jpg", RecMosaic)
        print i
        
        EList.append(AvgError)
        ImgList.append(i)
        
        if i==40:
            break
        
        
    cv2.imwrite("FinalMosaic.jpg", RecMosaic)
    return EList, ImgList, Matches


ErrorList, ImgNumbers, GoodMatches = giveMosaic(img1, 10000)
toc = time.clock()
print toc-tic



# In[ ]:


# plt.figure(1)

# plt.subplot(211)
# plt.plot(ImgNumbers, ErrorList)
# plt.ylabel('Avg reproj error/feature')
# plt.xlabel('No. of images stitched')
# plt.title('Reproject Error v/s no of images stitched (ORB_10K)')

# plt.subplot(212)
# plt.plot(ImgNumbers, GoodMatches)
# plt.ylabel('No of good matches')
# plt.xlabel('No. of images stitched')
# plt.title('No of good matches v/s no of images stitched (ORB_10K)')

# plt.subplots_adjust(hspace=1)


# In[124]:




