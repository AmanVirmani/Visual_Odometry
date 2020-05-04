import cv2
import os
import argparse
import numpy as np
import glob
import random
import math
import matplotlib.pyplot as plt

from Oxford_dataset.ReadCameraModel import ReadCameraModel
from Oxford_dataset.UndistortImage import UndistortImage

path = "Oxford_dataset/stereo/centre/*.png"
imagesPath = sorted(glob.glob(path))


fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('./Oxford_dataset/model/')

# Camera Calibration Matrix of the model
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
sift = cv2.xfeatures2d.SIFT_create()

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default="./Oxford_dataset/", help='Folder of dataset')  
    Args = Parser.parse_args()
    DataPath = Args.DataPath

    # Initial camera Pose is considered as an identity matrix
    H = np.identity(4)  
    origin = np.array([[0, 0, 0, 1]]).T
    
    trajPts = [] 
    
    # skippping first 20 frames
    for ind in range(20,len(imagesPath)-1):  

        print(ind)

        img1 = cv2.imread(imagesPath[ind], 0)  
        rgbImg1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
        undistortImage1 = UndistortImage(rgbImg1, LUT)  
        gray1 = cv2.cvtColor(undistortImage1, cv2.COLOR_BGR2GRAY)

        img2 = cv2.imread(imagesPath[ind+1], 0)  
        rgbImg2 = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)
        undistortImage2 = UndistortImage(rgbImg2, LUT)  
        gray2 = cv2.cvtColor(undistortImage2, cv2.COLOR_BGR2GRAY)

        # Cropping the area of interest 
        # This is used because the car will be there in all the frames which will cause issue in finding the tranformation in two consecutive frames.. 
        # pixels associated will car will lead to no rotation and translation between two consecutive frames
        gray1Crop = gray1[200:650, 0:1280]
        gray2Crop = gray2[200:650, 0:1280]

        
        kp1, des1 = sift.detectAndCompute(gray1Crop, None)
        kp2, des2 = sift.detectAndCompute(gray2Crop, None)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        points_a = []
        points_b = []

        for i, (m, n) in enumerate(matches):
            if m.distance < 1 * n.distance:
                good.append(m)
                points_b.append(kp2[m.trainIdx].pt)
                points_a.append(kp1[m.queryIdx].pt)

        points_a = np.int32(points_a)
        points_b = np.int32(points_b)
                 
        F, mask = cv2.findFundamentalMat(points_a, points_b, cv2.FM_RANSAC)
        mask = mask.astype(bool).flatten()
        E = np.dot(K.T, np.dot(F, K))
        _, R, t, _ = cv2.recoverPose(E, points_a[mask], points_b[mask], K)

        # Transforming from current frame to next frame
        Rt = np.column_stack((R, t)) 
        matHomogeneous = np.vstack((Rt, np.array([0, 0, 0, 1])))
        H = H @ matHomogeneous
        p = H @ origin 

        trajPts.append([p[0][0], -p[2][0]])
        print(trajPts[ind-20])
        plt.scatter(p[0][0], -p[2][0], color='b')
    np.save('trajInbuilt',trajPts)
    plt.show()

if __name__ == '__main__':
    main()
