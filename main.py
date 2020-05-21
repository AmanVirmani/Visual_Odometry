import cv2
import numpy as np
from utils.ReadCameraModel import ReadCameraModel
from utils.UndistortImage import UndistortImage
import matplotlib.pyplot as plt
from utils.visual_odometry import *


if __name__ == "__main__":
    fx, fy, cx, cy, G_camera_image, LUT=ReadCameraModel("../Oxford_dataset/model")
    # Define K matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    N = 3871
    camera_centre = np.zeros((4,1))
    camera_centre[3] = 1
    new_centre = []
    count_final = 0
    H_new = np.identity(4)
    C_origin = np.zeros((3, 1))
    R_origin = np.identity(3)
    final_points = []

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=1500)
    bf = cv2.BFMatcher()

    for i in range(20, N):
        print(i)
        frame = cv2.imread('../Visual-Odometry/FRAMES/{}.jpg'.format(i))
        frame_next = cv2.imread('../Visual-Odometry/FRAMES/{}.jpg'.format(i+1))

        frame = cv2.rectangle(frame,(np.float32(50),np.float32(np.shape(frame)[0])),(np.float32(1250),np.float32(800)),(0,0,0),-1)
        frame_next = cv2.rectangle(frame_next,(np.float32(50),np.float32(np.shape(frame_next)[0])),(np.float32(1250),np.float32(800)),(0,0,0),-1)

        keypoints, descriptors = sift.detectAndCompute(frame, None)
        keypoints_next, descriptors_next = sift.detectAndCompute(frame_next, None)

        matches = bf.match(descriptors,descriptors_next)

        U = []
        V = []
        for m in matches:
            U.append(keypoints[m.queryIdx].pt)
            V.append(keypoints_next[m.trainIdx].pt)
        U = np.array(U)
        V = np.array(V)

        F = getFundamentalMatrix(U, V, 100, 0.001)
        E = getEssentialMatrix(F, K)

        R, C, X = get_camera_pose(E, U, V, K)

        H_final = np.hstack((R, C))
        H_final = np.vstack((H_final,[0,0,0,1]))

        x_old = H_new[0][3]
        z_old = H_new[2][3]
        H_new = H_new@H_final
        x_new = H_new[0][3]
        z_new = H_new[2][3]

        plt.plot([-x_old, -x_new], [z_old, z_new], 'ro')
        final_points.append([x_old, x_new, -z_old, -z_new])
        fname = "Output/plotPoints.npy"
        plt.pause(0.01)
        if i % 50 == 0:
            plt.savefig("Output/plot_custom-{:05d}.png".format(i))
            np.save(fname, final_points)
    np.save(fname, final_points)
