import cv2
import glob
import random
import numpy as np
from utils import ReadCameraModel, UndistortImage, functions, DrawCameras
from utils.utils import *
import matplotlib.pyplot as plt


if __name__=="__main__":
    # laod image data
    img_path = "../Oxford_dataset/stereo/centre"
    images = glob.glob(img_path+"/*.png")
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel.ReadCameraModel('../Oxford_dataset/model/')

    # Define K matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K_inv = np.linalg.inv(K)

    # Initialize feature matching
    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(crossCheck=True)
    #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    FLANN_INDEX_LSH = 6
    index_params=dict(algorithm=FLANN_INDEX_LSH,
                   table_number=6, # 12
                   key_size=12,     # 20
                   multi_probe_level=1) #2
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Initialize plot window
    fig = plt.figure('Figure 1',figsize=(7,5))
    # fig.suptitle('Project 5 - Visual Odometry')
    ax1 = fig.add_subplot(111)
    ax1.set_title('Visual Odometry Map')

    # Data for the 30th frame as first frame
    start_frame = 50
    input_img = cv2.imread(images[start_frame], 0)
    bgr_img = cv2.cvtColor(input_img, cv2.COLOR_BAYER_GR2BGR)
    undistorted_img = UndistortImage.UndistortImage(bgr_img, LUT)
    gray_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray_img, None)
    y_bar, x_bar = np.array(gray_img.shape)/8

    # Create VisualOdometry class object
    func = functions.VisualOdometry()

    # Define Matrices
    oldHomogeneousMatrix = np.identity(4)
    oldTranslationMatrix = np.array([[0, 0, 0, 1]])
    oldTranslationMatrix = oldTranslationMatrix.T

    depth = []

    for i in range(start_frame+1, len(images)):
    #for i in range(start_frame + 1, start_frame+1000): #len(images)):
        print("Processing frame {}".format(i))
        # Data for current frame
        input_img = cv2.imread(images[i], 0)
        bgr_img = cv2.cvtColor(input_img, cv2.COLOR_BAYER_GR2BGR)
        undistorted_img = UndistortImage.UndistortImage(bgr_img, LUT)
        gray_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray_img, None)

        # Data for the next frame
        input_img_next = cv2.imread(images[i+1], 0)
        bgr_img_next = cv2.cvtColor(input_img_next, cv2.COLOR_BAYER_GR2BGR)
        undistorted_img_next = UndistortImage.UndistortImage(bgr_img_next, LUT)
        gray_img_next = cv2.cvtColor(undistorted_img_next, cv2.COLOR_BGR2GRAY)
        keypoints_next, descriptors_next = orb.detectAndCompute(gray_img_next, None)

        # getting feature matches
        matches = flann.knnMatch(descriptors, descriptors_next, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        good_matches, points, points_next = [], [], []
        for j, (m, n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[j]=[1,0]
                good_matches.append(m)
                points.append(keypoints[m.queryIdx].pt)
                points_next.append(keypoints_next[m.trainIdx].pt)

        points = np.int32(points)
        points_next = np.int32(points_next)
        if len(good_matches) == 0:
            continue
        F, mask = cv2.findFundamentalMat(points, points_next, cv2.FM_RANSAC)
        if F is None:
            continue
        #bestF, inliers_a, inliers_b = GetInliersRANSAC(points, points_next)

        points = points[mask.ravel() == 1]
        points_next = points_next[mask.ravel() == 1]

        E = get_essential_matrix(K, F[:3,:])

        Rset, Cset = ExtractCameraPose(E, K)
        X_set = []

        for n in range(4):
            X1= LinearTriangulation(K, np.zeros((3,1)), np.identity(3), Cset[n].T, Rset[n],
                                    points, points_next)
                                    #np.float(points), np.float(points_next))
            X_set.append(X1)

        X, R, t = DisambiguateCameraPose(Cset, Rset, X_set)

        ## OpenCV code
        #retval, R, t, mask = cv2.recoverPose(E, points, points_next, K)

        newHomogeneousMatrix = HomogeneousMatrix(R, t)
        oldHomogeneousMatrix = oldHomogeneousMatrix @ newHomogeneousMatrix

        pose = oldHomogeneousMatrix @ oldTranslationMatrix

        plt.scatter(pose[0][0], -pose[2][0], color='r')
        depth.append([pose[0][0], -pose[2][0]])
        if i%50 == 0:
            plt.savefig("./Output/"+str(i)+".png")

    plt.show()
        #   draw_params = dict(matchColor=(0,255,0),
        #                      singlePointColor=(255,0,0),
        #                      matchesMask=matchesMask,
        #                      flags=0)

        #   # uncomment to visualize matching points
        #   img3 = cv2.drawMatchesKnn(gray_img, keypoints, gray_img_next, keypoints_next, matches, None, **draw_params)
        #   #cv2.imshow('img', img3)
        #   #cv2.waitKey(1)

        #   #matches_img = cv2.drawMatches(gray_img, keypoints, gray_img_next, keypoints_next, good_matches, None,
        #   #                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        #   #img = cv2.drawKeypoints(gray_img, keypoints_next, None)
        #   #cv2.imshow("keypoints", matches_img)
        #   #cv2.waitKey(0)

        #   # -----> Initialise the grids and points array variables <----- #
        #   point_correspondence_cf = np.zeros((len(good_matches), 2))
        #   point_correspondence_nf = np.zeros((len(good_matches), 2))
        #   grid = np.empty((8, 8), dtype=object)
        #   grid[:, :] = functions.Cells()

        #   for n_match, match in enumerate(good_matches):
        #       j = int(keypoints[match.queryIdx].pt[0]/x_bar)
        #       k = int(keypoints[match.queryIdx].pt[1]/y_bar)
        #       grid[j,k].pts.append(keypoints[match.queryIdx].pt)
        #       grid[j,k].pairs[keypoints[match.queryIdx].pt] = keypoints_next[match.trainIdx].pt

        #       point_correspondence_cf[n_match] = keypoints[match.queryIdx].pt[0], keypoints[match.queryIdx].pt[1]
        #       point_correspondence_nf[n_match] = keypoints_next[match.trainIdx].pt[0], keypoints_next[match.trainIdx].pt[1]

        #   F, mask = cv2.findFundamentalMat(point_correspondence_cf, point_correspondence_nf)  	#Estimate the Fundamental matrix #
        #   if F is None:
        #       continue
        #   E = func.estimate_Essential_Matrix(K, F)

        #   pose = func.camera_pose(K, E)  # Estimate the Posses Matrix #

        #   # -----> Estimate Rotational and Translation points <----- #
        #   flag = 0
        #   Translation = np.zeros((3, 1))
        #   Rotation = np.eye(3)
        #   for p in range(4):
        #       R = pose['R' + str(p + 1)]
        #       T = pose['C' + str(p + 1)]
        #       Z = func.extract_Rot_and_Trans(R, T, point_correspondence_cf, point_correspondence_nf, K)
        #       if flag < Z: flag, reg = Z, str(p + 1)

        #   R = pose['R'+reg]
        #   t = pose['C'+reg]
        #   if t[2] < 0: t = -t
        #x_cf = Translation[0]
        #z_cf = Translation[2]
        #Translation += Rotation.dot(t)
        #Rotation = R.dot(Rotation)
        #x_nf = Translation[0]
        #z_nf = Translation[2]

        #ax1.plot([-x_cf, -x_nf],[z_cf, z_nf], 'o')
        #if i%50 == 0:
        #    plt.pause(1)
        #    plt.savefig("./Output/"+str(i)+".png")
        #else:
        #    plt.pause(0.001)

        #print('# -----> Frame No:'+str(i), '<----- #')

        #pass
    #cv2.destroyAllWindows()
