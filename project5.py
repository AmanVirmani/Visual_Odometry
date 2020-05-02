import cv2
import glob
import random
import numpy as np
from utils import ReadCameraModel, UndistortImage


def estimateFundamentalMatrix(points_a, points_b):
    
    points_num = points_a.shape[0]
    A = []
    B = np.ones((points_num, 1))

    cu_a = np.sum(points_a[:, 0]) / points_num
    cv_a = np.sum(points_a[:, 1]) / points_num

    s = points_num / np.sum(
        ((points_a[:, 0] - cu_a)**2 + (points_a[:, 1] - cv_a)**2)**(1 / 2))
    T_a = np.dot(
        np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]]),
        np.array([[1, 0, -cu_a], [0, 1, -cv_a], [0, 0, 1]]))

    points_a = np.array(points_a.T)
    points_a = np.append(points_a, B)

    points_a = np.reshape(points_a, (3, points_num))
    points_a = np.dot(T_a, points_a)
    points_a = points_a.T

    cu_b = np.sum(points_b[:, 0]) / points_num
    cv_b = np.sum(points_b[:, 1]) / points_num

    s = points_num / np.sum(
        ((points_b[:, 0] - cu_b)**2 + (points_b[:, 1] - cv_b)**2)**(1 / 2))
    T_b = np.dot(
        np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]]),
        np.array([[1, 0, -cu_b], [0, 1, -cv_b], [0, 0, 1]]))

    points_b = np.array(points_b.T)
    points_b = np.append(points_b, B)

    points_b = np.reshape(points_b, (3, points_num))
    points_b = np.dot(T_b, points_b)
    points_b = points_b.T

    for i in range(points_num):
        u_a = points_a[i, 0]
        v_a = points_a[i, 1]
        u_b = points_b[i, 0]
        v_b = points_b[i, 1]
        A.append([
            u_a * u_b, v_a * u_b, u_b, u_a * v_b, v_a * v_b, v_b, u_a, v_a, 1
        ])

    _, _, v = np.linalg.svd(A)
    F = v[-1]

    F = np.reshape(F, (3, 3)).T
    F = np.dot(T_a.T, F)
    F = np.dot(F, T_b)

    F = F.T
    U, S, V = np.linalg.svd(F)
    S = np.array([[S[0], 0, 0], [0, S[1], 0], [0, 0, 0]])
    F = np.dot(U, S)
    F = np.dot(F, V)

    F = F / F[2, 2]

    return F

def GetInliersRANSAC(matches_a, matches_b):
    
    matches_num = matches_a.shape[0]
    Best_count = 0

    for iter in range(500):
        sampled_idx = np.random.randint(0, matches_num, size=8)
        F = estimateFundamentalMatrix(matches_a[sampled_idx, :],
                                      matches_b[sampled_idx, :])
        in_a = []
        in_b = []
        #ind = []
        update = 0
        for i in range(matches_num):
            matches_aa = np.append(matches_a[i, :], 1)
            matches_bb = np.append(matches_b[i, :], 1)
            error = np.dot(matches_aa, F.T)
            error = np.dot(error, matches_bb.T)
            if abs(error) < 0.005:
                in_a.append(matches_a[i, :])
                in_b.append(matches_b[i, :])
                #ind.append(indices[i])
                update += 1

        if update > Best_count:
            Best_count = update
            best_F = F
            inliers_a = in_a
            inliers_b = in_b
            #inlier_index = ind

    inliers_a = np.array(inliers_a)
    inliers_b = np.array(inliers_b)
    # inlier_index = np.array(inlier_index)

    return best_F, inliers_a, inliers_b

class Cells:
    def __init__(self):
        self.pts = list()
        self.pairs = dict()

    def rand_pt(self):
        return random.choice(self.pts)


if __name__=="__main__":
    img_path = "Oxford_dataset/stereo/centre"
    images = sorted(glob.glob(img_path+"/*.png"))
    images = images[19:]
    print(len(images))
    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    input_img = cv2.imread(images[0], 0)
    bgr_img = cv2.cvtColor(input_img, cv2.COLOR_BAYER_GR2BGR)
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel.ReadCameraModel('Oxford_dataset/model/')
    undistorted_img = UndistortImage.UndistortImage(bgr_img, LUT)
    gray_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray_img, None)
    y_bar, x_bar = np.array(gray_img.shape)/8

    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel.ReadCameraModel('Oxford_dataset/model/')
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K_inv = np.linalg.inv(K)

    for i in range(1, len(images)):
        input_img_next = cv2.imread(images[i], 0)
        bgr_img_next = cv2.cvtColor(input_img_next, cv2.COLOR_BAYER_GR2BGR)
        undistorted_img_next = UndistortImage.UndistortImage(bgr_img_next, LUT)
        gray_img_next = cv2.cvtColor(undistorted_img_next, cv2.COLOR_BGR2GRAY)

        keypoints_next, descriptors_next = orb.detectAndCompute(gray_img_next, None)
        #gray_img = gray_img[200:650,:]
        #gray_img_next = gray_img_next[200:650,:]
        matches = bf.match(gray_img, gray_img_next)
        matches = sorted(matches, key=lambda x: x.distance)

        features, features_next = [], []
        for m in matches:
            features.append(keypoints[m.queryIdx].pt)
            features_next.append(keypoints_next[m.trainIdx].pt)


        matches_img = cv2.drawMatches(gray_img, keypoints, gray_img_next, keypoints_next, matches[:10], None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


        # img = cv2.drawKeypoints(gray_img, keypoints_next, None)
        cv2.namedWindow("keypoints", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("keypoints",(600,600))
        cv2.imshow("keypoints", matches_img)

        cv2.waitKey(1000)

        # -----> Initialise the grids and points array variables <----- #
        point_correspondence_cf = np.zeros((len(matches),2))
        point_correspondence_nf = np.zeros((len(matches),2))
        grid = np.empty((8,8), dtype=object)
        grid[:,:] = Cells()

        for i, match in enumerate(matches):
            j = int(keypoints[match.queryIdx].pt[0]/x_bar)
            k = int(keypoints[match.queryIdx].pt[1]/y_bar)
            grid[j,k].pts.append(keypoints[match.queryIdx].pt)
            grid[j,k].pairs[keypoints[match.queryIdx].pt] = keypoints_next[match.trainIdx].pt

            point_correspondence_cf[i] = keypoints[match.queryIdx].pt[0], keypoints[match.queryIdx].pt[1]
            point_correspondence_nf[i] = keypoints_next[match.trainIdx].pt[0], keypoints_next[match.trainIdx].pt[1]

        cv_F, mask = cv2.findFundamentalMat(point_correspondence_cf, point_correspondence_nf)				    # Estimate the Fundamental matrix #
        #F = estimateFundamentalMatrix(point_correspondence_cf, point_correspondence_nf)
        F, inliers_a, inliers_b = GetInliersRANSAC(point_correspondence_cf, point_correspondence_nf)
        print("F\n",F)
        print("CV_F \n",cv_F)
        gray_img = gray_img_next
        keypoints = keypoints_next
        pass

    cv2.destroyAllWindows()
