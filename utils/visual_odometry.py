import cv2
import numpy as np
import glob
from utils.ReadCameraModel import ReadCameraModel
from utils.UndistortImage import UndistortImage
import matplotlib.pyplot as plt
import os
import scipy
import scipy.optimize as opt


def get_camera_pose(E, U, V, K):
    R, C = ExtractCameraPose(E, K)

    count_numbers = []
    count_numbers.append(count_positive_depth_points(R[0], C[0], U, V, K))
    count_numbers.append(count_positive_depth_points(R[1], C[1], U, V, K))
    count_numbers.append(count_positive_depth_points(R[2], C[2], U, V, K))
    count_numbers.append(count_positive_depth_points(R[3], C[3], U, V, K))

    count_numbers = np.array(count_numbers)
    index = np.argmax(count_numbers[:,0])

    return R[index], C[index], count_numbers[index][1].T


def getFundamentalMatrix(U, V, max_iters=500, threshold=0.008):
    Inliers_UN = []
    max_inliers = 0
    M = len(U)

    ## get fundamental matrix
    for it in range(max_iters):
        X1,X2 = get_random_matches(U,V)
        F_r = get_normalized_fundamental_matrix(X1,X2)
        Inliers_U = []
        Inliers_V = []
        Inliers = 0
        for j in range(len(U)):
            U1 = np.array([U[j][0], U[j][1], 1]).reshape(1, -1)
            V1 = np.array([V[j][0], V[j][1], 1]).reshape(1, -1)

            epiline1 = F_r @ U1.T
            epiline2 = F_r.T @ V1.T
            error_bottom = epiline1[0]**2 + epiline1[1]**2 + epiline2[0]**2 + epiline2[1]**2

            error = ((V1 @ F_r @ U1.T)**2)/error_bottom

            if error[0, 0] < threshold:
                Inliers += 1
                Inliers_U.append([U[j][0], U[j][1]])
                Inliers_V.append([V[j][0], V[j][1]])

        if max_inliers < Inliers:
            max_inliers = Inliers
            Inliers_UN = Inliers_U
            Inliers_VN = Inliers_V
            F = F_r
    return F


def get_random_matches(points,points_next):
    random_indices = np.random.choice(len(points), size=8, replace=False)
    return points[random_indices, :], points_next[random_indices, :]


def get_normalized_fundamental_matrix(points1, points2):

    dist1 = np.sqrt((points1[:,0]- np.mean(points1[:,0]))**2 + (points1[:,1]- np.mean(points1[:,1]))**2)
    dist2 = np.sqrt((points2[:,0]- np.mean(points2[:,0]))**2 + (points2[:,1]- np.mean(points2[:,1]))**2)

    m_dist1 = np.mean(dist1)
    m_dist2 = np.mean(dist2)

    scale1 = np.sqrt(2)/m_dist1
    scale2 = np.sqrt(2)/m_dist2

    t1 = np.array([[scale1, 0, -scale1*np.mean(points1[:,0])],[0, scale1, -scale1*np.mean(points1[:,1])],[0, 0, 1]])
    t2 = np.array([[scale2, 0, -scale2*np.mean(points2[:,0])],[0, scale2, -scale2*np.mean(points2[:,1])],[0, 0, 1]])


    U_x = (points1[:,0] - np.mean(points1[:,0]))*scale1
    U_y = (points1[:,1] - np.mean(points1[:,1]))*scale1
    V_x = (points2[:,0] - np.mean(points2[:,0]))*scale2
    V_y = (points2[:,1] - np.mean(points2[:,1]))*scale2

    A = np.zeros((len(U_x),9))

    for i in range(len(U_x)):

        A[i] = np.array([U_x[i]*V_x[i], U_y[i]*V_x[i], V_x[i], U_x[i]*V_y[i], U_y[i]*V_y[i], V_y[i], U_x[i], U_y[i], 1])

    U,S,V = np.linalg.svd(A)
    V = V.T
    F = V[:,-1].reshape(3,3)

    # Enforcing rank 2 constraint on estimated F matrix
    Uf, Sf, Vf = np.linalg.svd(F)
    SF = np.diag(Sf)
    SF[2,2] = 0

    F = Uf @ SF @ Vf
    F = t2.T @ F @ t1
    F = F/F[2, 2]

    return F


def getEssentialMatrix(F, K):
    E = K.T @ F @ K

    U, S, V_T = np.linalg.svd(E)
    E = np.dot(U, np.dot(np.diag([1, 1, 0]), V_T))
    E = E/np.linalg.norm(E)
    return E


def ExtractCameraPose(E, K):
    """
    Args:
        E (array): Essential Matrix
        K (array): Intrinsic Matrix

    Returns:
        arrays: set of Rotation and Camera Centers
    """
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R = []
    C = []
    P = []
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    C.append(U[:, 2].reshape(-1, 1))
    C.append(-U[:, 2].reshape(-1, 1))
    C.append(U[:, 2].reshape(-1, 1))
    C.append(-U[:, 2].reshape(-1, 1))

    for i in range(len(R)):
        if np.linalg.det(R[i]) < 0:
            R[i] = -R[i]
            C[i] = -C[i]
    return R, C


def LinearTriangulation(P1, P2, pts1, pts2):
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))

    x = np.zeros((3, len(pts1)))

    pt1 = np.array(-np.eye(2, 3))
    pt2 = np.array(-np.eye(2, 3))

    for i in range(len(pts1)):
        pt1[:, 2] = pts1[i, :]
        pt2[:, 2] = pts2[i, :]

        A[0:2, :] = pt1.dot(P1[0:3, 0:3])
        A[2:4, :] = pt2.dot(P2[0:3, 0:3])

        b[0:2, :] = pt1.dot(P1[0:3, 3:4])
        b[2:4, :] = pt2.dot(P2[0:3, 3:4])

        cv2.solve(A, b, x[:, i:i+1], cv2.DECOMP_SVD)

    return x

def count_positive_depth_points(R, C, pts1, pts2, K):

    P1 = np.eye(3, 4)
    P2 = np.hstack((R , C.reshape(-1, 1)))

    X1_p = np.array(LinearTriangulation(P1, P2, np.array(pts1), np.array(pts2)))

    X1 = np.vstack((X1_p,np.ones((1,len(pts1))))).reshape(-1, 4)

    X1 = np.divide(X1,np.array([X1[:, 3], X1[:, 3], X1[:, 3], X1[:, 3]]).T)

    X1 = np.sum(P2@X1.T > 0)

    return X1, X1_p


def NonLinearTriangulation(K, x1, x2, X_init, R1, C1, R2, C2):
    sz = x1.shape[0]
    # print(R2)
    # print(C2)
    assert x1.shape[0] == x2.shape[0] == X_init.shape[0], "2D-3D corresspondences have different shape "
    X = np.zeros((sz, 3))

    init = X_init.flatten()
    #     Tracer()()
    optimized_params = opt.least_squares(
        fun=minimizeFunction,
        x0=init,
        method="dogbox",
        args=[K, x1, x2, R1, C1, R2, C2])

    X = np.reshape(optimized_params.x, (sz, 3))

    return X


def minimizeFunction(init, K, x1, x2, R1, C1, R2, C2):
    """Summary

    Args:
        init (TYPE): Description
        K (TYPE): Description
        x1 (TYPE): Description
        x2 (TYPE): Description
        R1 (TYPE): Description
        C1 (TYPE): Description
        R2 (TYPE): Description
        C2 (TYPE): Description

    Returns:
        TYPE: Description
    """
    sz = x1.shape[0]
    X = np.reshape(init, (sz, 3))

    I = np.identity(3)
    C2 = np.reshape(C2, (3, -1))

    X = np.hstack((X, np.ones((sz, 1))))

    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    error1 = 0
    error2 = 0
    error = []

    u1 = np.divide((np.dot(P1[0, :], X.T).T), (np.dot(P1[2, :], X.T).T))
    v1 = np.divide((np.dot(P1[1, :], X.T).T), (np.dot(P1[2, :], X.T).T))
    u2 = np.divide((np.dot(P2[0, :], X.T).T), (np.dot(P2[2, :], X.T).T))
    v2 = np.divide((np.dot(P2[1, :], X.T).T), (np.dot(P2[2, :], X.T).T))

    #     print(u1.shape,x1.shape)
    assert u1.shape[0] == x1.shape[0], "shape not matched"

    error1 = ((x1[:, 0] - u1) + (x1[:, 1] - v1))
    error2 = ((x2[:, 0] - u2) + (x2[:, 1] - v2))
    #     print(error1.shape)
    error = sum(error1, error2)

    return sum(error)


def reprojError(CQ, K, X, x):
    """Function to calculate reprojection error

    Args:
        K (TYPE): intrinsic matrix
        X (TYPE): 3D points
        x (TYPE): 2D points

    Returns:
        TYPE: Reprojection error
    """
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    C = CQ[0:3]
    R = CQ[3:7]
    C = C.reshape(-1, 1)
    r_temp = scipy.spatial.transform.Rotation.from_quat([R[0], R[1], R[2], R[3]])
    R = r_temp.as_dcm()

    P = np.dot(np.dot(K, R), np.hstack((np.identity(3), -C)))

    # print("P",P.shape, X3D.shape)
    u_rprj = (np.dot(P[0, :], X.T)).T / (np.dot(P[2, :], X.T)).T
    v_rprj = (np.dot(P[1, :], X.T)).T / (np.dot(P[2, :], X.T)).T
    e1 = x[:, 0] - u_rprj
    e2 = x[:, 1] - v_rprj
    e = e1 + e2

    return sum(e)


def NonLinearPnP(X, x, K, R0, C0):

    q_temp = scipy.spatial.transform.Rotation.from_dcm(R0)
    Q0 = q_temp.as_quat()
    # reprojE = reprojError(C0, K, X, x)

    CQ = [C0[0], C0[1], C0[2], Q0[0], Q0[1], Q0[2], Q0[3]]
    assert len(CQ) == 7, "length of init in nonlinearpnp not matched"
    optimized_param = opt.least_squares(
        fun=reprojError, method="dogbox", x0=CQ, args=[K, X, x])
    Cnew = optimized_param.x[0:3]
    assert len(Cnew) == 3, "Translation Nonlinearpnp error"
    R = optimized_param.x[3:7]
    r_temp = scipy.spatial.transform.Rotation.from_quat([R[0], R[1], R[2], R[3]])
    Rnew = r_temp.as_dcm()

    return Rnew, Cnew


def convertHomogeneouos(x):
    """Summary

    Args:
        x (array): 2D or 3D point

    Returns:
        TYPE: point appended with 1
    """
    m, n = x.shape
    if (n == 3 or n == 2):
        x_new = np.hstack((x, np.ones((m, 1))))
    else:
        x_new = x
    return x_new


def LinearPnP(X, x, K):
    """Summary

    Args:
        X (TYPE): 3D points
        x (TYPE): 2D points
        K (TYPE): intrinsic Matrix

    Returns:
        TYPE: C_set, R_set
    """
    N = X.shape[0]
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    x = np.hstack((x, np.ones((x.shape[0], 1))))

    x = np.transpose(np.dot(np.linalg.inv(K), x.T))
    A = []
    for i in range(N):
        xt = X[i, :].reshape((1, 4))
        z = np.zeros((1, 4))
        p = x[i, :]  #.reshape((1, 3))

        a1 = np.hstack((np.hstack((z, -xt)), p[1] * xt))
        a2 = np.hstack((np.hstack((xt, z)), -p[0] * xt))
        a3 = np.hstack((np.hstack((-p[1] * xt, p[0] * xt)), z))
        a = np.vstack((np.vstack((a1, a2)), a3))

        if (i == 0):
            A = a
        else:
            A = np.vstack((A, a))

    _, _, v = np.linalg.svd(A)
    P = v[-1].reshape((3, 4))
    R = P[:, 0:3]
    t = P[:, 3]
    u, _, v = np.linalg.svd(R)

    R = np.matmul(u, v)
    d = np.identity(3)
    d[2][2] = np.linalg.det(np.matmul(u, v))
    R = np.dot(np.dot(u, d), v)
    C = -np.dot(np.linalg.inv(R), t)
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
    return C, R

