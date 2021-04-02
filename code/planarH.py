import numpy as np
import cv2
from BRIEF import briefLite, briefMatch


def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)
    #############################
    # TO DO ...
    _, N = p1.shape
    x = np.transpose(p1[[0], :])
    y = np.transpose(p1[[1], :])
    u = np.transpose(p2[[0], :])
    v = np.transpose(p2[[1], :])
    matrix1 = np.hstack((-1 * u, -1 * v, -1 * np.ones((N, 1)), np.zeros((N, 3)), x * u, x * v, x))
    matrix2 = np.hstack((np.zeros((N, 3)), -1 * u, -1 * v, -1 * np.ones((N, 1)), y * u, y * v, y))
    A = np.vstack((matrix1, matrix2))

    u, sigma, vt = np.linalg.svd(A)
    h = vt[-1, :]
    H2to1 = h.reshape(3, 3)
    return H2to1


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''
    ###########################
    # TO DO ...
    base_location = locs1[matches[:, 0], : -1]
    target_location = locs2[matches[:, 1], : -1]
    N, _ = base_location.shape
    # base_homography = np.vstack((np.transpose(base_location), np.ones((1, N))))
    target_homography = np.vstack((np.transpose(target_location), np.ones((1, N))))

    count_in = 0
    inlier_coordinates = None
    for i in range(num_iter):
        index = np.random.randint(0, N, 4)
        points1 = np.transpose(base_location[index])
        points2 = np.transpose(target_location[index])
        H = computeH(points1, points2)
        # use the computed matrix to transform the points in homogeneous coordinates
        transformed_target = H @ target_homography
        lam = transformed_target[-1, :]
        transformed_target = transformed_target / lam
        # now we get the homo coordinate with third dimension equals to 1
        transformed_target = transformed_target[:-1, :]
        # now compute the difference
        diff = transformed_target - np.transpose(base_location)
        diff = np.linalg.norm(diff, axis=0)
        count = np.asarray(diff < tol).sum()

        if count > count_in:
            count_in = count
            inlier_coordinates = np.asarray(diff < tol).nonzero()

    base_inlier = base_location[inlier_coordinates]
    target_inlier = target_location[inlier_coordinates]
    bestH = computeH(base_inlier.T, target_inlier.T)
    return bestH


if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
