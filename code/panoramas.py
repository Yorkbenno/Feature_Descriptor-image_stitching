import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite, briefMatch, plotMatches


def imageStitching(im1, im2, H2to1):
    """
    Returns a panorama of im1 and im2 using the given
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    """
    #######################################
    # TO DO ...
    im1H, im1W, channels = im1.shape
    im1_stretch = np.uint8(np.hstack((im1, np.zeros((im1H, im1W, channels)))))
    warpim = np.uint8(cv2.warpPerspective(im2, H2to1, (im1W * 2, im1H)))
    pano_im = np.uint8(np.maximum(warpim, im1_stretch))
    cv2.imwrite('../results/q6_1.jpg', pano_im)
    np.save('../results/q6_1.npy', H2to1)
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    """
    Returns a panorama of im1 and im2 using the given
    homography matrix without cliping.
    """
    ######################################
    # TO DO ...
    im1H, im1W, channels = im1.shape
    im2H, im2W, channels = im2.shape
    corners = np.vstack(([0, 0, im2W - 1, im2W - 1], [0, im2H - 1, im2H - 1, 0], [1, 1, 1, 1]))
    transform_corner = H2to1 @ corners
    transform_corner = (transform_corner / transform_corner[-1, :]).round().astype(int)
    Height_max = np.max([im1H, np.max(transform_corner[1, :])])
    Width_max = np.max([im1W, np.max(transform_corner[0, :])])
    Height_min = np.min([0, np.min(transform_corner[1, :])])
    Width_min = np.min([0, np.min(transform_corner[0, :])])

    height = Height_max - Height_min
    width = Width_max - Width_min

    # print(Hmax, Hmin, Wman, Wmin)
    M = np.vstack(([1., 0., 0.], [0., 1., -Height_min], [0., 0., 1.]))
    trans_im1 = cv2.warpPerspective(im1, M, (width, height))
    trans_im2 = cv2.warpPerspective(im2, M @ H2to1, (width, height))

    pano_im = np.uint8(np.maximum(trans_im1, trans_im2))
    cv2.imwrite('../results/q6_2_pan.jpg', pano_im)
    return pano_im


def generatePanorama(im1, im2):
    """
    INPUT
        Two images im1 and im2

    OUTPUT
        im3 which is the panorama image
    """
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)

    cv2.imwrite('../results/q6_3.jpg', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)

    return pano_im


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    # print(im1.shape)
    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    # plotMatches(im1, im2, matches, locs1, locs2)
    # H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # pano_im = imageStitching_noClip(im1, im2, H2to1)
    # print(H2to1)
    pano_im = generatePanorama(im1, im2)
    # cv2.imwrite('../results/panoImg.png', pano_im)
    # cv2.imshow('panoramas', pano_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
