import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def createGaussianPyramid(im, sigma0=1,
                          k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4]):
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max() > 10:
        im = np.float32(im) / 255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0 * k ** i
        im_pyramid.append(cv2.GaussianBlur(im, (0, 0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0)  # press any key to exit
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1, 0, 1, 2, 3, 4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    imH, imW, _ = gaussian_pyramid.shape
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    # We only need the last n-1 levels
    DoG_levels = levels[1:]
    # The levels starts at 0, so just use without len() is okay
    for i in DoG_levels:
        DoG_pyramid.append(gaussian_pyramid[:, :, i + 1] - gaussian_pyramid[:, :, i])
    DoG_pyramid = np.stack(DoG_pyramid, axis=2)
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    imH, imW, num_levels = DoG_pyramid.shape
    principal_curvature = []
    ##################
    # TO DO ...
    # Compute principal curvature here
    for i in range(num_levels):
        dxx = cv2.Sobel(DoG_pyramid[:, :, i], cv2.CV_64F, 2, 0, ksize=3)
        dyy = cv2.Sobel(DoG_pyramid[:, :, i], cv2.CV_64F, 0, 2, ksize=3)
        dxy = cv2.Sobel(DoG_pyramid[:, :, i], cv2.CV_64F, 1, 1, ksize=3)
        trace = dxx + dyy
        det = dxx * dyy - dxy * dxy
        det[det == 0] = 0.001
        R = np.square(trace) / det
        principal_curvature.append(R)

    principal_curvature = np.stack(principal_curvature, axis=2)
    return principal_curvature


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
                    th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = []
    imH, imW, num_levels = DoG_pyramid.shape
    ##############
    #  TO DO ...
    # Compute locsDoG here
    right = np.roll(DoG_pyramid, 1, axis=1)
    right_up = np.roll(right, 1, axis=0)
    right_down = np.roll(right, -1, axis=0)
    up = np.roll(DoG_pyramid, 1, axis=0)
    left = np.roll(DoG_pyramid, -1, axis=1)
    left_up = np.roll(left, 1, axis=0)
    left_down = np.roll(left, -1, axis=0)
    down = np.roll(DoG_pyramid, -1, axis=0)
    forward = np.roll(DoG_pyramid, 1, axis=2)
    backward = np.roll(DoG_pyramid, -1, axis=2)

    maximun = (right < DoG_pyramid) & (right_up < DoG_pyramid) & (right_down < DoG_pyramid) & (up < DoG_pyramid) & (
            left < DoG_pyramid) & (left_up < DoG_pyramid) & (left_down < DoG_pyramid) & (down < DoG_pyramid) \
              & (forward < DoG_pyramid) & (backward < DoG_pyramid)
    minimum = (right > DoG_pyramid) & (right_up > DoG_pyramid) & (right_down > DoG_pyramid) & (up > DoG_pyramid) & (
            left > DoG_pyramid) & (left_up > DoG_pyramid) & (left_down > DoG_pyramid) & (down > DoG_pyramid) \
              & (forward > DoG_pyramid) & (backward > DoG_pyramid)

    result = maximun | minimum

    th_filter1 = abs(DoG_pyramid) > th_contrast
    result = result & th_filter1
    th_filter2 = abs(principal_curvature) < th_r
    result = result & th_filter2

    for i in range(1, imH-1):
        for j in range(1, imW-1):
            for level in range(1, num_levels-1):
                if result[i, j, level]:
                    locsDoG.append(np.array([j, i, level]))
    locsDoG = np.stack(locsDoG, axis=0)
    return locsDoG


def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4],
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyramid, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    principle_curvature = computePrincipalCurvature(DoG_pyramid)
    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, principle_curvature, th_contrast, th_r)

    return locsDoG, gauss_pyramid


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1, 0, 1, 2, 3, 4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    # displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    # displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    # displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    # print(locsDoG)
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    fig = plt.imshow(im, cmap='gray')
    plt.plot(locsDoG[:, 0], locsDoG[:, 1], 'o', color='lime', markersize=3)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if not os.path.exists('../results'):
        os.mkdir('../results')
    plt.savefig('../results/edge_suppresion.png')
    plt.draw()
    plt.show()
