from statistics import mode
from cv2 import threshold
import numpy as np
import scipy
import scipy.signal
import cv2
    
# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # Compute image gradients
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    kernel_x = 0.5 * np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
    kernel_y = 0.5 * np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
    Ix = scipy.signal.convolve2d(img, kernel_x, mode='same', boundary='symm')
    Iy = scipy.signal.convolve2d(img, kernel_y, mode='same', boundary='symm')
    
    # Compute local auto-correlation matrix
    # TODO: compute the auto-correlation matrix here
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    Ix2 = cv2.GaussianBlur(np.square(Ix), (3, 3), sigma, borderType=cv2.BORDER_REPLICATE)
    Iy2 = cv2.GaussianBlur(np.square(Iy), (3, 3), sigma, borderType=cv2.BORDER_REPLICATE)
    Ixy = cv2.GaussianBlur(Ix * Iy, (3, 3), sigma, borderType=cv2.BORDER_REPLICATE)         

    # Compute Harris response function
    # TODO: compute the Harris response function C here
    C = Ix2 * Iy2 - np.square(Ixy) - k * np.square(Ix2 + Iy2)
       
    # Detection with threshold
    # TODO: detection and find the corners here
    # For the local maximum check, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
   

    max_filter = scipy.ndimage.maximum_filter(C, size=3)
    suppressed_response = C == max_filter
    thresholded_response = C > thresh
    mask = np.logical_and(suppressed_response, thresholded_response)
    corners = np.argwhere(mask.transpose())
    
    return corners, np.array(C)

