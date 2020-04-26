import cv2
import numpy as np

def refine_mask(mask):
    '''
    Detects the lines on the lateral aspects of a mask representing the ultrasound beam. Returns a mask that isolates
    any areas bounded by these lines and the uppermost corners of the image.
    :param mask: Mask of the ultrasound beam after filtering out everything but the largest contour
    :return: A mask that filters out triangles formed by the lateral aspects of the input mask and the upper corners of
             the image.
    '''
    edged = cv2.Canny(mask, 50, 255)    # Perform edge detection on the mask
    lines = cv2.HoughLinesP(edged, rho=6, theta=np.pi / 60, threshold=160, lines=np.array([]), minLineLength=200,
        maxLineGap=40)                  # Try line detection on mask edges
    triangles = []
    y_bot = mask.shape[0]
    y_top = 0
    x_left = 0
    x_right = mask.shape[1]

    # Extend the lines to the borders of the images. Then get triangles from the line endpoints and top corners of mask.
    for line in lines[0:2]:
        (x1, y1, x2, y2) = line[0]      # Get coordinates of endpoints of detected lines
        m = (y2 - y1) / (x2 - x1)       # Find the line's slope
        x1_orig = x1
        if y1 < y2:
            x1 = max(x_left, x1 - 500, x1 - int((y1 - y_top) / m))
            x2 = min(x_right, x2 + 500, x2 + int((y2 + y_bot) / m))
            y1 = max(y_top, y1 + int((x1 - x1_orig) * m))
            y2 = y1 + int((x2 - x1) * m)
            triangle = [[x1, y1], [x2, y2], [x2, y1]]
        else:
            x1 = max(x_left, x1 - 500, x1 + int((y_bot - y1) / m))
            x2 = min(x_right, x2 + 500, x2 - int((y2 + y_top) / m))
            y1 = min(y_bot, y1 + int((x1 - x1_orig) * m))
            y2 = y1 + int((x2 - x1) * m)
            triangle = [[x1, y1], [x2, y2], [x1, y2]]
        triangles.append(np.array(triangle, np.int32).reshape(-1, 1, 2))

    triangle_mask = np.ones(mask.shape, np.uint8) * 255
    cv2.fillPoly(triangle_mask, triangles, [0, 0, 0])  # Create the mask by filling in the triangles with white.
    return triangle_mask


def filter_beam(orig_img):
    '''
    Create a mask that isolates the ultrasound beam in an ultrasound image. Use the mask to blacken all regions in the
    image that are not within the bounds of the ultrasound beam.
    :param orig_img: A numpy array representing the ultrasound image to filter
    :return: A numpy array representing the filtered ultrasound image
    '''
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)    # Get grey version of image.

    # Mask out any brightly coloured regions (e.g. manufacturer's logo)
    grey_mask = (np.logical_and.reduce((abs(orig_img[:,:,0] - orig_img[:,:,1]) < 25, abs(orig_img[:,:,1] - orig_img[:,:,2]) < 25,
                                         abs(orig_img[:,:,0] - orig_img[:,:,2]) < 25)) * 255).astype(np.uint8)
    bright_col_area = cv2.bitwise_and(img, cv2.bitwise_not(grey_mask))      # Select candidate brightly coloured regions
    bright_col_area = cv2.blur(bright_col_area, (4, 4))                     # Slightly blur the brightly coloured regions
    bright_col_mask = ((bright_col_area < 25) * 255).astype(np.uint8)       # Create a mask for brightly coloured regions
    img = cv2.bitwise_and(img, bright_col_mask)                             # Mask out brightly coloured regions

    # Mask out and select the largest continuous contour in this image (i.e. the ultrasound beam).
    ret, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)     # Threshold all non-black parts of the image.
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)    # Find contours in the image.
    beam_contour = max(contours, key=cv2.contourArea)               # Select the contour with the greatest area (the US beam).
    mask = np.zeros(orig_img.shape, np.uint8)
    cv2.fillPoly(mask, [beam_contour], [255,255,255])               # Create the mask by filling in the contour with white.

    # Attempt to filter out triangles in top corners of image
    try:
        triangle_mask = refine_mask(mask)
        mask = cv2.bitwise_and(mask, triangle_mask)
    except:
        print("Triangle mask failed.")

    final_img = cv2.bitwise_and(orig_img, mask)    # Mask everything out but the US beam.
    return final_img
