import cv2
import numpy as np

def refine_mask(mask):
    '''
    Detects the lines on the lateral aspects of a mask representing the ultrasound beam.
    Returns a mask that isolates any areas bounded by these lines and the uppermost
    corners of the image.
    :param mask: Mask of the ultrasound beam after filtering out everything but the largest contour
    :return: A mask that filters out triangles formed by the lateral aspects of the input mask
        and the upper corners of the image.
    '''
    edged = cv2.Canny(mask, 50, 255)    # Perform edge detection on the mask
    lines = cv2.HoughLinesP(edged, rho=6, theta=np.pi / 60, threshold=100, lines=np.array([]),
                            minLineLength=50, maxLineGap=30)      # Try line detection on mask edges
    final_lines = []

    # Find the leftmost and rightmost points in every row
    # Compare with the topmost points in every column to determine the right and left edges
    left_edgelist, right_edgelist, top_edgelist = [], [], []
    for i in range(edged.shape[0]):
        j = edged.shape[1]-1
        while edged[i][j] == 0 and j > 0:
            j -= 1
        if j != 0:
            right_edgelist.append((i, j))
        j = 0
        while edged[i][j] == 0 and j < edged.shape[1]-1:
            j += 1
        if j != edged.shape[1]:
            left_edgelist.append((i, j))
    for j in range(edged.shape[1]):
        i = 0
        while edged[i][j] == 0 and i < edged.shape[0]-1:
            i += 1
        if i != 0:
            top_edgelist.append((i, j))
    left_edgelist = [x for x in left_edgelist if x in top_edgelist]
    right_edgelist = [x for x in right_edgelist if x in top_edgelist]

    # Search through the lines to find the ones that best match the left and right edges by summing
    # the distance from the line to each point on the edges
    # Additional limits: slopes must be > 0.5 and < 2
    min_diff_left = 2**20 - 1
    min_diff_right = 2**20 - 1
    best_left = []
    best_right = []
    penalty = 30 # To ensure outlier points don't affect the line too much, penalty has a maximum

    for line in lines:
        if line[0][2] == line[0][0] or line[0][3] == line[0][1]:
            continue
        m = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])
        b = line[0][3] - m*line[0][2]

        if m < -0.5 and m > -2: # For lines of negative slope
            diff = 0
            for point in left_edgelist:
                diff += min(penalty, abs(point[0] - (m*point[1]+b)))
                diff += min(penalty, abs(point[1] - (point[0]-b)/m))
            if diff < min_diff_left:
                best_left = [m,b]
                min_diff_left = diff

        if m > 0.5 and m < 2: # For lines of positive slope
            diff = 0
            for point in right_edgelist:
                diff += min(penalty, abs(point[0] - (m*point[1]+b)))
                diff += min(penalty, abs(point[1] - (point[0]-b)/m))
            if diff < min_diff_right:
                best_right = [m,b]
                min_diff_right = diff

    final_lines = [best_left[0], best_left[1], best_right[0], best_right[1]]

    # Draw triangles using the determined lines and the corners of the image onto the mask.
    triangles = []
    triangle_mask = np.ones(mask.shape, np.uint8) * 255
    for i in range(2):
        m , b = final_lines[i*2], final_lines[i*2+1]
        x1 = 0 if m < 0 else mask.shape[1]
        y1 = 0
        x2 = int((y1 - b) / m)
        y2 = int(m * x1 + b)
        triangle = [[x1, y2], [x2, y1], [x1, y1]]
        triangles.append(np.array(triangle, np.int32))

    cv2.fillPoly(triangle_mask, triangles, [0, 0, 0])  # Fill in the triangles with black.
    lower_edge = find_lower_edge(mask, edged, final_lines) # Run method to find lower edge
    triangle_mask = cv2.bitwise_and(triangle_mask, lower_edge) # Combine with lower edge mask.

    return triangle_mask


def filter_beam(orig_img, triangles_mask=True):
    '''
    Create a mask that isolates the ultrasound beam in an ultrasound image.
    Use the mask to blacken all regions in the image that are not within
    the bounds of the ultrasound beam.
    :param orig_img: A numpy array representing the ultrasound image to filter
    :param triangles_mask: Boolean indicating whether to attempt filtering out
    :return: A numpy array representing the filtered ultrasound image
    '''
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)    # Get grey version of image.

    # Mask out any brightly coloured regions (e.g. manufacturer's logo)
    grey_mask = (np.logical_and.reduce((abs(orig_img[:,:,0] - orig_img[:,:,1]) < 25, 
                                        abs(orig_img[:,:,1] - orig_img[:,:,2]) < 25,
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
    if triangles_mask:
        triangle_mask = refine_mask(mask)
        # mask = cv2.bitwise_and(mask, triangle_mask) # Combines contour with masks

    final_img = cv2.bitwise_and(orig_img, triangle_mask)    # Mask everything out but the US beam.
    return final_img

def find_lower_edge(mask, edged, lines):
    '''
    Detects a circle with x-coordinate of centre given by lines of triangles isolated 
    by refine_mask. Returns a mask isolating all contents within the circle.
    :param edged: the edges of the largest contour of the ultrasound beam
    :param lines: An array containing the slopes and intercepts of the lines.
    :mask: A mask that filters out anything not bound by the generated circle.
    '''
    mi, bi, mj, bj = lines[0:4]

    # Find the bottom-most positive value in every column
    edgelist = []
    for j in range(edged.shape[1]):
        i = len(edged)-1
        while edged[i][j] == 0 and i != 0:
            i -= 1
        if i != 0:
            edgelist.append((i, j))

    # Sort the list and take the 400 lowest points
    edgelist.sort(reverse=True)
    edgelist = [x[::-1] for x in edgelist]

    # Determine the position and size of the circle using points spaced half the list apart
    # Attempt to minimize distance of the circle to every point in the bottom edge
    # Variables named after circle formula (x-h)^2 + (y-k)^2 = r^2
    h = (bj - bi) / (mi - mj)
    intersect = mi * h + bi
    k_final = 0
    max_radius = 0
    max_difference = 2**20 - 1
    for index in range(int(len(edgelist) * 0.5)):
        p1 = edgelist[index]
        p2 = edgelist[index + int(len(edgelist) * 0.5)]
        x1, x2, y1, y2 = p1[0], p2[0], p1[1], p2[1]
        k = (((x1 - h)**2 + y1**2) - ((x2 - h)**2 + y2**2)) / (2*(y1 - y2))
        r = abs(((x1 - h)**2 + (y1 - k)**2)**0.5) 
        difference = 0
        for point in edgelist:
            difference += abs(point[1] - ((r**2 - (point[0] - h)**2)**0.5 + k))
        if difference < max_difference:
            max_radius = r
            k_final = k
            max_difference = difference

    # Make an array of zeroes the size of the image
    # Fill the circle with ones
    lower_edge_mask = np.zeros(mask.shape, np.uint8)
    cv2.circle(lower_edge_mask, (int(h), int(k_final)), int(max_radius), [255, 255, 255], -1)

    return lower_edge_mask
