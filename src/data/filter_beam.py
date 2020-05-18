import cv2
import numpy as np

def refine_mask(mask):
    '''
    Detects the lines on the lateral aspects of a mask representing the ultrasound beam.
    Returns a mask that isolates any areas bounded by these lines and the uppermost
    corners of the image.
    :param mask: Mask of the ultrasound beam after filtering out everything but the largest contour
    :return: A mask that filters out triangles formed by the lateral aspects of the input mask
        and the upper corners of the image, or 0 if failed
    '''
    edged = cv2.Canny(mask, 50, 255)    # Perform edge detection on the mask
    lines1 = cv2.HoughLinesP(edged, rho=6, theta=np.pi / 60, threshold=100, lines=np.array([]),
                             minLineLength=30, maxLineGap=30)    # Look for short lines
    lines2 = cv2.HoughLinesP(edged, rho=6, theta=np.pi / 60, threshold=100, lines=np.array([]),
                             minLineLength=90, maxLineGap=50)    # Look for long lines
    lines = np.concatenate((lines1, lines2), axis=0)
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
    top_point = edged.shape[0]
    for j in range(edged.shape[1]):
        i = 50
        while edged[i][j] == 0 and i < edged.shape[0]-1:
            i += 1
        if i != 0:
            top_edgelist.append((i, j))
            if i < top_point:
                top_point = i
    left_edgelist = [x for x in left_edgelist if x in top_edgelist]
    right_edgelist = [x for x in right_edgelist if x in top_edgelist]

    # Search through the lines to find the ones that best match the left and right edges by summing
    # the distance from the line to each point on the edges
    # Additional limits: slopes must be > 0.6 and < 1.9
    min_diff_left = 2**20-1
    min_diff_right = 2**20-1
    best_left = []
    best_right = []
    penalty = 30 # Max penalty score to ensure outlier points don't affect it too much

    for line in lines:
        if line[0][2] == line[0][0] or line[0][3] == line[0][1]: # Eliminate straight lines
            continue
        m = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])
        b = line[0][3] - m*line[0][2]

        if m < -0.6 and m > -1.9: # For lines of negative slope
            diff = 0
            for point in left_edgelist:
                vert_diff = point[0] - (m*point[1]+b)
                horiz_diff = point[1] - (point[0]-b)/m
                diff += 10 if vert_diff > 10 else min(penalty, abs(vert_diff)) # Penalize less for points included in the cut
                diff += 10 if horiz_diff > 10 else min(penalty, abs(horiz_diff))
            if diff < min_diff_left:
                best_left = [m,b]
                min_diff_left = diff

        if m > 0.6 and m < 1.9: # For lines of positive slope
            diff = 0
            for point in right_edgelist:
                vert_diff = point[0] - (m*point[1]+b)
                horiz_diff = point[1] - (point[0]-b)/m
                diff += 10 if vert_diff > 10 else min(penalty, abs(vert_diff))
                diff += 10 if horiz_diff < -10 else min(penalty, abs(horiz_diff))
            if diff < min_diff_right:
                best_right = [m,b]
                min_diff_right = diff

    # Attempt to characterize two best lines.
    # Algorithm fails if the lines intersect below the contour or if the lines are not mirror images
    try:
        final_lines = [best_left[0], best_left[1], best_right[0], best_right[1]]
        intersect = best_right[0]*int((best_right[1] - best_left[1]) / (best_left[0] - best_right[0])) + best_right[1]
        if (intersect > top_point + 20) and (top_point > 20):
            raise Exception("intercept too low")
        if abs(best_left[0] + best_right[0]) > 0.15:
            raise Exception("lines not mirrored")
    except Exception: 
        return 0            # Algorithm could not find two best lines

    # Draw triangles using the determined lines and the corners of the image onto the mask.
    polygons = []
    triangle_mask = np.ones(mask.shape, np.uint8) * 255
    for i in range(2):
        m , b = final_lines[i*2], final_lines[i*2+1]
        x1 = 0 if m < 0 else mask.shape[1]
        y1 = 0
        x2 = int((y1 - b) / m)
        y2 = int(m * x1 + b)
        triangle = [[x1, y2], [x2, y1], [x1, y1]]
        cv2.fillPoly(triangle_mask, [np.array(triangle, np.int32)], [0, 0, 0])

    # Draw a rectangle to mask out any sections above the ultrasound beam.
    x1 = 0
    x2 = mask.shape[1]
    y1 = 0
    y2 = max(intersect, top_point - 30)
    rectangle = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
    cv2.fillPoly(triangle_mask, [np.array(rectangle, np.int32)], [0, 0, 0])

    #Attempt to find lower edge of ultrasound beam
    try:
        lower_edge = find_lower_edge(mask, edged, final_lines)
    except Exception:
        return 0        # Algorithm could not find matching circle within constraints

    triangle_mask = cv2.bitwise_and(triangle_mask, lower_edge) # Combine with lower edge mask.
    return triangle_mask

def find_contour_area(orig_img):
    '''
    Find the area of the largest contour of the image.
    :param orig_img: A numpy array of the ultrasound image
    :return: the area of the largest contour of the image.
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

    return cv2.contourArea(beam_contour)

def contour_image(orig_img):
    '''
    Create a mask that isolates the ultrasound beam in an ultrasound image using the largest contour in the image.
    Use the mask to blacken all regions in the image that are not within the bounds of the ultrasound beam.
    :param orig_img: A numpy array representing the ultrasound image to filter
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

    final_img = cv2.bitwise_and(orig_img, mask)    # Mask everything out but the US beam.
    return final_img

def filter_beam(orig_img):
    '''
    Create a mask that isolates the ultrasound beam in an ultrasound image. 
    :param orig_img: A numpy array representing the ultrasound image to filter
    :return: A numpy array representing the mask created by the algorithm
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

    return refine_mask(mask) # Return the mask used

def find_lower_edge(mask, edged, lines):
    '''
    Detects a circle with x-coordinate of centre given by lines of triangles isolated
    by refine_mask. Returns a mask isolating all contents within the circle.
    :mask: A mask that filters out anything not bound by the generated circle.
    :param edged: the edges of the largest contour of the ultrasound beam
    :param lines: An array containing the slopes and intercepts of the lines.
    :return: A mask isolating all contents within the circle.
    '''
    mi, bi, mj, bj = lines[0:4]

    # Find the bottom-most positive value in every column, up to 3/4 the height of the image.
    edgelist = []
    for j in range(edged.shape[1]):
        i = len(edged)-1
        while edged[i][j] == 0 and i != int(len(edged)*0.75):
            i -= 1
        if i != int(len(edged)*0.75):
            edgelist.append((i,j))

    # Sort the list and take the 400 lowest points
    edgelist.sort(reverse=True)
    edgelist = [x[::-1] for x in edgelist]

    # Determine the position and size of the circle using points spaced half the list apart
    # Attempt to minimize distance of the circle to every point in the bottom edge
    # Variables named after standard equations (x-h)^2 + (y-k)^2 = r^2 and y = mx + b
    h = (bj - bi) / (mi - mj)
    k_final = 0
    max_radius = 0
    max_difference = 2**20-1
    gap = int(len(edgelist)*0.5)

    for index in range(len(edgelist) - gap):
        p1 = edgelist[index]
        p2 = edgelist[index + gap]
        x1, x2, y1, y2 = p1[0], p2[0], p1[1], p2[1]
        k = (((x1 - h)**2 + y1**2) - ((x2 - h)**2 + y2**2)) / (2*(y1-y2))
        r = abs(((x1-h)**2+(y1-k)**2)**0.5)
        difference = 0
        for point in edgelist:
            difference += abs(point[1] - ((r**2 - (point[0] - h)**2)**0.5 + k))
        if difference < max_difference:
            max_radius = r
            k_final = k
            max_difference = difference
    
    # Algorithm fails if the circle found has a radius less than 0.6x or greater than the image height.
    if max_radius < 0.6 * mask.shape[0]:
        raise Exception('max_radius not large enough')
    if max_radius > mask.shape[0]:
        raise Exception('max_radius too large')

    # Make an array of zeroes the size of the image
    # Fill the circle with white
    lower_edge_mask = np.zeros(mask.shape, np.uint8)
    cv2.circle(lower_edge_mask, (int(h), int(k_final)), int(max_radius), [255, 255, 255], -1)

    return lower_edge_mask
