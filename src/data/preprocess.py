import cv2


def to_grayscale(img):
    '''
    Converts RBG image to greyscale.
    :param img: RBG image
    :return: Image to convert to greyscale
    '''
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)