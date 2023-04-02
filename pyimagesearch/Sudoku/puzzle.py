from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

def find_puzzle(image, debug=False):
    #convert the image to grayscale and gaussioan blur
    gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(gray,(7,7),3)

    #applying adaptive thresholding and then convert to threshold map
    thres=cv2.adaptiveThreshold(blurred,255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    thres=cv2.bitwise_not(thres)

    # check to see if we are visualizing each step of the image
	# processing pipeline (in this case, thresholding)

    if debug:
        cv2.imshow("Puzzle Thresh", thres)
        cv2.waitKey(0)

    # find contours in the thresholded image and sort them by size in
	# descending order
    cnts=cv2.findContours(thres.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    cnts=sorted(cnts, key=cv2.contourArea, reverse=True)

    # initialize a contour that corresponds to the puzzle outline
    puzzleCnt=None

    for c in cnts:
        pari=cv2.arcLength(c, True)
        approx=cv2.approxPolyDP(c,0.02*pari, True)
        if len(approx)==4:
            puzzleCnt=approx
            break
    
    # if the puzzle contour is empty then our script could not find
	# the outline of the Sudoku puzzle so raise an error
    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
			"Try debugging your thresholding and contour steps."))

    if debug:
        output=image.copy()
        cv2.drawContours(output,[puzzleCnt],-1,(0,22,0),2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)
    
    puzzle=four_point_transform(image, puzzleCnt.reshape(4,2))
    warped=four_point_transform(gray,puzzleCnt.reshape(4,2))

    if debug:
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)

    return (puzzle,warped)

def extract_digit(cell, debug=False):
    thres=cv2.threshold(cell,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thres=clear_border(thres)

    if debug:
        cv2.imshow("Cell Thresh",thres)
        cv2.waitKey(0)
    
    cnts=cv2.findContours(thres.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)

    if len(cnts)==0:
        return None
    
    c= max(cnts,key=cv2.contourArea)
    mask=np.zeros(thres.shape, dtype="uint8")
    cv2.drawContours(mask,[c],-1,255,-1)

    (h,w)=thres.shape
    percentFilled=cv2.countNonZero(mask)/float(w*h)

    if percentFilled<0.03:
        return None
    
    digit=cv2.bitwise_and(thres, thres, mask=mask)

    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)
    
    return digit
