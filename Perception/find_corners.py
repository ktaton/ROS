import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

#prepare object points
nx = 8
ny = 6

#Make a list of calibration images
images = glob.glob('test_images/GO*.jpg')
#select any index to grab an image from the list
idx = 4
#Read in the image
img = mpimg.imread(images[idx])

#Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray,(nx,ny),None)

#If found, draw corners
if ret == True:
    cv2.drawChessboardCorners(img,(nx,ny),corners,ret)
    #plt.imshow(img)
    cv2.imshow('frame',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
