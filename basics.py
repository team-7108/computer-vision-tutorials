# Import OpenCV module
import cv2
# Import numpy for array operations
import numpy as np
# Print the version of OpenCV 
print("OpenCV version: " + cv2.__version__)
# Read a test image 
image = cv2.imread('images/robot_and_cube.jpeg')
# Show the image
cv2.imshow('Image',image)
# Let's check the size of the image
height, width, channels = image.shape
print("Height:" + str(height) + " Width: " + str(width) + " Channels: " + str(channels))
# Please note that the "channels" variable is 3 as we are currently working on RGB images

# Resize the image if it is too big, also helps to speed up the processing
resizedImage = cv2.resize(image, (600, 600)) 
# Below is an another option for resizing, also keeps aspect ratio this way
# resizedImage = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
# Show the resized image
cv2.imshow('Resized Image',resizedImage)
# Let's check the size of the resized image
height, width, channels = resizedImage.shape
print("Height:" + str(height) + " Width: " + str(width) + " Channels: " + str(channels))
# Now we have actually verified that our resizing function worked properly

# In general it is desirable to process image in grayscale, as we reduce the number of channels to 1
grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', grayImage)
# Also there is HSV, which helps us to deal with some light effects and stuff
hsvImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV Image', hsvImage)
# But for now lets extract the yellow color from the resized image and see what we have
lower_yellow = np.array([0,100,80]) # Lower BGR values of the power cube
upper_yellow = np.array([60,160,255]) # Upper BGR values of the power cube
maskedImage = cv2.inRange(resizedImage, lower_yellow, upper_yellow) # Apply the pre defined mask
yellowImage = resizedImage.copy() # Have a backup for your image
yellowImage[np.where(maskedImage==0)] = 0 # Apply the mask to the backup image
cv2.imshow("Yellow Image", yellowImage) # Let's show the image and the see what we have
cv2.imshow("Masked Image", maskedImage) # Now we have a masked binary image

# Now as can be seen we have some extra pixels around the image
# We will use so called morphological transofrmations to obtain a better image
# We first define a kernel, basically a matrix filled with ones.
# If the kernel is larger, it will have more effect on the image
kernel = np.ones((5,5),np.uint8)
# the first morphological transformation is called opening, it will sweep out extra lone pixels around the image
openedImage = cv2.morphologyEx(maskedImage, cv2.MORPH_OPEN, kernel)
cv2.imshow("Open Image", openedImage)
# We have succesfully cleared the extra pixels around the image
# But now we have some missing pixels inside the cube, for this purpose we will use closing
# Also please note that we are using a bigger kernel for this purpose
kernel = np.ones((30,30),np.uint8)
filledImage = cv2.morphologyEx(openedImage, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Filled Image", filledImage)

# Now we will mark the cube to be more human readable
# We will use a method called, contour finding to extract the outline of the cube
# As we are using RETR_EXTERNAL approcach, actually we didn't need to fill inside
contourImage, contours, hierarchy = cv2.findContours(filledImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
print(cnt) # contours are the points on the outline of the image
# bounding rectangle is the minimum rectangle that includes all the contours
# this bounding rectangle is perpendicular to image
x,y,w,h = cv2.boundingRect(cnt) 
# We mark that bounding rectangle with green
cv2.rectangle(resizedImage,(x,y),(x+w,y+h),(0,255,0),2)
# If we want a rotated rectangle we use a slightly different method
rect = cv2.minAreaRect(cnt) # get the minimimum rectangle area
box = cv2.boxPoints(rect) # get the points of the rectangle
box = np.int0(box) # turn them into integers
cv2.drawContours(resizedImage,[box],0,(0,0,255),2) # draw the points
cv2.imshow("Bboxed Image", resizedImage)

keyPressed = cv2.waitKey(0)  # Look for keys to be pressed 
if keyPressed == 27: # if the key is ESC, check the ASCII table, 27 = ESC
	cv2.destroyAllWindows() # Destroy the windows and close the program
