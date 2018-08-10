# Import opencv library
import cv2
# Import numpy for array operations
import numpy as np

# This is a dummy function needed for creating trackbars
def nothing(x):
	pass

# Create a window named 'Colorbars'
cv2.namedWindow('Colorbars')
# Assign strings for ease of coding
hh='Hue High'
hl='Hue Low'
sh='Saturation High'
sl='Saturation Low'
vh='Value High'
vl='Value Low'
wnd = 'Colorbars'
# Begin Creating trackbars for each HSV value
cv2.createTrackbar(hl, wnd, 0,   255, nothing)
cv2.createTrackbar(hh, wnd, 112,  255, nothing)
cv2.createTrackbar(sl, wnd, 187,  255, nothing)
cv2.createTrackbar(sh, wnd, 255, 255, nothing)
cv2.createTrackbar(vl, wnd, 195, 255, nothing)
cv2.createTrackbar(vh, wnd, 255, 255, nothing)

# Start video capture from webcam
cap = cv2.VideoCapture(1)

# Loop through the video frames
while (True):
	# Read frames from the video capture
	ret, frame = cap.read()
	# Flip the frames upside down 
	# frame = cv2.flip(frame,0)
	# Show the frames
	cv2.imshow("Video Capture",frame)

	hul=cv2.getTrackbarPos(hl, wnd)
	huh=cv2.getTrackbarPos(hh, wnd)
	sal=cv2.getTrackbarPos(sl, wnd)
	sah=cv2.getTrackbarPos(sh, wnd)
	val=cv2.getTrackbarPos(vl, wnd)
	vah=cv2.getTrackbarPos(vh, wnd)
	hsvLow=np.array([hul,sal,val])
	hsvHigh=np.array([huh,sah,vah])

	maskedImage = cv2.inRange(frame, hsvLow, hsvHigh)
	cv2.imshow('Masked Image', maskedImage)

	kernel = np.ones((3,3),np.uint8)
	# the first morphological transformation is called opening, it will sweep out extra lone pixels around the image
	openedImage = cv2.morphologyEx(maskedImage, cv2.MORPH_OPEN, kernel)
	cv2.imshow("Open Image", openedImage)
	# Invert the black and white parts of the image for our next algorithm
	invertedImage = cv2.bitwise_not(openedImage)
	cv2.imshow("Inverted Image", invertedImage)
	# Let's implement some nice algorithm called blob detection to detect the cube
	# https://www.learnopencv.com/blob-detection-using-opencv-python-c/
	params = cv2.SimpleBlobDetector_Params()

	# Filter by Area.
	params.filterByArea = True
	params.minArea = 200
	 
	# Filter by Circularity
	params.filterByCircularity = False
	params.minCircularity = 0.3
	params.maxCircularity = 1
	 
	# Filter by Convexity
	params.filterByConvexity = False
	params.minConvexity = 0.87
	 
	# Filter by Inertia
	params.filterByInertia = False
	params.minInertiaRatio = 0.5

	# Apply the detector
	detector = cv2.SimpleBlobDetector_create(params)
	# Extract keypoints
	keypoints = detector.detect(invertedImage)
	# Apply to the Image 
	outputImage = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imshow("Output Image", outputImage)

	# A variable to carry the number of keypoints in the loop
	keypointCounter = 0
	# Iterate and extract the positions and size of each keypoint detected
	for keypoint in keypoints:
	    x = keypoint.pt[0]
	    y = keypoint.pt[1]
	    s = keypoint.size
	    keypointCounter = keypointCounter + 1
	    print("X: " + str(x) + " Y: " + str(y) + " Size: " + str(s))

	print("Number of blobs: " + str(keypointCounter))

	keyPressed = cv2.waitKey(1)  # Look for keys to be pressed 
	if keyPressed == 27: # if the key is ESC, check the ASCII table, 27 = ESC
		break		

cap.release() # Close the video capture
cv2.destroyAllWindows() # Destroy the windows and close the program
