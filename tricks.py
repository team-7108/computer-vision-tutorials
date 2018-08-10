# Import OpenCV module
import cv2
# Import numpy for array operations
import numpy as np

# This is a dummy function needed for creating trackbars
def nothing(x):
    pass

def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes 
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

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
cv2.createTrackbar(hh, wnd, 49,  255, nothing)
cv2.createTrackbar(sl, wnd, 191,  255, nothing)
cv2.createTrackbar(sh, wnd, 255, 255, nothing)
cv2.createTrackbar(vl, wnd, 105, 255, nothing)
cv2.createTrackbar(vh, wnd, 255, 255, nothing)

# Read a test image 
# image = cv2.imread('yellow_robot_and_cube.jpeg')
image = cv2.imread('images/five_cubes.jpeg')
# Resize the image if it is too big, also helps to speed up the processing
resizedImage = cv2.resize(image, (600, 600))

while True:
    hul=cv2.getTrackbarPos(hl, wnd)
    huh=cv2.getTrackbarPos(hh, wnd)
    sal=cv2.getTrackbarPos(sl, wnd)
    sah=cv2.getTrackbarPos(sh, wnd)
    val=cv2.getTrackbarPos(vl, wnd)
    vah=cv2.getTrackbarPos(vh, wnd)
    hsvLow=np.array([hul,sal,val])
    hsvHigh=np.array([huh,sah,vah])

    maskedImage = cv2.inRange(resizedImage, hsvLow, hsvHigh)
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
    outputImage = cv2.drawKeypoints(resizedImage, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Output Image", outputImage)

    # A variable to carry the number of keypoints in the loop
    keypointCounter = 0
    # Variables to hold x and y coordinates of the keypoints
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    # Bounding box array initializtion
    bboxes = np.array([[0,0,0,0]])
    # Iterate and extract the positions and size of each keypoint detected
    for keypoint in keypoints:
        # Get x andy coordinates and sizes of the keypoints
        x = keypoint.pt[0]
        y = keypoint.pt[1]
        s = keypoint.size
        
        x1.append((int)(x-s))
        y1.append((int)(y-s))
        x2.append((int)(x+s))
        y2.append((int)(y+s))

        print("X: " + str(x) + " Y: " + str(y) + " Size: " + str(s))
        # Draw bounding box
        cv2.rectangle(outputImage,(x1[keypointCounter],y1[keypointCounter]),(x2[keypointCounter],y2[keypointCounter]), (0,255,0),2)
        # Get bounding boxes for non-maximum suppression
        bbox = np.array([[x1[keypointCounter],y1[keypointCounter],x2[keypointCounter],y2[keypointCounter]]])
        bboxes = np.concatenate((bboxes, bbox))
        keypointCounter = keypointCounter + 1

    # Delete the first elemnt of bboxes, the one that we've used for initialization
    bboxes = np.delete(bboxes,0,0)
    # Show Bboxed Image
    cv2.imshow("Bboxed Image", outputImage)
    print("Number of blobs: " + str(keypointCounter))

    # Apply non-maximum suppression, reference: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    pick = non_max_suppression_fast(bboxes, 0.01)
    print "After applying non-maximum suppression we have, %d bounding boxes" % (len(pick))
    
    nmsImage = resizedImage.copy()
    for (startX, startY, endX, endY) in pick:
        if startY>420: # cut the upper side of the image because we already know that cubes are around floor level
            cv2.rectangle(nmsImage, (startX, startY), (endX, endY), (255, 0, 0), 4)
            print("X: " + str(startX) + " Y: " + str(startY) + " Size: " + str(endX-startX) + " x " + str(endY-startY))

    cv2.imshow("NMS Image", nmsImage)

    keyPressed = cv2.waitKey(1)  # Look for keys to be pressed 
    if keyPressed == 27: # if the key is ESC, check the ASCII table, 27 = ESC
        cv2.destroyAllWindows() # Destroy the windows and close the program
