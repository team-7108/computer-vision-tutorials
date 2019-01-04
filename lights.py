# Import OpenCV module
import cv2
# Import numpy for array operations
import numpy as np


image = cv2.imread('images/five_cubes.jpeg')

# Show the image
cv2.imshow('Image',image)

# Resize the image if it is too big, also helps to speed up the processing
image = cv2.resize(image, (600, 600))
cv2.imshow('Resized Image',image)

# Equalizing histograms, we try to reduce the effect of light here
image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
channel = cv2.split(image)
cv2.equalizeHist(channel[0], channel[0])
cv2.merge(channel,image)
image = cv2.cvtColor(image,cv2.COLOR_YUV2BGR)
cv2.imshow('Normalized Image',image)

# This is a dummy function needed for creating trackbars
def nothing(x):
    pass

# Create a window named 'Colorbars'
cv2.namedWindow('Colorbars')
# Assign strings for ease of coding
bh='Blue High'
bl='Blue Low'
gh='Green High'
gl='Green Low'
rh='Red High'
rl='Red Low'
wnd = 'Colorbars'
# Begin Creating trackbars for each BGR value
cv2.createTrackbar(bl, wnd, 0,   255, nothing)
cv2.createTrackbar(bh, wnd, 149,  255, nothing)
cv2.createTrackbar(gl, wnd, 156,  255, nothing)
cv2.createTrackbar(gh, wnd, 255, 255, nothing)
cv2.createTrackbar(rl, wnd, 199, 255, nothing)
cv2.createTrackbar(rh, wnd, 255, 255, nothing)


while True:
    mergedImage = np.zeros((600,150,3), np.uint8)
    # Split image into four pieces and merge again
    for i in range(0,4):
        resizedImage = image[0:600, i*150:(i+1)*150]
        cv2.imshow("cropped", resizedImage)

        bLow  = cv2.getTrackbarPos(bl, wnd)
        bHigh = cv2.getTrackbarPos(bh, wnd)
        gLow  = cv2.getTrackbarPos(gl, wnd)
        gHigh = cv2.getTrackbarPos(gh, wnd)
        rLow  = cv2.getTrackbarPos(rl, wnd)
        rHigh = cv2.getTrackbarPos(rh, wnd)
        
        rgbLow=np.array([bLow,gLow,rLow])
        rgbHigh=np.array([bHigh,gHigh,rHigh])

        maskedImage = cv2.inRange(resizedImage, rgbLow, rgbHigh)
        cv2.imshow('Masked Image', maskedImage)

        kernel = np.ones((15,15),np.uint8)
        # the first morphological transformation is called opening, it will sweep out extra lone pixels around the image
        openedImage = cv2.morphologyEx(maskedImage, cv2.MORPH_OPEN, kernel)
        cv2.imshow("Open Image", openedImage)
        outImage = resizedImage.copy()
        try: 
            contourImage, contours, hierarchy = cv2.findContours(openedImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            print(cnt) # contours are the points on the outline of the image
            # bounding rectangle is the minimum rectangle that includes all the contours
            # this bounding rectangle is perpendicular to image
            x,y,w,h = cv2.boundingRect(cnt) 
            # We mark that bounding rectangle with green
            cv2.rectangle(outImage,(x,y),(x+w,y+h),(255,0,0),4)
        except:
            pass
        cv2.imshow("Bboxed",outImage)
        mergedImage = np.concatenate((mergedImage,outImage), axis=1)

    mergedImage = mergedImage[0:600, 150:750]
    cv2.imshow("Merged",mergedImage)

    keyPressed = cv2.waitKey(1)  # Look for keys to be pressed 
    if keyPressed == 27: # if the key is ESC, check the ASCII table, 27 = ESC
        break # Exit the loop

cv2.destroyAllWindows() # Destroy the windows and close the program
