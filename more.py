"""
erodedImage = cv2.morphologyEx(maskedImage, cv2.MORPH_ERODE, kernel)
cv2.imshow("Eroded Image", erodedImage)
dilatedImage = cv2.morphologyEx(maskedImage, cv2.MORPH_DILATE, kernel)
cv2.imshow("Dilated Image", dilatedImage)

filledImage = openedImage.copy()
h, w = openedImage.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

cv2.floodFill(filledImage, mask, (0,0), 255)
invertedImage = cv2.bitwise_not(filledImage)
outImage = openedImage | invertedImage
outImage = cv2.bitwise_not(outImage)

cv2.imshow("Filled Image", filledImage)
cv2.imshow("Inverted Image", invertedImage)
cv2.imshow("Out Image", outImage)

There is also canny for edge detection

Machine Learning:
SVM + HOG

Descriptors
SIFT
SURF
ORB
FAST
BRIEF
FLANN
KNN
K-Means

Matching:
Brute Force
"""