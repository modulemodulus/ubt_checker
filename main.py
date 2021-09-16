import cv2
import numpy as np
import utils

questions = 5
choices = 5
widthImg = 350
heightImg = 700

img = cv2.imread("12.jpg")

img = cv2.resize(img, (350, 700))

imgBlank = np.zeros((350, 700, 3), np.uint8)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(img, 80, 100)
imgContours = img.copy()
imgBigContour = img.copy()

contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 5)

rectCon = utils.rectContour(contours)

biggestPoints = utils.getCornerPoints(rectCon[0])

for con in rectCon:
    cv2.drawContours(imgBigContour, con, -1, (0, 0, 255), 5)

if biggestPoints.size != 0:

    biggestPoints = utils.reorder(biggestPoints)
    cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 0, 255), 5)

    pts1 = np.float32(biggestPoints)  # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GET TRANSFORMATION MATRIX
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))  # APPLY WARP PERSPECTIVE

    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)  # CONVERT TO GRAYSCALE
    imgThresh = cv2.threshold(imgWarpGray, 200, 255, cv2.THRESH_BINARY_INV)[1]  # APPLY THRESHOLD AND INVERSE

    boxes = utils.splitBoxes(imgThresh, questions, choices)
    #cv2.imshow("Split Boxes", boxes[2])

cv2.imshow("test", imgBigContour)
cv2.waitKey(0)