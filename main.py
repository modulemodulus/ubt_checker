import cv2
import numpy as np
import utils

questions = 10
choices = 5
widthImg = 250
heightImg = 500

final_ans = [1, 0, 3, 1, 2, 2, 1, 1, 3, 1]

img = cv2.imread("12333.jpg")

img = cv2.resize(img, (widthImg, heightImg))

imgBlank = np.zeros((widthImg, heightImg, 3), np.uint8)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(img, 80, 100)
imgContours = img.copy()
imgBigContour = img.copy()
imgFinal = img.copy()

contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 5)

rectCon = utils.rectContour(contours)
print(len(rectCon))
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
    imgThresh = cv2.threshold(imgWarpGray, 130, 255, cv2.THRESH_BINARY_INV)[1]  # APPLY THRESHOLD AND INVERSE

    boxes = utils.splitBoxes(imgThresh, questions, choices)
    for x in range(10):
        del boxes[x][0]

    #for x in range(10):
    #    for y in range(4):
    #        cv2.imshow("test", boxes[x][y])
    #        cv2.waitKey(0)

    myPixelVal = np.zeros((questions, choices))

    for x in range(questions):
        for y in range(choices-1):
            myPixelVal[x][y] = cv2.countNonZero(boxes[x][y])

    myAns = []
    score = 0
    grading = np.zeros(questions)
    for x in range(questions):
        myAnsIndex = myPixelVal[x]
        qq = np.where(myAnsIndex == np.amax(myAnsIndex))
        myAns.append(qq[0][0])
        if myAns[x]==final_ans[x]:
            score += 1
            grading[x]=1

    total_score = score/questions * 100

    print("----------------")
    print(imgWarpColored.shape[0], imgWarpColored.shape[1])
    print("----------------")

    utils.showAnswers(imgWarpColored, myAns, grading, final_ans, questions=questions, choices=choices)
    imgRawDrawings = np.zeros_like(imgWarpColored)
    utils.showAnswers(imgRawDrawings, myAns, grading, final_ans, questions=questions, choices=choices)

    invMatrix = cv2.getPerspectiveTransform(pts2, pts1)  # INVERSE TRANSFORMATION MATRIX
    imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))  # INV IMAGE WARP

    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)



cv2.imshow("test", imgFinal)
cv2.waitKey(0)