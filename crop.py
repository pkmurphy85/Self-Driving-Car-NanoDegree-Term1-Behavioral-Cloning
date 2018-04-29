import cv2
img = cv2.imread("./Images/center_2018_04_15_17_34_49_863.jpg")
crop_img = img[75:75+65,0:320]
cv2.imshow("cropped", crop_img)
cv2.imwrite("cropped.png", crop_img)
