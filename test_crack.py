# This code is performing crack detection on an image using OpenCV library in Python.
import cv2
import numpy as np

image = cv2.imread('test/crack3.jpg')

image = cv2.resize(image, (640, 640))

blue, red, green = cv2.split(image)

red_blur = cv2.GaussianBlur(red, (3, 3), 0)

_, red_binary = cv2.threshold(red_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

red_green = cv2.bitwise_and(green, red_binary)

edges = cv2.Canny(red_green, 100, 300)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

erosion = cv2.erode(closing, kernel, iterations=1)
result = cv2.subtract(closing, erosion)

_, result_binary = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, image_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

final = cv2.subtract(image_binary, result_binary)

final_binary = cv2.bitwise_not(final)

contours, hierarchy = cv2.findContours(
    final_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

if len(contours) > 400:
    print("The image contains a crack.")
else:
    print("The image does not contain a crack.")


cv2.imshow("Crack Detection", image)
# cv2.imshow("Gaussian Blur", red_blur)
# cv2.imshow("Masked", red_green)
# cv2.imshow("Edges", edges)
# cv2.imshow("Closing", closing)
# cv2.imshow("Subtract", result)
# cv2.imshow("Binary", result_binary)
# cv2.imshow("Final", final)
# cv2.imshow("Final", final_binary)

cv2.waitKey(0)
cv2.destroyAllWindows()