
import cv2
import numpy as np


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygon = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, polygon, 100)
    maskedimage = cv2.bitwise_and(image, mask)
    return maskedimage

def draw_the_lines(image, lines):
    blank_image = np.zeros_like(image)
    for line in lines:
         x1, y1, x2, y2 = line.reshape(4)
         cv2.line(blank_image, (x1, y1), (x2, y2), (255, 255, 255), 5)

    image = cv2.addWeighted(image, 0.8, blank_image, 1 , 1)
    return image

def process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    cropped_image = region_of_interest(canny)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), 30, 2)
    line_image = draw_the_lines(image, lines)
    return line_image

cap = cv2.VideoCapture("lane_test_video.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = process(frame)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

