import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('Akatsuki.avi',fourcc,20.0,(640,480))
soap_film = cv2.imread('soap_film.jpeg')

backSub = cv2.createBackgroundSubtractorMOG2()
cap = cv2.VideoCapture(0)
time.sleep(5)
count = 0
background = 0
for i in range(60):
    ret, background = cap.read()
background = np.flip(background, axis=1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
soap_film = cv2.resize(soap_film, (width, height), interpolation=cv2.INTER_AREA)

while (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    count += 1
    img = np.flip(img, axis=1)
    blend = cv2.addWeighted(background, 0.8, soap_film, 0.4, 0)

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #
    # lower_red = np.array([0, 120, 50])
    # upper_red = np.array([10, 255, 255])
    # mask1 = cv2.inRange(hsv, lower_red, upper_red)
    #
    # lower_red = np.array([170, 120, 70])
    # upper_red = np.array([180, 255, 255])
    # mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask1 = backSub.apply(img)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    mask2 = cv2.bitwise_not(mask1)
    res1 = cv2.bitwise_and(img, img, mask=mask2)
    res2 = cv2.bitwise_and(blend, blend, mask=mask1)

    finalOutput = cv2.addWeighted(res1, 1, res2, 1, 0)
    out.write(finalOutput)
    cv2.imshow("akatsuki", finalOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()