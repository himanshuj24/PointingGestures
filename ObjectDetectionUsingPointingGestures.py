import cv2
import mediapipe as mp
import time
import numpy as np

#Initial point for the mouse click
point = (0,0)
#Width and Height values are divied by 2 in the following assignment. For eg:- to get a bounding box of 100*150, following values are given
target_w = 50
target_h = 75

#Funtion which takes the point as input and returns a bounding box by taking given point as the centroid
def target_model(point):
    # Creating 100*150 rectangle around the clicked point as the centroid.
    cv2.rectangle(img, (point[0] - target_w, point[1] - target_h), (point[0] + target_w, point[1] + target_h), (0, 0, 255), 2)
    cv2.circle(img, (point[0], point[1]), 2, (0, 0, 255), 2)
    text = "Location of target"
    cv2.putText(img, text, (point[0] - target_w, point[1] - (target_h+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# change the value '0' as per selected input device
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
cx1 = 0
cy1 = 0
cx2 = 0
cy2 = 0
cx3 = 0
cy3 = 0
temp_x = 0
temp_y = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # masking height of interest around the line of pointing gesture
    mask_h = 30
    x_shift= 20
    kp_orb_thresh= 100
    orb = cv2.ORB_create()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id == 6:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                    cx3 = cx
                    cy3 = cy
                if id == 7:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                    cx2 = cx
                    cy2 = cy
                if id == 8:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                    cx1 = cx
                    cy1 = cy

                if cx1!=0 and cx2!=0 and cx1!= cx2:

                    ratio = (cy2-cy1) / (cx2-cx1)

                    if cx1>cx2 :
                        temp_x = w
                        temp_y = ((temp_x - cx1) * (ratio)) + cy1
                        if 0 < temp_y < h:
                            temp_y = int(temp_y)
                            points_bb = np.array([[cx1+x_shift, int(cy1+(mask_h/2))], [temp_x, int(temp_y+(mask_h/2))], [temp_x, int(temp_y-(mask_h/2))], [cx1+x_shift, int(cy1-(mask_h/2))]])
                            cv2.fillPoly(mask, pts=[points_bb], color=(255, 255, 255)),
                            kp_orb = orb.detect(gray, mask)

                            if len(kp_orb) > kp_orb_thresh:
                                x_sum = 0
                                y_sum = 0
                                for i in kp_orb:
                                    x_sum = x_sum + i.pt[0]
                                    y_sum = y_sum + i.pt[1]
                                point= (int(x_sum/len(kp_orb)), int(y_sum/len(kp_orb)))
                                target_model(point)
                                img = cv2.drawKeypoints(img, kp_orb, img)
                                # cv2.putText(img, "SiftPoints " + str(int(len(kp_orb))), (20, 150),
                                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            continue

                    elif cx1<cx2:
                        temp_x = 0
                        temp_y = ((temp_x - cx1) * (ratio)) + cy1
                        if 0 < temp_y < h:
                            temp_y = int(temp_y)
                            points_bb = np.array([[cx1-x_shift, int(cy1+(mask_h/2))], [temp_x, int(temp_y+(mask_h/2))], [temp_x, int(temp_y-(mask_h/2))], [cx1-x_shift, int(cy1-(mask_h/2))]])
                            cv2.fillPoly(mask, pts=[points_bb], color=(255, 255, 255))
                            kp_orb = orb.detect(gray, mask)

                            if len(kp_orb) > kp_orb_thresh:
                                x_sum = 0
                                y_sum = 0
                                for i in kp_orb:
                                    x_sum = x_sum + i.pt[0]
                                    y_sum = y_sum + i.pt[1]
                                point = (int(x_sum / len(kp_orb)), int(y_sum / len(kp_orb)))
                                target_model(point)
                                img = cv2.drawKeypoints(img, kp_orb, img)
                                # cv2.putText(img, "SiftPoints " + str(int(len(kp_orb))), (20, 150),
                                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            continue

                    if cy1>cy2:
                        temp_y = h
                        temp_x = ((temp_y - cy1) / ratio) + cx1
                        if 0 < temp_x < w:
                            temp_x = int(temp_x)
                            points_bb = np.array([[cx1+x_shift, int(cy1+(mask_h/2))], [temp_x, int(temp_y+(mask_h/2))], [temp_x, int(temp_y-(mask_h/2))], [cx1+x_shift, int(cy1-(mask_h/2))]])
                            cv2.fillPoly(mask, pts=[points_bb], color=(255, 255, 255))
                            kp_orb = orb.detect(gray, mask)

                            if len(kp_orb) > kp_orb_thresh:
                                x_sum = 0
                                y_sum = 0
                                for i in kp_orb:
                                    x_sum = x_sum + i.pt[0]
                                    y_sum = y_sum + i.pt[1]
                                point = (int(x_sum / len(kp_orb)), int(y_sum / len(kp_orb)))
                                target_model(point)
                                img = cv2.drawKeypoints(img, kp_orb, img)
                                # cv2.putText(img, "SiftPoints " + str(int(len(kp_orb))), (20, 150),
                                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            continue

                    elif cy1<cy2:
                        temp_y = 0
                        temp_x = ((temp_y- cy1)/ratio) + cx1
                        if 0 < temp_x < w:
                            temp_x = int(temp_x)
                            points_bb = np.array([[cx1-x_shift, int(cy1 + (mask_h / 2))], [temp_x, int(temp_y + (mask_h / 2))],[temp_x, int(temp_y - (mask_h / 2))], [cx1-x_shift, int(cy1 - (mask_h / 2))]])
                            cv2.fillPoly(mask, pts=[points_bb], color=(255, 255, 255))
                            kp_orb = orb.detect(gray, mask)

                            if len(kp_orb) > kp_orb_thresh:
                                x_sum = 0
                                y_sum = 0
                                for i in kp_orb:
                                    x_sum = x_sum + i.pt[0]
                                    y_sum = y_sum + i.pt[1]
                                point = (int(x_sum / len(kp_orb)), int(y_sum / len(kp_orb)))
                                target_model(point)
                                img = cv2.drawKeypoints(img, kp_orb, img)
                                # cv2.putText(img, "SiftPoints " + str(int(len(kp_orb))), (20, 150),
                                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            continue

            line = cv2.line(img, (cx1, cy1), (temp_x, temp_y), (0, 255, 0), 3)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, "FPS " + str(int(fps)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Image", img)
    # cv2.imshow("MASK", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# if __name__ == '__main__':



