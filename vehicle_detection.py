import cv2
import numpy as np

RECT_MIN = 80
RECT_HEIGHT = 80
AREA_MIN = 500  

OFFSET = 6
CENTER_COORDINATES = []
VEHICLE_IN_COUNTER = 0
VEHICLE_OUT_COUNTER = 0

LINE_POSITION_Y = 550
LINE_IN_X_1 = 25
LINE_IN_X_2 = 600
LINE_OUT_X_1 = 600
LINE_OUT_X_2 = 1200


THRESHOLD = 50
INVERT_COLORS = False

def getCenter(x, y, w, h):
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy


cap = cv2.VideoCapture('video.mp4')


algorithm = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
algorithm.setShadowValue(0)
algorithm.setShadowThreshold(0.5)

while True:
    ret, frame1 = cap.read()
    if not ret:
        break
    
    frame1 = cv2.resize(frame1, (1280, 720))

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = algorithm.apply(blur)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation = cv2.morphologyEx(img_sub, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (LINE_IN_X_1, LINE_POSITION_Y), (LINE_IN_X_2, LINE_POSITION_Y), (0, 0, 255), 2)
    cv2.line(frame1, (LINE_IN_X_1, LINE_POSITION_Y + OFFSET), (LINE_IN_X_2, LINE_POSITION_Y + OFFSET), (255, 255, 255), 1)
    cv2.line(frame1, (LINE_IN_X_1, LINE_POSITION_Y - OFFSET), (LINE_IN_X_2, LINE_POSITION_Y - OFFSET), (255, 255, 255), 1)
    cv2.line(frame1, (LINE_OUT_X_1, LINE_POSITION_Y), (LINE_OUT_X_2, LINE_POSITION_Y), (0, 255, 0), 2)
    cv2.line(frame1, (LINE_OUT_X_1, LINE_POSITION_Y + OFFSET), (LINE_OUT_X_2, LINE_POSITION_Y + OFFSET), (255, 255, 255), 1)
    cv2.line(frame1, (LINE_OUT_X_1, LINE_POSITION_Y - OFFSET), (LINE_OUT_X_2, LINE_POSITION_Y - OFFSET), (255, 255, 255), 1)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if w >= RECT_MIN and h >= RECT_HEIGHT and area >= AREA_MIN:
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 255, 255), 2)
            center_coord = getCenter(x, y, w, h)
            CENTER_COORDINATES.append(center_coord)
            cv2.circle(frame1, center_coord, 4, (0, 0, 255), -1)

            increment_value = min(3, w // RECT_MIN)


    updated_center_coordinates = []
    for (cx, cy) in CENTER_COORDINATES:
        if LINE_IN_X_1 < cx < LINE_IN_X_2 and (LINE_POSITION_Y - OFFSET) < cy < (LINE_POSITION_Y + OFFSET):
            VEHICLE_IN_COUNTER += increment_value
            
            continue
        elif LINE_OUT_X_1 < cx < LINE_OUT_X_2 and (LINE_POSITION_Y - OFFSET) < cy < (LINE_POSITION_Y + OFFSET):
            VEHICLE_OUT_COUNTER += increment_value
            
            continue
        
        updated_center_coordinates.append((cx, cy))

    
    CENTER_COORDINATES = updated_center_coordinates


    if VEHICLE_IN_COUNTER >= THRESHOLD or VEHICLE_OUT_COUNTER >= THRESHOLD:
        INVERT_COLORS = not INVERT_COLORS  
        VEHICLE_IN_COUNTER = 0  
        VEHICLE_OUT_COUNTER = 0

    
    if (VEHICLE_IN_COUNTER > VEHICLE_OUT_COUNTER) ^ INVERT_COLORS:
     
        cv2.circle(frame1, (100, 100), 40, (0, 255, 0), -1)  
        cv2.circle(frame1, (1180, 100), 40, (0, 0, 255), -1)  
    else:
        
        cv2.circle(frame1, (100, 100), 40, (0, 0, 255), -1)  
        cv2.circle(frame1, (1180, 100), 40, (0, 255, 0), -1)  

    
    cv2.putText(frame1, "In: " + str(VEHICLE_IN_COUNTER), (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(frame1, "Out: " + str(VEHICLE_OUT_COUNTER), (800, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

  
    cv2.imshow("Foreground Mask", dilation)
    cv2.imshow("Vehicle Counter", frame1)

    if cv2.waitKey(1) == 27:  
        break

cv2.destroyAllWindows()
cap.release()
