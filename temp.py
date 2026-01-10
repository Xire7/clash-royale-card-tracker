import cv2
import detector
import numpy as np
import torch
import sys

def sample_hsv_colors(image_path):
    """
    Click on the image to see its HSV values
    """
    img = cv2.imread(image_path)
    height = img.shape[0]
    width = img.shape[1]

    # REPLAY CONSTANTS
    KY1_FACTOR_REPLAY = 2/10
    KY2_FACTOR_REPLAY = 4/15

    KX1_FACTOR_REPLAY = 4/10
    KX2_FACTOR_REPLAY = 6/10

    # REAL GAME CONSTANTS
    KY1_FACTOR_GAME = 1/14
    KY2_FACTOR_GAME = 2/14

    KX1_FACTOR_GAME = 4/10
    KX2_FACTOR_GAME = 6/10


    img = img[int(height*KY1_FACTOR_REPLAY):int(height*KY2_FACTOR_REPLAY), int(width*KX1_FACTOR_REPLAY):int(width*KX2_FACTOR_REPLAY)] # replay king tower coordinates
    # img = img[int(height*1/14):int(height*2/14), int(width*4/10):int(width*6/10)] # actual game king tower coordinates

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    kernel = np.ones((3,3), np.uint8)

    lower_orange = np.array([16, 100, 250])
    upper_orange = np.array([30, 255, 255])

    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 600
    margin = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        bx = x - margin
        by = y - margin
        bw = w + 2*margin
        bh = h + 2*margin

        if area < min_area:
            continue

        cv2.drawContours(img, [contour], -1, (0,255,0), 2)
        cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (255,0,0), 2)
        cv2.putText(img, f"area: {area}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("Orange Mask", orange_mask)

    # cv2.imshow("Contours Debugged", debug_contours)
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get pixel at click location
            pixel_hsv = hsv[y, x]
            pixel_bgr = img[y, x]
            
            h, s, v = pixel_hsv
            b, g, r = pixel_bgr
            
            print(f"Position: ({x}, {y})")
            print(f"  BGR: ({b}, {g}, {r})")
            print(f"  HSV: ({h}, {s}, {v})")
            print(f"  HSV (degrees): H:{h*2}°, S:{s*100/255:.1f}%, V:{v*100/255:.1f}%")
            print()
    
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)
    
    while True:
        cv2.imshow('Image', img)
        if cv2.waitKey(1) == 27:  # ESC
            break
    
    cv2.destroyAllWindows()

    return

sample_hsv_colors('fireball10.png')