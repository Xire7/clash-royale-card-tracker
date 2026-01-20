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

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    kernel = np.ones((7,7), np.uint8)

    lower_gold = np.array([10, 72, 167])
    upper_gold = np.array([17, 205, 250])

    gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
    gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel, iterations=3)


    lower_red = np.array([172, 170, 80])
    upper_red = np.array([177, 255, 230])

    red_mask = cv2.inRange(hsv, lower_red, upper_red)


    contours, _ = cv2.findContours(gold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 300
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

        red_roi = red_mask[by:by+h, bx:bx+w]
        red_count = np.sum(red_roi > 0)

        red_ratio = red_count / area

        print(f"red_ratio: {red_ratio:.1%}")

        if 0.2 < red_ratio < 0.3:
            cv2.drawContours(img, [contour], -1, (0,255,0), 2)
            cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (255,0,0), 2)
            cv2.putText(img, f"red: {red_ratio:.1%}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("Gold Mask", gold_mask)
    cv2.imshow("Red Mask", red_mask)

    
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

sample_hsv_colors('evo1.png')