import cv2
import detector
import numpy as np
import torch
import sys

def sample_hsv_colors(image_path):
    """
    Click on the image to see its HSV values
    """
    detected_log = None
    img = cv2.imread(image_path)
    height = img.shape[0]

    # img = img[:height//2, :]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    debug_contours = hsv.copy()
    kernel = np.ones((3,3), np.uint8)

    lower_brown = np.array([10, 110, 70])
    upper_brown = np.array([14, 155, 210])

    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel)
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_CLOSE, kernel)

    lower_silver = np.array([106, 36, 90])
    upper_silver = np.array([120, 88, 240])

    silver_mask = cv2.inRange(hsv, lower_silver, upper_silver)
    silver_mask = cv2.morphologyEx(silver_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 2000
    margin = 10
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        bx = x - margin
        by = y - margin
        bw = w + 2*margin
        bh = h + 2*margin

        if area < min_area:
            continue

        silver_roi = silver_mask[by:by+bh, bx:bx+bw]
        silver_count = np.sum(silver_roi > 0)

        metal_ratio = silver_count / area

        cv2.putText(debug_contours, f"Metal: {metal_ratio: .1%}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.rectangle(debug_contours, (bx, by), (bx+bw, by+bh), (255,0,0), 2)

        print("Area reqs met")
        if 0.15 < metal_ratio < 0.20:
            print("Metal ratio:", metal_ratio)
            detected_log = contour
            cv2.drawContours(img, [detected_log], -1, (0, 255, 0), 2)
            cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (255,0,0), 2)

            # bounding box for label
            cv2.putText(img, f"Log w/ metal: {metal_ratio: .1%}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)


    cv2.drawContours(debug_contours, contours, -1, (255, 0, 0), 2)

    cv2.imshow("Brown Mask", brown_mask)

    cv2.imshow("Silver Mask", silver_mask)

    cv2.imshow("Contours Debugged", debug_contours)
    
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

sample_hsv_colors('log3.png')