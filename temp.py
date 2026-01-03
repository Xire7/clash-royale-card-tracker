import cv2
import detector
import numpy as np

def sample_clock_colors(image_path):
    """
    Click on the clock to see its HSV values
    """
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
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
    
    print("Click on different parts of the gold ring to see HSV values")
    print("Press ESC to exit")
    
    while True:
        cv2.imshow('Image', img)
        if cv2.waitKey(1) == 27:  # ESC
            break
    
    cv2.destroyAllWindows()

def clock_detection_test(path):
    img = cv2.imread(path)

    detect = detector.DeploymentDetector(None, False)

    detections, white_mask, red_mask = detect.detect_opponent_clocks(img, show_debug=True)

    print(f"Found {len(detections)} red clocks")

    print("Detections:", detections)

    cv2.waitKey(0)

# sample_clock_colors('red_clock_006.png')
clock_detection_test('red_clock_008.png')
