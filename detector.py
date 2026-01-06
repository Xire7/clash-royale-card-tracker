import cv2
import torch
import numpy as np
import time
import os
from datetime import datetime
import mss
import argparse

class DeploymentDetector:
    def __init__(self, card_classifier_model, save_detections=True):
        self.card_classifier_model = card_classifier_model
        self.save_detections = save_detections
        self.detection_count = 0
        self.last_detection_time = 0
        self.detection_cooldown = 0.3

        if save_detections:
            self.output_dir = f"detections/detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Detections will be saved to: {self.output_dir}")

    def detect_opponent_clocks(self, frame, show_debug=False):
        """
        Enemy clock detection with openCV. detect high white concentrations with small red presence
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # clock white
        lower_white = np.array([2, 0, 200])
        upper_white = np.array([3, 13, 217])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # red
        lower_red1 = np.array([0, 80, 50])
        upper_red1 = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1)
        
        # clean white mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        white_mask = cv2.dilate(white_mask, kernel, iterations=2)
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # debug images
        debug_result = frame.copy()
        debug_all_contours = frame.copy()
        
        # draw all contours in blue first (before any filtering)
        cv2.drawContours(debug_all_contours, contours, -1, (255, 0, 0), 2)  # Blue
        
        detections = []
                
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / float(h) if h > 0 else 0
            
            # calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            area_pass = 400 < area < 800
            aspect_pass = 0.6 < aspect < 1.5
            circ_pass = circularity > 0.40

            if area_pass and aspect_pass and circ_pass:
                # check for red presence inside contour
                margin = 2
                check_x = max(0, x - margin)
                check_y = max(0, y - margin)
                check_w = min(w + 2*margin, frame.shape[1] - check_x)
                check_h = min(h + 2*margin, frame.shape[0] - check_y)

                red_roi = red_mask[check_y:check_y+check_h, check_x:check_x+check_w]
                red_count = np.sum(red_roi > 0)
                red_ratio = red_count / (check_w * check_h)

                red_pass = 0.03 < red_ratio < 0.20 # should be only a little red

                if red_pass:
                    # deployment detected
                    detections.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'red_ratio': red_ratio,
                        'circularity': circularity
                        })
                    cv2.rectangle(debug_result, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(debug_result, f"DEPLOY {len(detections)}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(debug_result, f"A:{area:.0f} R:{red_ratio:.2%} C:{circularity:.2f}", 
                            (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                else:
                    # white region but no/too much red
                    cv2.rectangle(debug_result, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(debug_result, f"R:{red_ratio:.2%}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                # failed basic filters
                reasons = []
                if not area_pass:
                    reasons.append(f"A:{area:.0f}")
                if not aspect_pass:
                    reasons.append(f"AS:{aspect:.2f}")
                if not circ_pass:
                    reasons.append(f"C:{circularity:.2f}")
                
                if reasons:  # only show if there was some white detected
                    cv2.rectangle(debug_result, (x, y), (x+w, y+h), (0, 165, 255), 1)
                    cv2.putText(debug_result, " ".join(reasons), (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)
                
        if show_debug:
            cv2.imshow("1. White Mask", white_mask)
            cv2.imshow("2. Red Mask", red_mask)
            cv2.imshow("3. ALL White Contours (blue)", debug_all_contours)
            cv2.imshow("4. Result (filtered)", debug_result)
            cv2.waitKey(1)
        
        return detections, white_mask, red_mask
    
    def extract_troop_region(self, frame, clock_bbox):
        """
        Given clock bounding box, extract the troop deploy region
        """
        OFFSET_Y = 10 # adjusting region upward to frame below/at half of the clock

        x, y, w, h = clock_bbox
        
        # bounding box: centered on clock, extends upward, includes half of the clock

        troop_w, troop_h = 170, 170

        clock_center = x + (w // 2)
        troop_x = max(0, clock_center - (troop_w // 2))
        troop_y = max(0, (y + OFFSET_Y) - troop_h) # extend downward by offset Y

        troop_region = frame[troop_y:troop_y+troop_h, troop_x:troop_x+troop_w]

        return troop_region, (troop_x, troop_y, troop_w, troop_h)

    
    def classify_card(self, troop_region):
        """
        Use CNN model to classify placed troop in region
        """
        pass


    def run(self, monitor_region=None, show_debug=False):
        """
        Real-time detection with overlay

        monitor_region: dict with keys 'top', 'left', 'width', 'height' for mss

        if none, captures entire screen
        """

        sct = mss.mss()

        if monitor_region is None:
            monitor = sct.monitors[1] # full screen
        else:
            monitor = monitor_region

        print("REAL-TIME CLOCK DETECTION")
        print("=" * 50)
        print(f"Monitoring region: {monitor}")
        print("\nControls:")
        print("  SPACE - Pause/Resume")
        print("  S - Save current detections manually")
        print("  D - Toggle debug windows")
        print("  Q - Quit")
        print("\nDetections will be saved automatically when triggered.")
        print("=" * 50)


        paused = False
        fps_times = []

        while True:
            if not paused:
                screenshot = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

                current_time = time.time()
                detections, white_mask, red_mask = self.detect_opponent_clocks(frame)

                # create overlay
                overlay = frame.copy()

                for detection in detections:
                    x, y, w, h = detection['bbox']
                    red_ratio = detection['red_ratio']

                    troop_region, (tx, ty, tw, th) = self.extract_troop_region(frame, (x,y,w,h))

                    # draw clock bounding box (yellow)

                    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(overlay, "CLOCK", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # draw troop region bounding box (green)

                    cv2.rectangle(overlay, (tx, ty), (tx+tw, ty+th), (0, 255, 0), 3)
                    cv2.putText(overlay, f"TROOP REGION", (tx, ty-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(overlay, f"Red: {red_ratio:.1%}", (tx, ty+th+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # TODO: simple cooldown per detection, will need to update later
                    if self.save_detections and (current_time - self.last_detection_time) > self.detection_cooldown:
                        self.detection_count += 1
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

                        # save full frame w/ overlay
                        full_path = os.path.join(self.output_dir, f"full_{self.detection_count:03d}_{timestamp}.png")
                        cv2.imwrite(full_path, overlay)

                        # save troop region (feed to CNN later)
                        troop_path = os.path.join(self.output_dir, f"troop_{self.detection_count:03d}_{timestamp}.png")
                        cv2.imwrite(troop_path, troop_region)
                        
                        self.last_detection_time = current_time
                        print(f"[{timestamp}] Detection #{self.detection_count} saved (Red: {red_ratio:.1%})")
                
                info_y = 30
                cv2.putText(overlay, f"Detections: {len(detections)}", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(overlay, f"Total Saved: {self.detection_count}", (10, info_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # calculate FPS
                fps_times.append(current_time)
                fps_times = [t for t in fps_times if current_time - t < 1.0]
                fps = len(fps_times)

                cv2.putText(overlay, f"FPS: {fps}", (10, info_y + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # status
                status_color = (0, 255, 0) if len(detections) > 0 else (255, 255, 255)
                cv2.putText(overlay, "DETECTING..." if len(detections) > 0 else "Waiting...",
                            (10, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                cv2.imshow("Deployment Detection", overlay)

                if show_debug:
                    cv2.imshow("Debug - White Mask", white_mask)
                    cv2.imshow("Debug - Red Mask", red_mask)
            else:
                paused_frame = overlay.copy()
                cv2.putText(paused_frame, "PAUSED (SPACE to resume)", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("Deployment Detection", paused_frame)
            
            # keyboard input handling
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord(' '):
                paused = not paused
                print("PPAUSED" if paused else "RESUMED")
            elif key == ord('s') and not paused:
                # manual save

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                manual_path = os.path.join(self.output_dir, f"manual_{timestamp}.png")
                cv2.imwrite(manual_path, frame)
                print(f"Manual save: {manual_path}")

            elif key == ord('d'):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow("White Mask")
                    cv2.destroyWindow("Red Mask")
                print(f"Debug windows: {'ON' if show_debug else 'OFF'}")
        
        cv2.destroyAllWindows()
        print(f"\n Total detections saved: {self.detection_count}")
        if self.save_detections:
            print(f"Detections are saved in folder: {self.output_dir}")


def select_region():
    sct = mss.mss()
    monitor = sct.monitors[1]
    screenshot = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    # let user select region
    print("\n=== REGION SELECTION ===")
    print("1. A window will open showing your full screen")
    print("2. Click and drag to select game window area")
    print("3. Press ENTER when done, ESC to cancel")

    roi = cv2.selectROI("Select Region of Interest (ROI)", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Region of Interest (ROI)")

    if roi[2] > 0 and roi[3] > 0:  # Valid selection
        x, y, w, h = roi
        region = {
            'left': int(x),
            'top': int(y),
            'width': int(w),
            'height': int(h)
        }
        print(f"\nSelected region: {region}")
        print(f"Command line argument: --region \"{x},{y},{w},{h}\"")
        return region
    else:
        print("No region selected, using full screen")
        return None
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time card deployment detector')
    parser.add_argument('--no-save', action='store_true', help='Do not save detections to disk')
    parser.add_argument('--region', type=str, help='Monitor region a "x,y,width,height" (e.g. "100,100,800,600")')
    parser.add_argument('--select-region', action='store_true', help='Select region interactively')

    args = parser.parse_args()

    monitor_region = {'left': 1043, 'top': 32, 'width': 760, 'height': 1360} # empirical full screen region for 1920x1080

    if args.select_region:
        monitor_region = select_region()
    elif args.region:
        try:
            parts = args.region.split(',')
            monitor_region = {
                'left': int(parts[0]),
                'top': int(parts[1]),
                'width': int(parts[2]),
                'height': int(parts[3])
            }
            print(f"Using custom monitor region: {monitor_region}")
        except:
            print("Invalid region format. Using full screen.")
    
    tester = DeploymentDetector(card_classifier_model=None, save_detections=not args.no_save)
    tester.run(monitor_region=monitor_region)

