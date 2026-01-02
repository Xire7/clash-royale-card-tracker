import cv2
import torch
import numpy as np
import time

class CardPlacementDetector:
    def __init__(self, card_classifier_model):
        self.card_classifier_model = card_classifier_model
        self.pending_detections = []
        self.last_detection_time = {}

    def detect_card_placements(self, frame):
        """
        Detection pipeline
        """
        current_time = time.time()
        detected_cards = []

        # step 1: find red clocks (heuristic color-based detection)
        red_clock_regions = self.detect_opponent_clocks(frame)

        # step 2: track these regions over time (ensure no duplicates)
        for (x, y, w, h) in red_clock_regions:
            region_key = f"{x}_{y}"

            # check if this is new detection
            if region_key not in self.last_detection_time:
                # store new detection
                self.pending_detections.append({
                    'region': (x,y,w,h),
                    'detected_at': current_time,
                    'classified': False
                })
                self.last_detection_time[region_key] = current_time

        # step 3: classify cards in regions where clock is now gone
        for detection in self.pending_detections:
            time_since_detection = current_time - detection['detected_at']

            if 0.5 <= time_since_detection <= 1.5 and not detection['classified']:
                x, y, w, h = detection['region']

                margin = 50
                troop_region = frame[
                    max(0, y-margin):y+h+margin,
                    max(0, x-margin):x+w+margin
                ]

                # step 4: use CNN to classify the card without clock

                if troop_region.size > 0:
                    card_type = self.classify_card(troop_region)
                    detected_cards.append(card_type)
                    detection['classified'] = True

        # only keep recent pending detections & new region tracking now
        self.pending_detections = [
            d for d in self.pending_detections
            if current_time - d['detected_at'] < 2.0
        ]

        self.last_detection_time = {
            k: v for k,v in self.last_detection-time.items()
            if current_time - v < 2.0
        }

        return detected_cards

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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        white_mask = cv2.dilate(white_mask, kernel, iterations=3)
        # white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # debug images
        debug_result = frame.copy()
        debug_all_contours = frame.copy()
        
        # draw all contours in blue first (before any filtering)
        cv2.drawContours(debug_all_contours, contours, -1, (255, 0, 0), 2)  # Blue
        
        clock_regions = []
                
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
            
            area_pass = 300 < area < 2000
            aspect_pass = 0.6 < aspect < 1.5
            circ_pass = circularity > 0.5

            if area_pass and aspect_pass and circ_pass:
                # check for red presence inside contour
                margin = 10
                check_x = max(0, x - margin)
                check_y = max(0, y - margin)
                check_w = min(w + 2*margin, frame.shape[1] - check_x)
                check_h = min(h + 2*margin, frame.shape[0] - check_y)

                red_roi = red_mask[check_y:check_y+check_h, check_x:check_x+check_w]
                red_count = np.sum(red_roi > 0)
                red_ratio = red_count / (check_w * check_h)

                red_pass = 0.03 < red_ratio < 0.5

                if red_pass:
                    # deployment detected
                    clock_regions.append((x, y, w, h))
                    cv2.rectangle(debug_result, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(debug_result, f"DEPLOY {len(clock_regions)}", (x, y-10),
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
        
        print(f"{len(clock_regions)} DEPLOYMENTS DETECTED\n")
        
        if show_debug:
            cv2.imshow("1. White Mask", white_mask)
            cv2.imshow("2. Red Mask", red_mask)
            cv2.imshow("3. ALL White Contours (blue)", debug_all_contours)
            cv2.imshow("4. Result (filtered)", debug_result)
            cv2.waitKey(1)
        
        return clock_regions

    
    def classify_card(self, troop_region):
        """
        Use CNN model to classify placed troop in region
        """

        transform = get_card_transform()
        tensor = transform(troop_region).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.card_classifier(tensor)
            _, predicted = torch.max(output, 1)

        card_names = ['hog_rider', 'ice_golem', 'musketeer', 'ice_spirit', 'skeletons', 'fireball', 'log', 'cannon']

        return card_names[predicted.item()]
    

    def visualize_detections(frame, clock_regions):
        """
        Draw boxes around detected red clocks for debugging
        """
        debug_frame = frame.copy()
        
        for (x, y, w, h) in clock_regions:
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_frame, "RED CLOCK", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Clock Detection", debug_frame)
        cv2.waitKey(1)
        
        return debug_frame