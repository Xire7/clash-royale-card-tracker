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
        Enemy clock detection with openCV for circular gold clocks with red pixels
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # gold
        lower_gold = np.array([15, 150, 80])
        upper_gold = np.array([30, 255, 255])
        gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
        
        # red
        lower_red1 = np.array([0, 80, 50])
        upper_red1 = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1)
        
        # close gold ring gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        gold_closed = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(gold_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
            
            # check red inside
            red_roi = red_mask[y:y+h, x:x+w]
            red_count = np.sum(red_roi > 0)
            
            # check each filter
            area_pass = 400 < area < 6000
            aspect_pass = 0.6 < aspect < 1.5
            red_pass = red_count > 100
            circ_pass = circularity > 0.7

            # labels on debug images
            if area_pass and aspect_pass:
                if red_pass and circ_pass:
                    # all pass - GREEN
                    clock_regions.append((x, y, w, h))
                    cv2.rectangle(debug_result, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(debug_result, f"CLOCK {len(clock_regions)}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(debug_result, f"A:{area:.0f} R:{red_count} C:{circularity:.2f}", 
                            (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                else:
                    # failed red or circularity - RED
                    reasons = []
                    if not red_pass:
                        reasons.append(f"R:{red_count}")
                    if not circ_pass:
                        reasons.append(f"C:{circularity:.2f}")
                    
                    cv2.rectangle(debug_result, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(debug_result, " ".join(reasons), (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                # failed area or aspect - YELLOW
                reasons = []
                if not area_pass:
                    reasons.append(f"ID: {i}")
                    reasons.append(f"A:{area:.0f}")
                if not aspect_pass:
                    reasons.append(f"AS:{aspect:.2f}")
                
                cv2.rectangle(debug_result, (x, y), (x+w, y+h), (0, 165, 255), 1)
                cv2.putText(debug_result, " ".join(reasons), (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)
        
        print(f"{len(clock_regions)} CLOCKS DETECTED\n")
        
        if show_debug:
            cv2.imshow("1. Gold Mask (original)", gold_mask)
            cv2.imshow("2. Gold Mask (closed)", gold_closed)
            cv2.imshow("3. Red Mask", red_mask)
            cv2.imshow("4. ALL Contours (blue)", debug_all_contours)  # NEW!
            cv2.imshow("5. Result (filtered)", debug_result)
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