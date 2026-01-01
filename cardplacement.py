import cv2
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
        red_clock_regions = self.detect_red_clocks(frame)

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

    def detect_red_clocks(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # establish lower & upper bounds for red range
        lower_red1 = np.array([0, 150, 150]) # bright red
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 150, 150])
        upper_red2 = np.array([180, 255, 255])

        # create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        red_mask = mask1 | mask2

        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        clock_regions = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # filter by size
            if 300 < area < 3000:
                x, y, w, h = cv2.boundingRect(contour)

                # filter by shape (clocks are circular, aspect ratio ~1)
                aspect_ratio = w / float(h)

                if 0.6 < aspect_ratio < 1.4:
                    perimeter= cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0

                    if circularity > 0.5: # reasonably circular
                        clock_regions.append((x, y, w, h))
        
        return clock_regions