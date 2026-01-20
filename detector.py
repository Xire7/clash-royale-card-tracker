import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import os
from datetime import datetime
import mss
import argparse
from torchvision import models, transforms
from PIL import Image

# empirical full screen constants
MONITOR_LEFT = 940
MONITOR_TOP = 32
MONITOR_WIDTH = 760
MONITOR_HEIGHT = 1360

# cycle indicator color constants
OK_COLOR = (115,179,59) # 2-4 cards away
CLOSE_COLOR = (48, 119, 186) # 1 card away
IN_CYCLE_COLOR = (59, 59, 179)

# text color
WHITE_COLOR = (255,255,255)

# cooldowns
TROOP_DETECTION_COOLDOWN = 0.3
LOG_DETECTION_COOLDOWN = 3
FIREBALL_DETECTION_COOLDOWN = 3

# KING TOWER: REPLAY CONSTANTS
KY1_FACTOR_REPLAY = 2/10
KY2_FACTOR_REPLAY = 4/15

KX1_FACTOR_REPLAY = 5/12
KX2_FACTOR_REPLAY = 7/12

# KING TOWER: REAL GAME CONSTANTS
KY1_FACTOR_GAME = 1/14
KY2_FACTOR_GAME = 2/14

KX1_FACTOR_GAME = 4/10
KX2_FACTOR_GAME = 6/10

# EVO CONSTANTS
EVO_DELAY_FRAMES = 7
EVO_THRESHOLD = 0.65

def load_classifier(model_path, device='cuda'):
    print(torch.cuda.is_available())
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        device = 'cpu'
        
    # load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)


    # get model type from config in best_model dict

    config = checkpoint.get('config', {})
    model_type = config.get('model_type', 'mobilenet_v2')

    print(f"Loading model type: {model_type}")

    if model_type == 'resnet18':
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_type == 'resnet34':
        model = models.resnet34(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_type == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


    # load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # mimic validation transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"Model loaded successfully: {model_type} with {num_classes} classes")
    return model, class_names, transform

class DeploymentDetector:
    def __init__(self, model_path=None, save_detections=True, device='cuda'):
        self.save_detections = save_detections
        self.detection_count = 0
        self.last_detection_time = 0
        self.last_log_detection_time = 0
        self.last_fireball_detection_time = 0
        self.last_detected_cards = []
        self.detection_cooldown = 0.3
        self.pending_evo_classifications = []

        # card tracker variables
        self.slots = ["None", "None", "None", "None"]
        self.slot_count = [-1,-1,-1,-1]
        
        # Auto-detect CUDA
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA not available, using CPU")
            device = 'cpu'
        self.device = device

        # persistent overlay
        self.detection_history = [] # list of (bbox, card, confidence, frames_remaining)
        self.persistence_frames = 90

        if save_detections:
            self.output_dir = f"detections/detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Detections will be saved to: {self.output_dir}")
        
        if model_path and os.path.exists(model_path):
            self.model, self.class_names, self.transform = load_classifier(model_path, device)
            self.classify_enabled = True
        else:
            self.model = None
            self.class_names = None
            self.transform = None
            self.classify_enabled = False
            if model_path:
                print(f"Warning: model path '{model_path}' not found. Classifier disabled")

    def detect_opponent_clocks(self, frame):
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

                red_pass = 0.03 < red_ratio < 0.18 # should be only a little red

                # 0.03 - 0.20

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
        
        return detections, white_mask, red_mask

    # TODO: edge cases - misses middle logs from the enemy when one tower down, mirrored log is beyond cooldown constant
    def detect_log_spell(self, frame, time):
        frame = frame.copy() # explicit copy
        log_detected = False
        fh = frame.shape[0]
        frame = frame[:fh//2, :] # only observe to the end of top half
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            bx = x - margin
            by = y - margin
            bw = w + 2*margin
            bh = h + 2*margin

            silver_roi = silver_mask[by:by+bh, bx:bx+bw]
            silver_count = np.sum(silver_roi > 0)

            metal_ratio = silver_count / area

            if 0.15 < metal_ratio < 0.20:
                log_detected = True
                cv2.drawContours(debug_contours, [contour], -1, (0, 255, 0), 2)
                # bounding box for label
                cv2.rectangle(debug_contours, (bx, by), (bx+bw, by+bh), (255,0,0), 2)
                cv2.putText(debug_contours, f"Log w/ metal: {metal_ratio: .1%}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                os.makedirs("log_detections", exist_ok=True)
                cv2.imwrite(f"log_detections/detected_log_{time}.png", debug_contours)
                break

        return log_detected
    
    # TODO: add a CV heuristic game_type_checker, REPLAY or LIVE, so king tower crop is accurate for fireball check
    def detect_fireball_spell(self, frame, time):
        frame = frame.copy()
        fireball_detected = False

        height = frame.shape[0]
        width = frame.shape[1]

        frame = frame[int(height*KY1_FACTOR_REPLAY):int(height*KY2_FACTOR_REPLAY), int(width*KX1_FACTOR_REPLAY):int(width*KX2_FACTOR_REPLAY)] # replay king tower coordinates

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        kernel = np.ones((3,3), np.uint8)

        lower_orange = np.array([16, 100, 250])
        upper_orange = np.array([30, 255, 255])

        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 1200
        max_area = 1600
        margin = 0
        min_circularity = 0.23 # fireball should be somewhat circular when first thrown
        max_circularity = 0.60 # fireball isnt super round

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            bx = x - margin
            by = y - margin
            bw = w + 2*margin
            bh = h + 2*margin

            if area < min_area or area > max_area:
                continue

            # calculate circularity

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter**2)

            if circularity < min_circularity or circularity > max_circularity:
                continue

            cv2.drawContours(frame, [contour], -1, (0,255,0), 2)
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255,0,0), 2)
            cv2.putText(frame, f"A: {area} C:{circularity}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            os.makedirs("fireball_detections", exist_ok=True)
            cv2.imwrite(f"fireball_detections/detected_fireball_{time}.png", frame)
            print(f"Fireball detected. Area: {area}, Circularity: {circularity}")
            fireball_detected = True
            break

        return fireball_detected
    
    def detect_evo_border(self, image):

        print("hello chat")
        kernel = np.ones((7,7), np.uint8)

        lower_gold = np.array([10, 72, 167])
        upper_gold = np.array([17, 205, 250])

        gold_mask = cv2.inRange(image, lower_gold, upper_gold)
        gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        lower_red = np.array([172, 170, 80])
        upper_red = np.array([177, 255, 230])

        red_mask = cv2.inRange(image, lower_red, upper_red)

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

            print(f"Red Ratio found: {red_ratio}")

            if 0.2 < red_ratio < 0.3:
                print(f"Evo border detected")
                return True
            
        return False




    def extract_troop_region(self, frame, clock_bbox):
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
        if not self.classify_enabled or troop_region.size == 0:
            return "Unknown", 0.0
        
        try:
            # convert BGR to RGB
            rgb_image = cv2.cvtColor(troop_region, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # apply transform
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            # inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

                predicted_class = self.class_names[predicted_idx.item()]
                confidence_score = confidence.item()
            
            return predicted_class, confidence_score
        except Exception as e:
            print(f"Classification error: {e}")
            return "Error", 0.0


    def run(self, monitor_region=None, show_debug=False):
        sct = mss.mss()

        if monitor_region is None:
            monitor = sct.monitors[1] # full screen
        else:
            monitor = monitor_region

        print("REAL-TIME CLOCK DETECTION" + (" + CLASSIFICATION" if self.classify_enabled else ""))
        print("=" * 50)
        print(f"Monitoring region: {monitor}")
        print(f"Device: {self.device.upper()}")
        print("\nControls:")
        print("  D - Debug mode (see bounding boxes)")
        print("  Q - Quit")
        print("\nDetections will be saved automatically when triggered.")
        print("=" * 50)


        paused = False
        fps_times = []

        while True:
            screenshot = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

            current_time = time.time()

            # create overlay with all persistent detections
            overlay = frame.copy()

            log_detected = False
            fireball_detected = False

            # log CV-heuristic detection
            if (current_time - self.last_log_detection_time > LOG_DETECTION_COOLDOWN):
                log_detected = self.detect_log_spell(frame, current_time)
                if log_detected:
                    self.last_log_detection_time = current_time

            # fireball CV-heuristic detection
            if (current_time - self.last_fireball_detection_time > FIREBALL_DETECTION_COOLDOWN):
                fireball_detected = self.detect_fireball_spell(frame, current_time)
                if fireball_detected:
                    self.last_fireball_detection_time = current_time

            stored_predictions = []

            # check pending EVOs and add to detections
            for pending_evo in self.pending_evo_classifications:
                pending_evo['frames_remaining'] -= 1

                if pending_evo['frames_remaining'] <= 0:
                    x,y,w,h = pending_evo['clock_bbox']
                    troop_region, (tx, ty, tw, th) = self.extract_troop_region(frame, (x,y,w,h))
                    predicted_card, confidence = self.classify_card(troop_region)

                    if confidence < EVO_THRESHOLD:
                        print(f"RETRIED, UNCERTAIN || 1st Prediction: {pending_evo['uncertain_card'].upper()} | 2nd Prediction: {predicted_card.upper()} .. Continuing with assumption ..")

                    frames_remaining = 90
                    red_ratio = -1

                    self._draw_bounding_box(overlay, x, y, w, h, tx, ty, tw, th, predicted_card, confidence, red_ratio, frames_remaining)
                    stored_predictions.append({
                        'clock_bbox': pending_evo['clock_bbox'],
                        'troop_bbox': (tx, ty, tw, th),
                        'card': predicted_card,
                        'confidence': confidence,
                        'red_ratio': red_ratio,
                        'frames_remaining': frames_remaining - 1
                    })
                    self.detection_count += 1
                    print(f"EVO(?) Detection #{self.detection_count} | Card: {predicted_card.upper()} ({confidence:.1%})")

                    troop_path = os.path.join(self.output_dir, f"troop_{self.detection_count:03d}_evo_flagged_{timestamp}.png")
                    cv2.imwrite(troop_path, troop_region)
                
            
            self.pending_evo_classifications = [c for c in self.pending_evo_classifications if c['frames_remaining'] > 0]


            # only classify and add to history at the end. create these bounding boxes first too
            if self.save_detections and (current_time - self.last_detection_time) > self.detection_cooldown:

                detections, white_mask, red_mask = self.detect_opponent_clocks(frame)

                for detection in detections:
                    x, y, w, h = detection['bbox']
                    red_ratio = detection['red_ratio']
                    frames_remaining = 90
                    troop_region, (tx, ty, tw, th) = self.extract_troop_region(frame, (x,y,w,h))
                    if self.classify_enabled:
                        predicted_card, confidence = self.classify_card(troop_region)
                    
                    if confidence < EVO_THRESHOLD or self.detect_evo_border(troop_region):
                        print(f"(EVO?) Predicted Card: {predicted_card.upper()}. Retrying in {EVO_DELAY_FRAMES} frames..")
                        self.pending_evo_classifications.append({
                            'clock_bbox': (x, y, w, h),
                            'frames_remaining': EVO_DELAY_FRAMES - 1,
                            'uncertain_card': predicted_card
                        })
                        self.last_detection_time = current_time
                        continue

                    self.detection_count += 1
                    self._draw_bounding_box(overlay, x, y, w, h, tx, ty, tw, th, predicted_card, confidence, red_ratio, frames_remaining)
                    stored_predictions.append({
                        'clock_bbox': (x, y, w, h),
                        'troop_bbox': (tx, ty, tw, th),
                        'card': predicted_card,
                        'confidence': confidence,
                        'red_ratio': red_ratio,
                        'frames_remaining': frames_remaining - 1
                    })
                    # console log
                    log_msg = f"[{timestamp}] Detection #{self.detection_count}"
                    if self.classify_enabled:
                        log_msg += f" | Card: {predicted_card.upper()} ({confidence:.1%})"
                    log_msg += f" | Red: {red_ratio:.1%}"
                    print(log_msg)

                    troop_path = os.path.join(self.output_dir, f"troop_{self.detection_count:03d}_{timestamp}.png")
                    cv2.imwrite(troop_path, troop_region)      
                    self.last_detection_time = current_time  

            for det in self.detection_history:
                x, y, w, h = det['clock_bbox']
                tx, ty, tw, th = det['troop_bbox']

                self._draw_bounding_box(overlay, x, y, w, h, tx, ty, tw, th, det['card'], det['confidence'], det['red_ratio'], det['frames_remaining'])

                # decrement frame counter
                det['frames_remaining'] -= 1
                    
            # remove expired detections
            self.detection_history = [d for d in self.detection_history if d['frames_remaining'] > 0]

            self.detection_history.extend(stored_predictions)
            # info overlay
            info_y = 30
            cv2.putText(overlay, f"Active: {len(self.detection_history)}", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Saved: {self.detection_count}", (10, info_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # calculate FPS
            fps_times.append(current_time)
            fps_times = [t for t in fps_times if current_time - t < 1.0]
            fps = len(fps_times)

            cv2.putText(overlay, f"FPS: {fps}", (10, info_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # status
            status_color = (0, 255, 0) if len(self.detection_history) > 0 else (255, 255, 255)
            status_text = f"Tracking {len(self.detection_history)}" if len(self.detection_history) > 0 else "Waiting..."
            cv2.putText(overlay, status_text,
                        (10, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            

            # card tracker
            predicted_cards = [p['card'] for p in stored_predictions]

            if log_detected:
                predicted_cards.append("Log")
                log_detected = False
            if fireball_detected:
                predicted_cards.append("Fireball")
                fireball_detected = False
            
            for card in predicted_cards:
                print(f"Predicted Card: {card}")

            if len(predicted_cards) > 0:
                self.last_detected_cards = predicted_cards

            for i in range(len(self.slots)):
                for card in predicted_cards:
                    if self.slot_count[i] == 0:
                        if self.slots[i] == card:
                            self.slot_count[i] = 4
                        else:
                            continue
                    else:
                        self.slot_count[i] -= 1

            self._draw_card_tracker(overlay)
            
            cropped_overlay = overlay.copy()

            ch = cropped_overlay.shape[0]
            cw = cropped_overlay.shape[1]

            cropped_overlay = cropped_overlay[int(ch*13/20):int(ch*17/20), :]

            cv2.imshow("Card Tracker Cropped", cropped_overlay)
            
            if show_debug:
                cv2.imshow("Card Tracker Full", overlay)
            
            # keyboard input handling
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('d'):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow("Card Tracker Full")
                print(f"Debug windows: {'ON' if show_debug else 'OFF'}")
            elif key == ord('1'):
                print(f"Slot 1 assigned to {predicted_cards}")
                self._track_card(0)
            elif key == ord('2'):
                print(f"Slot 2 assigned to {predicted_cards}")
                self._track_card(1)
            elif key == ord('3'):
                print(f"Slot 3 assigned to {predicted_cards}")
                self._track_card(2)
            elif key == ord('4'):
                print(f"Slot 4 assigned to {predicted_cards}")
                self._track_card(3)
        
        cv2.destroyAllWindows()
        print(f"\nTotal detections saved: {self.detection_count}")
        if self.save_detections:
            print(f"Detections are saved in folder: {self.output_dir}")
    
    def _track_card(self, key):
        if len(self.last_detected_cards) == 0:
            return
        
        self.slots[key] = self.last_detected_cards[0]
        self.slot_count[key] = 4
        if len(self.last_detected_cards) > 1:
            for i in range(key + 1, key + 1 + len(self.last_detected_cards)):
                self.slots[i] = self.last_detected_cards[i]
                self.slot_count[i] = 4

    
    def _draw_card_tracker(self, overlay):
            bw = 150
            by = MONITOR_HEIGHT - 400
            bh = 150
            gap = 32

            b1x = gap
            b2x = gap*2 + bw
            b3x = gap*3 + bw*2
            b4x = gap*4 + bw*3

            if self.slots[0] != "None":
                cv2.rectangle(overlay, (b1x, by), (b1x + bw, by + bh), self._determine_color(self.slot_count[0]), -1)
                cv2.putText(overlay, f"{self.slots[0]}:", (int(b1x + gap/2), int(by + bh/4)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE_COLOR, 2)
                cv2.putText(overlay, f"{self.slot_count[0]}", (int(b1x + bw/2 - gap), int(by + bh*0.8)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.5, WHITE_COLOR, 2)
            if self.slots[1] != "None":
                cv2.rectangle(overlay, (b2x, by), (b2x + bw, by + bh), self._determine_color(self.slot_count[1]), -1)
                cv2.putText(overlay, f"{self.slots[1]}:", (int(b2x + gap/2), int(by + bh/4)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE_COLOR, 2)
                cv2.putText(overlay, f"{self.slot_count[1]}", (int(b2x + bw/2 - gap), int(by + bh*0.8)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.5, WHITE_COLOR, 2)
            if self.slots[2] != "None":
                cv2.rectangle(overlay, (b3x, by), (b3x + bw, by + bh), self._determine_color(self.slot_count[2]), -1)
                cv2.putText(overlay, f"{self.slots[2]}:", (int(b3x + gap/2), int(by + bh/4)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE_COLOR, 2)
                cv2.putText(overlay, f"{self.slot_count[2]}", (int(b3x + bw/2 - gap), int(by + bh*0.8)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.5, WHITE_COLOR, 2)
            if self.slots[3] != "None":
                cv2.rectangle(overlay, (b4x, by), (b4x + bw, by + bh), self._determine_color(self.slot_count[3]), -1)
                cv2.putText(overlay, f"{self.slots[3]}:", (int(b4x + gap/2), int(by + bh/4)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE_COLOR, 2)
                cv2.putText(overlay, f"{self.slot_count[3]}", (int(b4x + bw/2 - gap), int(by + bh*0.8)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.5, WHITE_COLOR, 2)
            
    
    def _determine_color(self, count):
        if count > 2:
            return OK_COLOR
        if count > 0:
            return CLOSE_COLOR
        return IN_CYCLE_COLOR
    
    def _draw_bounding_box(self, overlay, x, y, w, h, tx, ty, tw, th, predicted_card, confidence, red_ratio, frames_remaining):
        # fade effect
        alpha = frames_remaining / self.persistence_frames
        color_intensity = int(255 * alpha)

        # draw clock bbox
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, color_intensity, color_intensity), 2)
        
        # draw troop region bbox
        cv2.rectangle(overlay, (tx, ty), (tx+tw, ty+th), (0, color_intensity, 0), 3)

        if self.classify_enabled:
            cv2.putText(overlay, f"{predicted_card.upper()}", (tx, ty-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, color_intensity, 0), 3)
            cv2.putText(overlay, f"{confidence:.1%}", (tx, ty+th+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, color_intensity, 0), 2)
        else:
            cv2.putText(overlay, f"TROOP", (tx, ty-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, color_intensity, 0), 2)


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

    if roi[2] > 0 and roi[3] > 0:  # valid
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
    parser.add_argument('--region', type=str, help='Monitor region as "x,y,width,height" (e.g. "100,100,800,600")')
    parser.add_argument('--select-region', action='store_true', help='Select region interactively')
    parser.add_argument('--model', type=str, help='Path to trained model .pth file', default=None)

    args = parser.parse_args()

    monitor_region = {'left': MONITOR_LEFT, 'top': MONITOR_TOP, 'width': MONITOR_WIDTH, 'height': MONITOR_HEIGHT}
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
            print("Invalid region format. Using default region.")
    
    detector = DeploymentDetector(model_path=args.model, save_detections=not args.no_save, device='cuda') # gigachad GPU user
    detector.run(monitor_region=monitor_region, show_debug=True)