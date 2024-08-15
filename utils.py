import math
import cv2
import numpy as np
from constants import *

def load_classes():
    with open(CLASSES_FILE, "r") as f:
        return [line.strip() for line in f.readlines()]

def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def get_clock_direction(center_x, center_y, frame_width, frame_height):
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    dx = -(center_x - frame_center_x)
    dy = -(center_y - frame_center_y)
    angle = math.atan2(dy, dx)
    angle_deg = math.degrees(angle)
    if angle_deg < 0:
        angle_deg += 360
    angle_deg = (angle_deg - 90) % 360
    clock_direction = round(angle_deg / 30)
    if clock_direction == 0:
        clock_direction = 12
    return f"{clock_direction} o'clock"

def process_detections(outs, width, height, classes):
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD and classes[class_id] == TARGET_CLASS:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

def draw_predictions(frame, boxes, class_ids, indexes, width, height):
    font = cv2.FONT_HERSHEY_PLAIN
    direction = ""
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = TARGET_CLASS
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            center_x = x + w // 2
            center_y = y + h // 2
            direction = get_clock_direction(center_x, center_y, width, height)
            cv2.putText(frame, f"{label}: {direction}", (x, y - 10), font, 1, color, 2)
    cv2.line(frame, (width//2, 0), (width//2, height), (255, 0, 0), 1)
    cv2.line(frame, (0, height//2), (width, height//2), (255, 0, 0), 1)
    cv2.putText(frame, "12", (width//2 - 10, 20), font, 1, (255, 0, 0), 2)
    cv2.putText(frame, "6", (width//2 - 10, height - 20), font, 1, (255, 0, 0), 2)
    cv2.putText(frame, "9", (20, height//2 + 10), font, 1, (255, 0, 0), 2)
    cv2.putText(frame, "3", (width - 20, height//2 + 10), font, 1, (255, 0, 0), 2)
    return direction