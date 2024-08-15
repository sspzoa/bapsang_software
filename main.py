from utils import *
import subprocess

net = cv2.dnn.readNet(MODEL_WEIGHTS, MODEL_CONFIG)
classes = load_classes()
output_layers = get_output_layers(net)

cap = cv2.VideoCapture(0)

current_direction = ""

def speak(text):
    subprocess.run(["say", text])

def extract_number(direction):
    return direction.split()[0]

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = process_detections(outs, width, height, classes)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    current_direction = draw_predictions(frame, boxes, class_ids, indexes, width, height)

    cv2.imshow("Image", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        if current_direction:
            number_only = extract_number(current_direction)
            speak(f"사과가 {number_only}시 방향에 있습니다.")
        else:
            speak("No apple detected")

cap.release()
cv2.destroyAllWindows()