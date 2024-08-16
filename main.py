from utils import *
import subprocess

net = cv2.dnn.readNet(MODEL_WEIGHTS, MODEL_CONFIG)
classes = load_classes()
output_layers = get_output_layers(net)

cap = cv2.VideoCapture(0)

current_directions = {}

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

    current_directions = draw_predictions(frame, boxes, class_ids, indexes, width, height, classes)

    cv2.imshow("Image", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        speech_text = ""
        for target_class in TARGET_CLASSES:
            if target_class in current_directions and current_directions[target_class]:
                directions = [extract_number(d) for d in current_directions[target_class]]
                if len(directions) == 1:
                    speech_text += f"{target_class}가 {directions[0]}시 방향에 있습니다. "
                elif len(directions) > 1:
                    speech_text += f"{target_class}가 {', '.join(directions[:-1])} 그리고 {directions[-1]}시 방향에 있습니다. "

        if speech_text:
            speak(speech_text)
        else:
            speak("사과와 바나나가 감지되지 않았습니다.")

cap.release()
cv2.destroyAllWindows()