import argparse
import json
import os
from collections import deque

import cv2
import numpy as np
import tensorflow as tf


Interpreter = tf.lite.Interpreter

TOTAL_FRAMES = 22
LIP_HEIGHT = 80
LIP_WIDTH = 112
NOT_TALKING_FRAMES = 8
DRAW_PRED_FRAMES = 20


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def center_crop(frame):
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = frame[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(cropped, (LIP_WIDTH, LIP_HEIGHT))


def preprocess(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
    l_eq = clahe.apply(l_channel)
    eq = cv2.merge((l_eq, a_channel, b_channel))
    eq = cv2.cvtColor(eq, cv2.COLOR_LAB2BGR)
    eq = cv2.GaussianBlur(eq, (5, 5), 0)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    eq = cv2.filter2D(eq, -1, kernel)
    eq = cv2.GaussianBlur(eq, (5, 5), 0)
    return eq.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipcam-url", required=True, help="IP camera URL (HTTP/RTSP).")
    parser.add_argument("--model-path", default=os.path.join("model", "lipread_model.tflite"))
    parser.add_argument("--labels-path", default=os.path.join("model", "labels.json"))
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Missing model: {args.model_path}")
    if not os.path.exists(args.labels_path):
        raise FileNotFoundError(f"Missing labels: {args.labels_path}")

    labels = load_labels(args.labels_path)

    # IMPORTANT: model contains Select TF Ops (FlexMaxPool3D), so TensorFlow runtime is required.
    interpreter = Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    cap = cv2.VideoCapture(args.ipcam_url)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open IP camera URL: {args.ipcam_url}")

    frame_buffer = deque(maxlen=TOTAL_FRAMES)
    not_talking_counter = 0
    draw_prediction = False
    draw_count = 0
    predicted_word_label = ""
    spoken_already = deque(maxlen=8)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("IP camera frame read failed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        movement = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F))))
        talking = movement > 8.0
        mouth_frame = preprocess(center_crop(frame))

        if talking:
            cv2.putText(frame, "Talking", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame_buffer.append(mouth_frame)
            not_talking_counter = 0
            draw_prediction = False
        else:
            cv2.putText(frame, "Not talking", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            not_talking_counter += 1
            if not_talking_counter >= NOT_TALKING_FRAMES and len(frame_buffer) == TOTAL_FRAMES:
                sample = np.array([list(frame_buffer)], dtype=np.float32)
                if input_details["dtype"] == np.uint8:
                    scale, zero_point = input_details["quantization"]
                    if scale > 0:
                        sample = (sample / scale + zero_point).astype(np.uint8)
                    else:
                        sample = sample.astype(np.uint8)

                interpreter.set_tensor(input_details["index"], sample)
                interpreter.invoke()
                out = interpreter.get_tensor(output_details["index"])[0]

                pred_idx = int(np.argmax(out))
                candidate = labels[pred_idx]
                if spoken_already and candidate == spoken_already[-1]:
                    ranked = np.argsort(out)[::-1]
                    for i in ranked:
                        alt = labels[int(i)]
                        if alt != spoken_already[-1]:
                            candidate = alt
                            break
                predicted_word_label = candidate
                spoken_already.append(predicted_word_label)
                print("FINISHED!", predicted_word_label)

                draw_prediction = True
                draw_count = 0
                frame_buffer.clear()
                not_talking_counter = 0
            elif len(frame_buffer) < TOTAL_FRAMES:
                frame_buffer.append(mouth_frame)

        if draw_prediction and draw_count < DRAW_PRED_FRAMES:
            draw_count += 1
            cv2.putText(frame, predicted_word_label, (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

        cv2.imshow("LipRead IP Cam (TFLite Flex)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            spoken_already.clear()
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
