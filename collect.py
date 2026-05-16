import os
import cv2
import dlib
import math
import json
import statistics
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import csv
from collections import deque
from constants import TOTAL_FRAMES, VALID_WORD_THRESHOLD, NOT_TALKING_THRESHOLD, PAST_BUFFER_SIZE, LIP_WIDTH, LIP_HEIGHT


# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("D:\\lip_tracking_project\\model\\face_weights.dat")

# read the image
cap = cv2.VideoCapture(0)

# storing all the collected data here
all_words = []

# temporary storage for each word
curr_word_frames = []

# counter
not_talking_counter = 0


data_count = 1
words = ["here", "is", "a", "demo", "can", "you", "read", "my", "lips", "cat", "dog", "hello", "bye"]
options = ", ".join(words)
label = input("What word you like to collect data for? The options are \n" + options + ": ")
labels = []

custom_distance = input("If you want, enter a custom lip distance threshold or -1: ")

clean_output_dir = input("To clean output directory of the current word, type 'yes': ")

# clear the directory if needed
if clean_output_dir == "yes":
    root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

    # 🔥 FIX: correct outputs_dir handling
    outputs_dir = os.path.join(root_dir, "outputs")

    if os.path.exists(outputs_dir):   # <-- FIX
        for folder_name in os.listdir(outputs_dir):
            folder_path = os.path.join(outputs_dir, folder_name)
            if os.path.isdir(folder_path) and label in folder_path:
                print(f"Removing folder {folder_name}...")
                import shutil
                shutil.rmtree(folder_path)


# circular buffer for storing "previous" frames
past_word_frames = deque(maxlen=PAST_BUFFER_SIZE)


# counter for number of frames needed to calibrate the not-talking lip distance
determining_lip_distance = 50

# store the not-talking lip distances when averaging
lip_distances = []

# threshold for determing if user is talking or not talking
LIP_DISTANCE_THRESHOLD = None

if custom_distance != -1 and custom_distance.isdigit() and int(custom_distance) > 0:
    custom_distance = int(custom_distance)
    determining_lip_distance = 0
    LIP_DISTANCE_THRESHOLD = custom_distance
    print("USING CUSTOM DISTANCE")

while True:

    ret, frame = cap.read()

    # 🔥 FIX: webcam safety
    if not ret or frame is None:
        continue

    # Convert image into grayscale
    rgb = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # 🔥 FIX: ensure proper format for dlib
    rgb = np.asarray(rgb, dtype=np.uint8, order='C')

    # Use detector to find landmarks
    faces = detector(rgb)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = predictor(image=rgb, box=face)

        mouth_top = (landmarks.part(51).x, landmarks.part(51).y)
        mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)
        lip_distance = math.hypot(mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1])

        lip_left = landmarks.part(48).x
        lip_right = landmarks.part(54).x
        lip_top = landmarks.part(50).y
        lip_bottom = landmarks.part(58).y

        if(determining_lip_distance != 0 and LIP_DISTANCE_THRESHOLD != None):

            width_diff = LIP_WIDTH - (lip_right - lip_left)
            height_diff = LIP_HEIGHT - (lip_bottom - lip_top)
            pad_left = width_diff // 2
            pad_right = width_diff - pad_left
            pad_top = height_diff // 2
            pad_bottom = height_diff - pad_top

            pad_left = min(pad_left, lip_left)
            pad_right = min(pad_right, frame.shape[1] - lip_right)
            pad_top = min(pad_top, lip_top)
            pad_bottom = min(pad_bottom, frame.shape[0] - lip_bottom)

            lip_frame = frame[lip_top - pad_top:lip_bottom + pad_bottom, lip_left - pad_left:lip_right + pad_right]
            lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))

            lip_frame_lab = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2LAB)

            l_channel, a_channel, b_channel = cv2.split(lip_frame_lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
            l_channel_eq = clahe.apply(l_channel)

            lip_frame_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
            lip_frame_eq = cv2.cvtColor(lip_frame_eq, cv2.COLOR_LAB2BGR)
            lip_frame_eq= cv2.GaussianBlur(lip_frame_eq, (7, 7), 0)
            lip_frame_eq = cv2.bilateralFilter(lip_frame_eq, 5, 75, 75)

            kernel = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])

            lip_frame_eq = cv2.filter2D(lip_frame_eq, -1, kernel)
            lip_frame_eq= cv2.GaussianBlur(lip_frame_eq, (5, 5), 0)
            lip_frame = lip_frame_eq

            for n in range(48, 61):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

            ORANGE =  (0, 180, 255)
            BLUE = (255, 0, 0)
            RED = (0, 0, 255)

            if lip_distance > LIP_DISTANCE_THRESHOLD:
                cv2.putText(frame, "Talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                curr_word_frames += [lip_frame.tolist()]
                not_talking_counter = 0

                cv2.putText(frame, "RECORDING WORD RIGHT NOW", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 2)

            else:
                cv2.putText(frame, "Not talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
                not_talking_counter += 1

                if not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE == TOTAL_FRAMES:

                    cv2.putText(frame, "NOT RECORDING WORD", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE, 2)

                    data_count += 1
                    curr_word_frames = list(past_word_frames) + curr_word_frames

                    print(f"adding {label.upper()} shape", lip_frame.shape, "count is", data_count, "frames is", len(curr_word_frames))

                    all_words.append(curr_word_frames)
                    labels.append(label)
                    curr_word_frames = []
                    not_talking_counter = 0

                elif not_talking_counter < NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE < TOTAL_FRAMES and len(curr_word_frames) > VALID_WORD_THRESHOLD:

                    cv2.putText(frame, "RECORDING WORD RIGHT NOW", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 2)
                    curr_word_frames += [lip_frame.tolist()]
                    not_talking_counter = 0

                elif len(curr_word_frames) < VALID_WORD_THRESHOLD or (not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE > TOTAL_FRAMES):

                    cv2.putText(frame, "NOT RECORDING WORD", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE, 2)
                    curr_word_frames = []

                elif not_talking_counter < NOT_TALKING_THRESHOLD:
                    cv2.putText(frame, "RECORDING WORD RIGHT NOW", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 2)
                else:
                    cv2.putText(frame, "NOT RECORDING WORD", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE, 2)

                past_word_frames+= [lip_frame.tolist()]

                if len(past_word_frames) > PAST_BUFFER_SIZE:
                    past_word_frames.pop(0)

        else:
            cv2.putText(frame, "KEEP MOUTH CLOSED, CALIBRATING DISTANCE BETWEEN LIPS", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            determining_lip_distance -= 1

            distance = landmarks.part(58).y - landmarks.part(50).y 
            cv2.putText(frame, "Current distance: " + str(distance + 2), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            lip_distances.append(distance)
            if(determining_lip_distance == 0):
                LIP_DISTANCE_THRESHOLD = sum(lip_distances) / len(lip_distances) + 2

    cv2.imshow("Mouth", frame)

    if cv2.waitKey(1) == 27:
        break