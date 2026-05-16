# ================== IMPORTS ==================
import os, cv2, dlib, math, numpy as np, tensorflow as tf
from collections import deque
import queue, threading
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

sys.path.append(os.path.join(PROJECT_ROOT, "data_collection"))
from constants import *

# ================== SPEAKER ==================
try:
    import pyttsx3
except:
    pyttsx3 = None

def speaker_worker(q):
    if pyttsx3 is None:
        return
    engine = pyttsx3.init()
    while True:
        text = q.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        q.task_done()

speech_queue = queue.Queue()
threading.Thread(target=speaker_worker, args=(speech_queue,), daemon=True).start()

# ================== MODEL ==================
label_dict = {
    6:'hello',5:'dog',10:'my',12:'you',9:'lips',
    3:'cat',11:'read',0:'a',4:'demo',7:'here',
    8:'is',1:'bye',2:'can'
}

input_shape = (TOTAL_FRAMES, 80, 112, 3)

model = tf.keras.Sequential([
    tf.keras.layers.Conv3D(16,(3,3,3),activation='relu',input_shape=input_shape),
    tf.keras.layers.MaxPooling3D((2,2,2)),
    tf.keras.layers.Conv3D(64,(3,3,3),activation='relu'),
    tf.keras.layers.MaxPooling3D((2,2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_dict),activation='softmax')
])

model.load_weights(os.path.join("model","model_weights.h5"))

# ================== DETECTOR ==================
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join("model","face_weights.dat"))

cap = cv2.VideoCapture(0)

# ================== VARIABLES ==================
curr_frames = []
past_frames = deque(maxlen=PAST_BUFFER_SIZE)
not_talking = 0

# 🔥 NEW (STABILITY)
CONF_THRESHOLD = 0.75
VOTE_BUFFER = deque(maxlen=5)
LAST_OUTPUT = None

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # ===== LIP DISTANCE =====
        top = landmarks.part(51)
        bottom = landmarks.part(57)
        lip_dist = math.hypot(bottom.x-top.x, bottom.y-top.y)

        # ===== LIP REGION =====
        l = landmarks.part(48).x
        r = landmarks.part(54).x
        t = landmarks.part(50).y
        b = landmarks.part(58).y

        lip = frame[t:b, l:r]
        lip = cv2.resize(lip, (112,80))

        if lip_dist > 45:
            curr_frames.append(lip)
            not_talking = 0
        else:
            not_talking += 1

            # ===== PREDICT =====
            if not_talking > NOT_TALKING_THRESHOLD and len(curr_frames) > VALID_WORD_THRESHOLD:

                seq = list(past_frames) + curr_frames
                seq = np.array([seq[:TOTAL_FRAMES]])

                pred = model.predict(seq, verbose=0)[0]

                max_prob = np.max(pred)
                pred_idx = np.argmax(pred)
                word = label_dict[pred_idx]

                # 🔥 CONFIDENCE CHECK
                if max_prob > CONF_THRESHOLD:
                    VOTE_BUFFER.append(word)

                # 🔥 VOTING (STABILITY)
                if len(VOTE_BUFFER) == VOTE_BUFFER.maxlen:
                    final_word = max(set(VOTE_BUFFER), key=VOTE_BUFFER.count)

                    # 🔥 DUPLICATE SUPPRESSION
                    if final_word != LAST_OUTPUT:
                        print("FINISHED!", final_word)
                        speech_queue.put(final_word)
                        LAST_OUTPUT = final_word

                    VOTE_BUFFER.clear()

                curr_frames = []
                not_talking = 0

        past_frames.append(lip)

        # ===== DRAW =====
        cv2.putText(frame, f"Talking: {lip_dist:.1f}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

    cv2.imshow("Silent Speech", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
speech_queue.put(None)