import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model


# Initialize models
mpHands = mp.solutions.hands  # performs the hand recognition algorithm
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


# Initialize Tensorflow
model = load_model('./models/mp_hand_gesture')


f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()


cap = cv2.VideoCapture(0)


while True:
    _, frame = cap.read()
    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)
    className = ''

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x*x)
                lmy = int(lm.y*y)

                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        prediction = model.predict([landmarks])
        print(prediction)
        class_id = np.argmax(prediction)
        class_name = classNames[class_id]
        cv2.putText(frame, class_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        print("Quitting")
        break


cap.release()
cv2.destroyAllWindows()