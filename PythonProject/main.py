#gaze tracking
import cv2
import mediapipe as mp

import time
from collections import deque


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


LEFT_PUPIL = 468
RIGHT_PUPIL = 473
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
NOSE_TIP = 1


eye_direction_buffer = deque(maxlen=10)
eye_direction_start_time = None
EYE_DIRECTION_TIME_THRESHOLD = 7
EYE_DIRECTION_COUNT_THRESHOLD = 10

face_down_count = 0
FACE_DOWN_THRESHOLD = 3

face_lost_start_time = None
FACE_LOST_THRESHOLD = 15


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_lost_start_time = None

        for face_landmarks in results.multi_face_landmarks:

            left_pupil = face_landmarks.landmark[LEFT_PUPIL]
            right_pupil = face_landmarks.landmark[RIGHT_PUPIL]
            left_eye_inner = face_landmarks.landmark[LEFT_EYE_INNER]
            left_eye_outer = face_landmarks.landmark[LEFT_EYE_OUTER]
            right_eye_inner = face_landmarks.landmark[RIGHT_EYE_INNER]
            right_eye_outer = face_landmarks.landmark[RIGHT_EYE_OUTER]
            nose_tip = face_landmarks.landmark[NOSE_TIP]


            left_pupil_x = int(left_pupil.x * w)
            left_pupil_y = int(left_pupil.y * h)
            right_pupil_x = int(right_pupil.x * w)
            right_pupil_y = int(right_pupil.y * h)

            left_eye_inner_x = int(left_eye_inner.x * w)
            left_eye_outer_x = int(left_eye_outer.x * w)
            right_eye_inner_x = int(right_eye_inner.x * w)
            right_eye_outer_x = int(right_eye_outer.x * w)

            nose_tip_y = int(nose_tip.y * h)


            left_eye_width = left_eye_inner_x - left_eye_outer_x
            right_eye_width = right_eye_outer_x - right_eye_inner_x

            left_pupil_ratio = (left_pupil_x - left_eye_outer_x) / left_eye_width
            right_pupil_ratio = (right_pupil_x - right_eye_inner_x) / right_eye_width


            if left_pupil_ratio < 0.38 and right_pupil_ratio < 0.38:
                eye_direction = "Looking Left"
            elif left_pupil_ratio > 0.62 and right_pupil_ratio > 0.62:
                eye_direction = "Looking Right"
            elif 0.40 <= left_pupil_ratio <= 0.60 and 0.40 <= right_pupil_ratio <= 0.60:
                eye_direction = "Looking Center"
            else:
                eye_direction = "Looking Up"


            eye_direction_buffer.append(eye_direction)


            if len(eye_direction_buffer) == eye_direction_buffer.maxlen:
                if all(direct == eye_direction_buffer[0] for direct in eye_direction_buffer) and eye_direction != "Looking Center":
                    if eye_direction_start_time is None:
                        eye_direction_start_time = time.time()
                    elif time.time() - eye_direction_start_time > EYE_DIRECTION_TIME_THRESHOLD:
                        cv2.putText(frame, "WARNING: Eyes Fixed for Too Long!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    eye_direction_start_time = None


            if eye_direction_buffer.count(eye_direction) > EYE_DIRECTION_COUNT_THRESHOLD and eye_direction != "Looking Center":
                cv2.putText(frame, "WARNING: Repeated Eye Movement!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            if nose_tip_y > h * 0.6:
                face_down_count += 1
                if face_down_count > FACE_DOWN_THRESHOLD:
                    cv2.putText(frame, "WARNING: Looking Down Too Often!", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                face_down_count = 0


            cv2.circle(frame, (left_pupil_x, left_pupil_y), 3, (0, 255, 0), -1)
            cv2.circle(frame, (right_pupil_x, right_pupil_y), 3, (0, 255, 0), -1)


            cv2.putText(frame, eye_direction, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    else:

        if face_lost_start_time is None:
            face_lost_start_time = time.time()
        else:
            elapsed_time = time.time() - face_lost_start_time
            if elapsed_time > FACE_LOST_THRESHOLD:
                cv2.putText(frame, "WARNING: Face Not Detected!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow('AI Cheating Detector', frame)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

#side face tracking

#hand gesture tracking

#