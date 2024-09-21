#importing the libraries
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calc_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ab = np.subtract(a, b)
    bc = np.subtract(b, c)
    theta = np.arccos(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)))
    theta = np.degrees(theta)
    return theta

 # Counter 
def count_bends(angle, flag, count):
    if angle > 140 and flag != 'down':
        count += 1
        flag = 'down'
    elif angle < 40 and flag == 'down':
        flag = None
    return flag, count

flag_left = None
flag_right = None

right_count = 0
left_count = 0
cap = cv2.VideoCapture(0)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

while cap.isOpened():
    _, frame = cap.read()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Extract Landmarks
    try:
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        left_angle = calc_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calc_angle(right_shoulder, right_elbow, right_wrist)

        cv2.putText(image, f'Left Angle: {left_angle}', (9, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Right Angle: {right_angle}', (9,90 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        flag_left, left_count = count_bends(left_angle, flag_left, left_count)
        flag_right, right_count = count_bends(right_angle, flag_right, right_count)

    except Exception as e:
        print(f"Error: {e}")
        pass

    cv2.putText(image, f'Left Count: {left_count}', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (355, 100, 222), 2, cv2.LINE_AA)
    cv2.putText(image, f'Right Count: {right_count}', (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (355, 100, 222), 2, cv2.LINE_AA)

# Render Detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('MediaPipe feed', image)

    if cv2.waitKey(5) & 0xFF == ord('z'):
        break

cap.release()
cv2.destroyAllWindows()