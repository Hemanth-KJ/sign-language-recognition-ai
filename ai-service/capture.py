import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ==========================
# MEDIA PIPE
# ==========================
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1
)

pose = mp_pose.Pose()

face = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False
)

# ==========================
# DATASET PATH
# ==========================
dataset_path = "../datasets"

os.makedirs(
    dataset_path,
    exist_ok=True
)

# ==========================
# CAMERA
# ==========================
cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    print("Camera not found")
    exit()

# ==========================
# INPUT
# ==========================
sign_name = input(
    "Enter sign/action name: "
).strip()

save_path = os.path.join(
    dataset_path,
    f"{sign_name}.npy"
)

print("\nDataset Save Path:")
print(save_path)

# ==========================
# RECORD SETTINGS
# ==========================
RECORD_TIME = 20

recording = False

start_time = 0

data = []

print("\nPress S to Start Recording")

# ==========================
# LOOP
# ==========================
while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(
        frame,
        1
    )

    # ======================
    # BLACK WHITE
    # ======================
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    frame = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    hand_res = hands.process(rgb)
    pose_res = pose.process(rgb)
    face_res = face.process(rgb)

    row = []

    # ======================
    # HAND
    # ======================
    if hand_res.multi_hand_landmarks:

        for hand in hand_res.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            for lm in hand.landmark:

                row.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

    else:
        row.extend([0] * 63)

    # ======================
    # POSE
    # ======================
    if pose_res.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            pose_res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        for lm in pose_res.pose_landmarks.landmark:

            row.extend([
                lm.x,
                lm.y,
                lm.z
            ])

    else:
        row.extend([0] * 99)

    # ======================
    # FACE
    # ======================
    if face_res.multi_face_landmarks:

        for face_landmarks in face_res.multi_face_landmarks:

            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=(255, 255, 255),
                    thickness=1,
                    circle_radius=1
                )
            )

            for lm in face_landmarks.landmark:

                row.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

    else:

        row.extend(
            [0] * (468 * 3)
        )

    # ======================
    # RECORD
    # ======================
    if recording:

        elapsed = (
            time.time()
            - start_time
        )

        remaining = max(
            0,
            int(
                RECORD_TIME
                - elapsed
            )
        )

        data.append(row)

        cv2.putText(
            frame,
            f"RECORDING {remaining}s",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # AUTO STOP
        if elapsed >= RECORD_TIME:

            recording = False

            np.save(
                save_path,
                np.array(data)
            )

            print("\nSaved")

            print(
                save_path
            )

            print(
                np.array(data).shape
            )

            break

    else:

        cv2.putText(
            frame,
            "PRESS S TO START",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # ======================
    # TOP BAR
    # ======================
    cv2.rectangle(
        frame,
        (0, 0),
        (900, 50),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"SIGN : {sign_name}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "AI Capture System",
        frame
    )

    key = (
        cv2.waitKey(1)
        & 0xFF
    )

    # START
    if (
        key == ord("s")
        and not recording
    ):

        data = []

        start_time = time.time()

        recording = True

        print(
            "\nRecording Started"
        )

    # EXIT
    if key == ord("q"):
        break

# ==========================
# CLOSE
# ==========================
cap.release()

cv2.destroyAllWindows()