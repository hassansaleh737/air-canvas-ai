# file: hand_tracker.py

import cv2
import mediapipe as mp


class HandTracker:
    def __init__(
        self,
        max_hands=2,
        detection_confidence=0.7,
        tracking_confidence=0.7
    ):
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )

    def find_hands(self, frame, draw=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks and draw:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        return frame, results

    def get_landmark_positions(self, frame, hand_landmarks):
        h, w, _ = frame.shape
        landmarks = []

        for landmark_id, landmark in enumerate(hand_landmarks.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((landmark_id, x, y))

        return landmarks

    def fingers_up(self, landmarks):
        """
        Returns:
        [thumb, index, middle, ring, pinky]
        1 = finger up
        0 = finger down
        """

        if not landmarks or len(landmarks) < 21:
            return [0, 0, 0, 0, 0]

        fingers = []

        # Thumb - not very important for our drawing logic
        if landmarks[4][1] > landmarks[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Index finger
        fingers.append(1 if landmarks[8][2] < landmarks[6][2] else 0)

        # Middle finger
        fingers.append(1 if landmarks[12][2] < landmarks[10][2] else 0)

        # Ring finger
        fingers.append(1 if landmarks[16][2] < landmarks[14][2] else 0)

        # Pinky finger
        fingers.append(1 if landmarks[20][2] < landmarks[18][2] else 0)

        return fingers

    def get_index_finger_tip(self, landmarks):
        if not landmarks or len(landmarks) < 9:
            return None

        _, x, y = landmarks[8]
        return x, y

    def get_hand_label(self, results, hand_index):
        """
        Returns hand label from MediaPipe:
        'Left' or 'Right'

        ملاحظة:
        بما إننا عاملين flip للصورة عشان تبقى زي المراية،
        ممكن أحيانًا تحس إن الاسم معكوس حسب الكاميرا.
        لو حصل كده، هنبدلهم بسهولة من main.py.
        """

        if not results.multi_handedness:
            return f"Hand_{hand_index}"

        label = results.multi_handedness[hand_index].classification[0].label
        return label