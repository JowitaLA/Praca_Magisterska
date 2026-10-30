from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .i18n import LanguageManager


logger = logging.getLogger("FaceRegisterApp")


def _tr(i18n: LanguageManager | None, key: str, default: str) -> str:
    # Funkcja pomocnicza do tłumaczenia tekstów poza klasami.
    if i18n:
        return i18n.tr(key, default)
    return default

def compute_ear_from_landmarks(
    landmarks,
    width: int,
    height: int
) -> float:

    indices = [
        33, 160, 158, 133, 144, 153,
        362, 385, 387, 263, 373, 380
    ]

    points = [
        np.array(
            [landmarks[i].x * width, landmarks[i].y * height],
            dtype=np.float32
        )
        for i in indices
    ]

    def calculate_eye_ear(pts):
        p1, p2, p3, p4, p5, p6 = pts

        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)

        if horizontal < 1e-6:
            return 0.0

        return float(
            (vertical_1 + vertical_2) / (2.0 * horizontal)
        )

    left_eye = points[:6]
    right_eye = points[6:]

    return float(
        (
            calculate_eye_ear(left_eye) +
            calculate_eye_ear(right_eye)
        ) / 2.0
    )

def compute_mouth_from_landmarks(
    landmarks,
    width: int,
    height: int
) -> float:

    upper = np.array(
        [landmarks[13].x * width, landmarks[13].y * height],
        dtype=np.float32
    )

    lower = np.array(
        [landmarks[14].x * width, landmarks[14].y * height],
        dtype=np.float32
    )

    return float(
        np.linalg.norm(lower - upper) /
        max(width, height, 1)
    )

def evaluate_liveness_signals(
    ear_values: list[float],
    motion_values: list[float],
    mouth_values: list[float],
    i18n: LanguageManager | None = None,
) -> tuple[bool, dict]:
    ear_values = [float(v) for v in ear_values if v is not None]
    motion_values = [float(v) for v in motion_values if v is not None]
    mouth_values = [float(v) for v in mouth_values if v is not None]

    if not ear_values and not motion_values and not mouth_values:
        return False, {
            "reason": _tr(i18n, "liveness_no_signals", "Brak sygnałów twarzy do oceny"),
            "ear_values": ear_values,
            "motion_values": motion_values,
            "mouth_values": mouth_values,
        }

    ear_min = min(ear_values) if ear_values else 0.0
    ear_max = max(ear_values) if ear_values else 0.0
    ear_change = ear_max - ear_min

    motion_min = min(motion_values) if motion_values else 0.0
    motion_max = max(motion_values) if motion_values else 0.0
    motion_change = motion_max - motion_min

    mouth_min = min(mouth_values) if mouth_values else 0.0
    mouth_max = max(mouth_values) if mouth_values else 0.0
    mouth_change = mouth_max - mouth_min

    blink_detected = ear_change >= 0.05 or (ear_max > 0.20 and ear_min < 0.16)
    micro_motion_detected = motion_change >= 0.008
    mouth_change_detected = mouth_change >= 0.04
    
    passed = sum((
        blink_detected,
        micro_motion_detected,
        mouth_change_detected,
    )) >= 2

    if passed:
        reasons = []
        if blink_detected:
            reasons.append(_tr(i18n, "liveness_blink", "mrugnięcie"))
        if micro_motion_detected:
            reasons.append(_tr(i18n, "liveness_motion", "mikroruch twarzy"))
        if mouth_change_detected:
            reasons.append(_tr(i18n, "liveness_mouth", "zmianę mimiki ust"))
        
        reasons_str = ", ".join(reasons)
        fmt = _tr(i18n, "liveness_detected_fmt", "Wykryto {reasons} w sygnałach liveness")
        reason = fmt.format(reasons=reasons_str)
    else:
        reason = _tr(i18n, "liveness_failed", "Nie wykryto co najmniej dwóch sygnałów żywotności")

    return passed, {
        "reason": reason,
        "ear_values": ear_values,
        "motion_values": motion_values,
        "mouth_values": mouth_values,
        "ear_change": ear_change,
        "motion_change": motion_change,
        "mouth_change": mouth_change,
    }

class LivenessChecker:
    def __init__(
        self,
        face_mesh: mp.solutions.face_mesh.FaceMesh,
        i18n: LanguageManager | None = None
    ):
        self.face_mesh = face_mesh
        self.i18n = i18n

    def _tr(self, key: str, default: str) -> str:
        if self.i18n:
            return self.i18n.tr(key, default)
        return default

    def _get_landmark_points(self, frame: np.ndarray, indices: list[int]) -> list[np.ndarray] | None:
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result or not result.multi_face_landmarks:
            return None

        h, w = frame.shape[:2]
        landmarks = result.multi_face_landmarks[0].landmark
        return [np.array([landmarks[idx].x * w, landmarks[idx].y * h], dtype=np.float32) for idx in indices]

    # def compute_ear(self, frame: np.ndarray) -> float | None:
    #     points = self._get_landmark_points(frame, [33, 160, 158, 133, 144, 153, 362, 385, 387, 263, 373, 380])
    #     if points is None:
    #         return None

    #     left_eye, right_eye = points[:6], points[6:]

    #     def ear(pts: list[np.ndarray]) -> float:
    #         p1, p2, p3, p4, p5, p6 = pts
    #         v1 = np.linalg.norm(p2 - p6)
    #         v2 = np.linalg.norm(p3 - p5)
    #         h = np.linalg.norm(p1 - p4)
    #         return float((v1 + v2) / (2.0 * h)) if h >= 1e-6 else 0.0

    #     return float((ear(left_eye) + ear(right_eye)) / 2.0)

    # def compute_motion(self, frame: np.ndarray, previous_frame: np.ndarray | None) -> float | None:
    #     if frame is None or previous_frame is None:
    #         return None
    #     curr_pts = self._get_landmark_points(frame, [1, 33, 61, 263, 291])
    #     prev_pts = self._get_landmark_points(previous_frame, [1, 33, 61, 263, 291])

    #     if curr_pts is None or prev_pts is None:
    #         return None

    #     h, w = frame.shape[:2]
    #     denom = max(w, h, 1)
    #     movement = [np.linalg.norm(curr - prev) / denom for curr, prev in zip(curr_pts, prev_pts)]
    #     return float(np.mean(movement)) if movement else None

    # def compute_mouth(self, frame: np.ndarray) -> float | None:
    #     points = self._get_landmark_points(frame, [13, 14])
    #     if points is None or len(points) < 2:
    #         return None
    #     h, w = frame.shape[:2]
    #     return float(np.linalg.norm(points[1] - points[0]) / max(w, h, 1))

    def check_liveness_frames(self, frames):
        ear_values = []
        motion_values = []
        mouth_values = []
        previous_nose = None

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb)

            if not result.multi_face_landmarks:
                continue

            landmarks = result.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            nose = np.array([landmarks[1].x, landmarks[1].y], dtype=np.float32)
            if previous_nose is not None:
                motion_values.append(float(np.linalg.norm(nose - previous_nose)))
            previous_nose = nose

            ear = compute_ear_from_landmarks(landmarks, w, h)
            mouth = compute_mouth_from_landmarks(landmarks, w, h)

            if ear is not None:
                ear_values.append(ear)
            if mouth is not None:
                mouth_values.append(mouth)

        passed, info = evaluate_liveness_signals(
            ear_values, motion_values, mouth_values, i18n=self.i18n
        )
        log = logger.info if passed else logger.warning
        log("Próba detekcji żywotności twarzy zakończona %s: %s",
            "pomyślnie" if passed else "niepowodzeniem",
            info["reason"])
        return passed, info

    # def check_liveness(self, current_frame: np.ndarray, cap: cv2.VideoCapture | None) -> tuple[bool, dict]:
        # if current_frame is None:
        #     return False, {
        #         "reason": self._tr("liveness_no_frame", "Brak klatki do sprawdzenia"),
        #         "ear_values": [],
        #         "motion_values": [],
        #         "mouth_values": []
        #     }

        # if cap is None or not cap.isOpened():
        #     return False, {
        #         "reason": self._tr("liveness_no_camera", "Kamera niedostępna"),
        #         "ear_values": [],
        #         "motion_values": [],
        #         "mouth_values": []
        #     }

        # ear_values, motion_values, mouth_values = [], [], []
        # frame = current_frame
        # previous_frame = None

        # time.sleep(0.05)
        
        # for _ in range(4):
        #     if cap is not None and cap.isOpened():
        #         ok, read_frame = cap.read()
        #         if ok and read_frame is not None:
        #             frame = read_frame

        #     ear = self.compute_ear(frame)
        #     if ear is not None:
        #         ear_values.append(ear)

        #     motion = self.compute_motion(frame, previous_frame)
        #     if motion is not None:
        #         motion_values.append(motion)

        #     mouth = self.compute_mouth(frame)
        #     if mouth is not None:
        #         mouth_values.append(mouth)

        #     previous_frame = frame

        # if len(ear_values) < 2 and not motion_values and not mouth_values:
        #     return False, {
        #         "reason": self._tr("liveness_face_not_detected_multiple", "Nie wykryto twarzy w kilku klatkach"),
        #         "ear_values": ear_values,
        #         "motion_values": motion_values,
        #         "mouth_values": mouth_values,
        #     }

        # return evaluate_liveness_signals(ear_values, motion_values, mouth_values, i18n=self.i18n)