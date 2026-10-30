from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
from typing import TYPE_CHECKING

from .config import (
    POSES, SHOTS_PER_POSE, AUTO_STABLE_FRAMES, MIN_BRIGHTNESS, MAX_BRIGHTNESS,
    MIN_BLUR_SCORE, MIN_FACE_WIDTH, MAX_FACE_WIDTH, MAX_CENTER_OFFSET_X,
    MAX_CENTER_OFFSET_Y, ENFORCE_HEAD_POSE, YAW_FRONT_MAX, PITCH_FRONT_MAX,
    YAW_TURN_MIN, YAW_TURN_MAX, PITCH_TURN_MIN, PITCH_TURN_MAX
)

if TYPE_CHECKING:
    from .i18n import LanguageManager


class FaceGeometryAnalyzer:
    def __init__(
        self,
        face_detector: mp.solutions.face_detection.FaceDetection,
        face_mesh: mp.solutions.face_mesh.FaceMesh,
        i18n: LanguageManager | None = None
    ):
        self.face_detector = face_detector
        self.face_mesh = face_mesh
        self.i18n = i18n

    def _tr(self, key: str, default: str) -> str:
        # Funkcja odpowiedzialna za pobranie przetłumaczonego tekstu, jeśli i18n jest dostępne.
        if self.i18n:
            return self.i18n.tr(key, default)
        return default

    def estimate_head_pose(self, frame: np.ndarray) -> tuple[bool, float, float, float]:
        try:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb)

            if not result.multi_face_landmarks:
                return False, 0.0, 0.0, 0.0

            landmarks = result.multi_face_landmarks[0].landmark

            left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h], dtype=np.float64)
            right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h], dtype=np.float64)

            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]

            eye_roll = float(np.degrees(np.arctan2(dy, dx)))
            if eye_roll > 90:
                eye_roll -= 180
            elif eye_roll < -90:
                eye_roll += 180

            eye_mid = (left_eye + right_eye) / 2.0
            mouth_mid = np.array([
                (landmarks[61].x * w + landmarks[291].x * w) / 2.0,
                (landmarks[61].y * h + landmarks[291].y * h) / 2.0,
            ], dtype=np.float64)
            face_center = eye_mid + 0.5 * (mouth_mid - eye_mid)

            nose = np.array([landmarks[1].x * w, landmarks[1].y * h], dtype=np.float64)
            nose_vec = nose - face_center

            roll_rad = np.deg2rad(-eye_roll)
            cos_r = np.cos(roll_rad)
            sin_r = np.sin(roll_rad)

            corrected_dx = nose_vec[0] * cos_r - nose_vec[1] * sin_r
            corrected_dy = nose_vec[0] * sin_r + nose_vec[1] * cos_r

            eye_distance = max(np.linalg.norm(right_eye - left_eye), 1.0)
            ref = max(eye_distance * 0.8, 1.0)

            yaw_deg = float(np.clip(np.degrees(np.arctan2(corrected_dx, ref)), -45.0, 45.0))
            pitch_deg = float(np.clip(np.degrees(np.arctan2(corrected_dy, ref)), -35.0, 35.0))

            return True, yaw_deg, pitch_deg, eye_roll
        except Exception:
            return False, 0.0, 0.0, 0.0

    def check_expected_pose(self, pose_idx: int, yaw: float, pitch: float, roll: float) -> tuple[bool, str]:
        if pose_idx >= len(POSES):
            return True, self._tr("pose_ok", "Pozycja OK")

        pose_key, _ = POSES[pose_idx]

        if pose_key in {"front", "frontv2", "pose_front", "pose_frontv2"}:
            if abs(yaw) <= YAW_FRONT_MAX and abs(pitch) <= PITCH_FRONT_MAX:
                return True, self._tr("msg_head_straight_ok", "Dobrze, głowa prosto")
            return False, self._tr("msg_head_straight_fix", "Ustaw głowę bardziej prosto")

        if pose_key in {"left", "pose_left"}:
            if yaw > -YAW_TURN_MIN:
                return False, self._tr("msg_head_left_fix", "Obróć całą głowę lekko w lewo")
            if yaw < -YAW_TURN_MAX:
                return False, self._tr("msg_head_left_too_much", "Za mocno w lewo, wróć trochę")
            return True, self._tr("msg_head_left_ok", "Dobrze, głowa lekko w lewo")

        if pose_key in {"right", "pose_right"}:
            if yaw < YAW_TURN_MIN:
                return False, self._tr("msg_head_right_fix", "Obróć całą głowę lekko w prawo")
            if yaw > YAW_TURN_MAX:
                return False, self._tr("msg_head_right_too_much", "Za mocno w prawo, wróć trochę")
            return True, self._tr("msg_head_right_ok", "Dobrze, głowa lekko w prawo")

        if pose_key in {"up", "pose_up"}:
            if abs(yaw) > YAW_FRONT_MAX:
                return False, self._tr("msg_head_up_no_turn", "Nie skręcaj głowy na bok")
            if pitch > -PITCH_TURN_MIN:
                return False, self._tr("msg_head_up_fix", "Unieś całą głowę lekko do góry")
            if pitch < -PITCH_TURN_MAX:
                return False, self._tr("msg_head_up_too_much", "Za mocno do góry, wróć trochę")
            return True, self._tr("msg_head_up_ok", "Dobrze, głowa lekko do góry")

        if pose_key in {"down", "pose_down"}:
            if abs(yaw) > YAW_FRONT_MAX:
                return False, self._tr("msg_head_down_no_turn", "Nie skręcaj głowy na bok")
            if pitch < PITCH_TURN_MIN:
                return False, self._tr("msg_head_down_fix", "Opuść całą głowę lekko w dół")
            if pitch > PITCH_TURN_MAX:
                return False, self._tr("msg_head_down_too_much", "Za mocno w dół, wróć trochę")
            return True, self._tr("msg_head_down_ok", "Dobrze, głowa lekko w dół")

        return True, self._tr("pose_ok", "Pozycja OK")

    def assess_frame_quality(self, frame: np.ndarray, current_pose_idx: int) -> tuple[bool, str, tuple | None]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if brightness < MIN_BRIGHTNESS:
            return False, self._tr("too_dark", "Za ciemno"), None
        if brightness > MAX_BRIGHTNESS:
            return False, self._tr("too_jasno", "Za jasno"), None
        if blur_score < MIN_BLUR_SCORE:
            return False, self._tr("image_blurry", "Obraz nieostry"), None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_detector.process(rgb)

        if not res.detections:
            return False, self._tr("no_face_detected", "Nie wykryto twarzy"), None

        det = max(res.detections, key=lambda d: float(d.score[0]))
        rb = det.location_data.relative_bounding_box

        x, y, bw, bh = rb.xmin, rb.ymin, rb.width, rb.height
        cx, cy = x + bw / 2, y + bh / 2

        if bw < MIN_FACE_WIDTH:
            return False, self._tr("come_closer", "Podejdź bliżej"), None
        if bw > MAX_FACE_WIDTH:
            return False, self._tr("step_back", "Odsuń się trochę"), None
        if abs(cx - 0.5) > MAX_CENTER_OFFSET_X:
            return False, self._tr("center_face", "Ustaw twarz na środku"), None
        if abs(cy - 0.5) > MAX_CENTER_OFFSET_Y:
            return False, self._tr("adjust_face_height", "Ustaw twarz wyżej / niżej"), None

        pose_ok, yaw, pitch, roll = self.estimate_head_pose(frame)
        if not pose_ok:
            return False, self._tr("head_pose_failed", "Nie udało się określić pozycji głowy"), None

        expected_ok, pose_msg = self.check_expected_pose(current_pose_idx, yaw, pitch, roll)
        pose_msg = f"{pose_msg} | yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}"

        if ENFORCE_HEAD_POSE and not expected_ok:
            return False, pose_msg, None

        bbox = (int(x * w), int(y * h), int(bw * w), int(bh * h))
        return True, pose_msg, bbox

    def draw_face_guide(self, frame: np.ndarray, ok_quality: bool, msg: str,
                        bbox=None, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
                        head_detected: bool = False, user_dir=None, current_pose_idx: int = 0,
                        current_shot_idx: int = 0, stable_count: int = 0) -> np.ndarray:
        h, w = frame.shape[:2]
        color = (0, 220, 0) if ok_quality else (0, 0, 255)

        center = (w // 2, h // 2)
        axes = (int(w * 0.28), int(h * 0.38))
        cv2.ellipse(frame, center, axes, 0, 0, 360, color, 3)

        if bbox is not None:
            bx, by, bw, bh = bbox
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)

        if head_detected:
            debug_text = f"Yaw: {yaw:5.1f}  Pitch: {pitch:5.1f}  Roll: {roll:5.1f}"
            cv2.putText(frame, debug_text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 0), 2, cv2.LINE_AA)

        return frame
    
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # pil_image = Image.fromarray(frame_rgb)
        # draw = ImageDraw.Draw(pil_image)

        # try:
        #     font = ImageFont.truetype("arial.ttf", 32)
        #     font_small = ImageFont.truetype("arial.ttf", 24)
        #     font_tiny = ImageFont.truetype("arial.ttf", 20)
        # except Exception:
        #     font = font_small = font_tiny = ImageFont.load_default()

        # draw.text((30, 50), msg, fill=tuple(reversed(color)), font=font)

        # if head_detected:
        #     debug_text = f"Yaw:{yaw:6.1f}° Pitch:{pitch:6.1f}°"
        #     draw.text((30, 10), debug_text, fill=(255, 200, 0), font=font_tiny)

        # if user_dir is not None and current_pose_idx < len(POSES):
        #     pose_key, pose_default = POSES[current_pose_idx]
        #     pose_text = self._tr(pose_key, pose_default)
        #     photo_str = self._tr("photo", "zdjęcie")

        #     text = f"{pose_text} | {photo_str} {current_shot_idx + 1}/{SHOTS_PER_POSE}"
        #     draw.text((30, 95), text, fill=(255, 255, 255), font=font_small)

        #     hold_pos_str = self._tr("hold_position", "Trzymaj pozycję")
        #     stable_text = f"{hold_pos_str}: {stable_count}/{AUTO_STABLE_FRAMES}"
        #     draw.text((30, 135), stable_text, fill=(255, 255, 255), font=font_tiny)

        # return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)