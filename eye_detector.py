# -*- coding: utf-8 -*-
# @Time    : 2025/05/05
# @Author  : 技术导师 Copilot (@mosivez)
# @Module  : 眼睛和虹膜检测

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass


@dataclass
class EyePoint:
    """眼睛关键点数据结构"""
    left_eye_corner_left: tuple = None
    left_eye_corner_right: tuple = None
    right_eye_corner_left: tuple = None
    right_eye_corner_right: tuple = None
    left_eye_center: tuple = None
    right_eye_center: tuple = None
    left_iris_center: tuple = None
    right_iris_center: tuple = None
    left_offset: tuple = None
    right_offset: tuple = None


class EyeDetector:
    """眼睛和虹膜检测模块"""

    def __init__(self, config_manager):
        """
        初始化眼睛检测器

        Args:
            config_manager: 配置管理器实例
        """
        self.config = config_manager

        # MediaPipe初始化
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # 尝试使用refine_landmarks参数初始化
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=self.config.get('mediapipe', 'max_num_faces'),
                refine_landmarks=True,
                min_detection_confidence=self.config.get('mediapipe', 'min_detection_confidence'),
                min_tracking_confidence=self.config.get('mediapipe', 'min_tracking_confidence')
            )
        except TypeError:
            # 旧版本MediaPipe不支持refine_landmarks
            print("警告: 使用不带refine_landmarks的MediaPipe初始化（较旧版本）")
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=self.config.get('mediapipe', 'max_num_faces'),
                min_detection_confidence=self.config.get('mediapipe', 'min_detection_confidence'),
                min_tracking_confidence=self.config.get('mediapipe', 'min_tracking_confidence')
            )

        # 定义关键点索引
        self.L_EYE_CORNER_L = 33
        self.L_EYE_CORNER_R = 133
        self.R_EYE_CORNER_L = 362
        self.R_EYE_CORNER_R = 263
        self.L_IRIS_POINTS = [474, 475, 476, 477]  # MediaPipe虹膜点
        self.R_IRIS_POINTS = [469, 470, 471, 472]
        self.L_EYE_CENTER_APPROX_IDS = [self.L_EYE_CORNER_L, self.L_EYE_CORNER_R]
        self.R_EYE_CENTER_APPROX_IDS = [self.R_EYE_CORNER_L, self.R_EYE_CORNER_R]

        # 图像尺寸
        self.image_width = self.config.get('camera', 'width')
        self.image_height = self.config.get('camera', 'height')

    def detect_eyes(self, frame):
        """
        检测眼睛和虹膜位置

        Args:
            frame: 输入视频帧

        Returns:
            landmarks: MediaPipe面部关键点
            eye_points: 眼睛关键点数据
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        eye_points = EyePoint()

        if not results.multi_face_landmarks:
            return None, eye_points

        landmarks = results.multi_face_landmarks[0]

        # 提取眼角坐标
        eye_points.left_eye_corner_left = self._get_landmark_coords(landmarks, self.L_EYE_CORNER_L)
        eye_points.left_eye_corner_right = self._get_landmark_coords(landmarks, self.L_EYE_CORNER_R)
        eye_points.right_eye_corner_left = self._get_landmark_coords(landmarks, self.R_EYE_CORNER_L)
        eye_points.right_eye_corner_right = self._get_landmark_coords(landmarks, self.R_EYE_CORNER_R)

        # 计算眼睛中心坐标
        eye_points.left_eye_center = self._calculate_average_coords(landmarks, self.L_EYE_CENTER_APPROX_IDS)
        eye_points.right_eye_center = self._calculate_average_coords(landmarks, self.R_EYE_CENTER_APPROX_IDS)

        # 计算虹膜中心坐标
        eye_points.left_iris_center = self._calculate_average_coords(landmarks, self.L_IRIS_POINTS)
        eye_points.right_iris_center = self._calculate_average_coords(landmarks, self.R_IRIS_POINTS)

        # 计算虹膜相对眼睛中心的偏移
        eye_points.left_offset = self._calculate_relative_offset(
            eye_points.left_eye_center, eye_points.left_iris_center)
        eye_points.right_offset = self._calculate_relative_offset(
            eye_points.right_eye_center, eye_points.right_iris_center)

        return landmarks, eye_points

    def _get_landmark_coords(self, landmarks, index):
        """获取单个关键点的坐标"""
        if 0 <= index < len(landmarks.landmark):
            lm = landmarks.landmark[index]
            if lm and 0 <= lm.x <= 1 and 0 <= lm.y <= 1:
                return (int(lm.x * self.image_width), int(lm.y * self.image_height))
        return None

    def _calculate_average_coords(self, landmarks, indices):
        """计算多个关键点的平均坐标"""
        coords = [self._get_landmark_coords(landmarks, i) for i in indices]
        coords = [c for c in coords if c is not None]
        if coords:
            return tuple(np.mean(np.array(coords), axis=0).astype(int))
        return None

    def _calculate_relative_offset(self, eye_center, iris_center):
        """计算虹膜中心相对于眼睛中心的偏移"""
        if eye_center and iris_center:
            return (iris_center[0] - eye_center[0], iris_center[1] - eye_center[1])
        return None

    def close(self):
        """释放资源"""
        self.face_mesh.close()