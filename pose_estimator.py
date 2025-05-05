# -*- coding: utf-8 -*-
# @Time    : 2025/05/05
# @Author  : 技术导师 Copilot (@mosivez)
# @Module  : 头部姿态估计

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class HeadPose:
    """头部姿态数据结构"""
    rvec: np.ndarray = None  # 旋转向量
    tvec: np.ndarray = None  # 平移向量
    success: bool = False  # 估计是否成功
    pnp_points_2d: list = None  # 用于PnP计算的2D点


class PoseEstimator:
    """头部姿态估计模块"""

    def __init__(self, config_manager):
        """
        初始化姿态估计器

        Args:
            config_manager: 配置管理器实例
        """
        self.config = config_manager

        # 获取3D模型点
        self.face_3d_model = self.config.get('face_model', '3d_points')
        self.pnp_3d_points = self.face_3d_model[:-2]  # 除了眼睛中心的点
        self.eye_centers_3d_model = self.face_3d_model[-2:]  # 左右眼中心的3D位置

        # 初始化相机参数
        self.camera_matrix = np.eye(3, dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 姿态估计用的关键点索引
        self.HEAD_POSE_IDS = [1, 33, 263, 61, 291, 199]  # 鼻尖、左眼角、右眼角、嘴角等

        # 图像尺寸，用于初始化相机参数
        self.image_width = self.config.get('camera', 'width')
        self.image_height = self.config.get('camera', 'height')
        self.update_camera_params()

    def update_camera_params(self, width=None, height=None):
        """
        更新相机内参

        Args:
            width: 图像宽度
            height: 图像高度
        """
        if width is not None:
            self.image_width = width
        if height is not None:
            self.image_height = height

        # 根据图像尺寸设置相机矩阵
        focal_length = float(self.image_width)
        center = (self.image_width / 2, self.image_height / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

    def estimate_pose(self, landmarks, get_landmarks_func):
        """
        估计头部姿态

        Args:
            landmarks: MediaPipe面部关键点
            get_landmarks_func: 获取关键点坐标的函数

        Returns:
            head_pose: 头部姿态数据
        """
        head_pose = HeadPose()

        if not landmarks:
            return head_pose

        # 获取PnP所需的2D点
        pnp_points_2d = []
        valid_pnp = True

        for idx in self.HEAD_POSE_IDS:
            coord = get_landmarks_func(landmarks, idx)
            if coord is None:
                valid_pnp = False
                break
            pnp_points_2d.append(coord)

        head_pose.pnp_points_2d = pnp_points_2d

        if not valid_pnp:
            return head_pose

        # 解算PnP问题，得到头部姿态
        try:
            pnp_points_2d_np = np.array(pnp_points_2d, dtype=np.float64)
            success_pnp, rvec_raw, tvec_raw = cv2.solvePnP(
                self.pnp_3d_points, pnp_points_2d_np,
                self.camera_matrix, self.dist_coeffs
            )

            if success_pnp:
                head_pose.rvec = rvec_raw
                head_pose.tvec = tvec_raw
                head_pose.success = True
                return head_pose

        except Exception as e:
            print(f"PnP姿态估计失败: {e}")

        return head_pose

    def get_pose_matrix(self, rvec, tvec):
        """
        获取姿态矩阵

        Args:
            rvec: 旋转向量
            tvec: 平移向量

        Returns:
            rotation_matrix: 旋转矩阵
            translation_vector: 平移向量
        """
        if rvec is None or tvec is None:
            return None, None

        try:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            return rotation_matrix, tvec
        except Exception as e:
            print(f"计算姿态矩阵失败: {e}")
            return None, None

    def calculate_gaze_direction(self, rvec, tvec, eye_offset, eye_type='left'):
        """
        计算视线方向

        Args:
            rvec: 旋转向量
            tvec: 平移向量
            eye_offset: 眼睛偏移
            eye_type: 'left'或'right'，指定使用哪只眼睛

        Returns:
            gaze_origin: 视线起点(3D)
            gaze_direction: 视线方向向量
        """
        if rvec is None or tvec is None or eye_offset is None:
            return None, None

        try:
            # 获取旋转矩阵
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # 选择左眼或右眼的3D模型点
            eye_idx = 0 if eye_type == 'left' else 1
            eye_center_3d = rotation_matrix @ self.eye_centers_3d_model[eye_idx].reshape(3, 1) + tvec

            # 基础视线方向 (模型空间中的Z轴方向)
            forward_vector = np.array([[0.], [0.], [1000.]])
            base_gaze_direction = rotation_matrix @ forward_vector

            # 根据眼睛偏移修正视线方向
            dx, dy = 0, 0
            if eye_offset:
                dx, dy = eye_offset

            # 敏感度参数
            sensitivity_x = self.config.get('system', 'gaze_sensitivity_x')
            sensitivity_y = self.config.get('system', 'gaze_sensitivity_y')

            # 根据眼睛偏移计算旋转角度
            ay = -dx * sensitivity_x * (np.pi / 180.)  # 水平偏移影响y轴旋转
            ax = dy * sensitivity_y * (np.pi / 180.)  # 垂直偏移影响x轴旋转

            # 创建X轴和Y轴旋转矩阵
            cosx, sinx = np.cos(ax), np.sin(ax)
            rot_x = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])

            cosy, siny = np.cos(ay), np.sin(ay)
            rot_y = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])

            # 应用偏移旋转，得到实际视线方向
            actual_gaze_direction = rotation_matrix @ rot_y @ rot_x @ forward_vector

            return eye_center_3d, actual_gaze_direction

        except Exception as e:
            print(f"计算视线方向失败: {e}")
            return None, None

    def project_gaze_to_screen(self, gaze_origin, gaze_direction, distance=1000):
        """
        将视线投影到屏幕平面

        Args:
            gaze_origin: 视线起点
            gaze_direction: 视线方向
            distance: 投影距离

        Returns:
            screen_point: 屏幕上的投影点(2D)
        """
        if gaze_origin is None or gaze_direction is None:
            return None

        try:
            # 计算视线线段的终点
            gaze_end = gaze_origin + gaze_direction * (distance / np.linalg.norm(gaze_direction))

            # 投影到图像平面
            points_3d = np.array([gaze_origin.flatten(), gaze_end.flatten()])
            points_2d, _ = cv2.projectPoints(
                points_3d, np.zeros(3), np.zeros(3),
                self.camera_matrix, self.dist_coeffs
            )

            if points_2d is not None and len(points_2d) == 2:
                start_point = tuple(np.int32(points_2d[0].ravel()))
                end_point = tuple(np.int32(points_2d[1].ravel()))
                return start_point, end_point

        except Exception as e:
            print(f"投影视线失败: {e}")

        return None, None