# -*- coding: utf-8 -*-
# @Time    : 2025/05/05
# @Author  : 技术导师 Copilot (@mosivez)
# @Module  : 卡尔曼滤波器

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class FilteredData:
    """滤波后的数据结构"""
    rvec: np.ndarray = None  # 滤波后的旋转向量
    tvec: np.ndarray = None  # 滤波后的平移向量
    offset: np.ndarray = None  # 滤波后的眼睛偏移
    feature_vector: np.ndarray = None  # 组合特征向量


class KalmanFilterManager:
    """卡尔曼滤波管理器"""

    def __init__(self, config_manager):
        """
        初始化卡尔曼滤波器

        Args:
            config_manager: 配置管理器实例
        """
        self.config = config_manager

        # 检查是否启用卡尔曼滤波
        self.use_kalman = self.config.get('filtering', 'use_kalman')

        if self.use_kalman:
            # 创建姿态和偏移的卡尔曼滤波器
            self.kf_pose = self._create_kalman_filter(
                6, 6,
                self.config.get('filtering', 'pose_process_noise'),
                self.config.get('filtering', 'pose_measure_noise')
            )

            self.kf_offset = self._create_kalman_filter(
                2, 2,
                self.config.get('filtering', 'offset_process_noise'),
                self.config.get('filtering', 'offset_measure_noise')
            )

            self.pose_kf_initialized = False
            self.offset_kf_initialized = False

        # 上一帧的预测点，用于速度限制
        self.last_prediction = None

    def _create_kalman_filter(self, state_dim, measure_dim, Q_val=1e-5, R_val=1e-4):
        """创建卡尔曼滤波器"""
        kf = cv2.KalmanFilter(state_dim, measure_dim)
        kf.transitionMatrix = np.eye(state_dim, dtype=np.float32)
        kf.measurementMatrix = np.eye(measure_dim, state_dim, dtype=np.float32)
        kf.processNoiseCov = np.eye(state_dim, dtype=np.float32) * Q_val
        kf.measurementNoiseCov = np.eye(measure_dim, dtype=np.float32) * R_val
        kf.errorCovPost = np.eye(state_dim, dtype=np.float32) * 1.0
        kf.statePost = np.zeros((state_dim, 1), dtype=np.float32)
        return kf

    def filter_data(self, head_pose, eye_offset):
        """
        过滤头部姿态和眼睛偏移数据

        Args:
            head_pose: 头部姿态数据
            eye_offset: 眼睛偏移数据(左右眼的平均偏移)

        Returns:
            filtered_data: 滤波后的数据
        """
        filtered_data = FilteredData()

        if not self.use_kalman:
            # 如果不使用卡尔曼滤波，直接返回原始数据
            if head_pose.success:
                filtered_data.rvec = head_pose.rvec
                filtered_data.tvec = head_pose.tvec

            filtered_data.offset = eye_offset

            # 组合特征向量
            if head_pose.success and eye_offset is not None:
                filtered_data.feature_vector = np.concatenate((
                    head_pose.rvec.flatten(),
                    head_pose.tvec.flatten(),
                    eye_offset.flatten()
                ))

            return filtered_data

        # --- 使用卡尔曼滤波 ---
        # 进行预测
        if self.pose_kf_initialized:
            self.kf_pose.predict()

        if self.offset_kf_initialized:
            self.kf_offset.predict()

        # 处理头部姿态
        if head_pose.success:
            pose_measured = np.vstack((
                head_pose.rvec.astype(np.float32),
                head_pose.tvec.astype(np.float32)
            ))

            if not self.pose_kf_initialized:
                self.kf_pose.statePost = pose_measured
                self.kf_pose.errorCovPost *= 0.1
                self.pose_kf_initialized = True
            else:
                self.kf_pose.correct(pose_measured)

            filtered_data.rvec = self.kf_pose.statePost[:3]
            filtered_data.tvec = self.kf_pose.statePost[3:]
        elif self.pose_kf_initialized:
            filtered_data.rvec = self.kf_pose.statePre[:3]
            filtered_data.tvec = self.kf_pose.statePre[3:]

        # 处理眼睛偏移
        if eye_offset is not None:
            offset_measured = eye_offset.reshape(-1, 1)

            if not self.offset_kf_initialized:
                self.kf_offset.statePost = offset_measured
                self.kf_offset.errorCovPost *= 0.1
                self.offset_kf_initialized = True
            else:
                self.kf_offset.correct(offset_measured)

            filtered_data.offset = self.kf_offset.statePost
        elif self.offset_kf_initialized:
            filtered_data.offset = self.kf_offset.statePre

        # 组合特征向量
        if filtered_data.rvec is not None and filtered_data.tvec is not None and filtered_data.offset is not None:
            filtered_data.feature_vector = np.concatenate((
                filtered_data.rvec.flatten(),
                filtered_data.tvec.flatten(),
                filtered_data.offset.flatten()
            ))

        return filtered_data

    def limit_prediction_speed(self, new_prediction, max_speed=None):
        """
        限制预测点的移动速度，避免大幅跳变

        Args:
            new_prediction: 新的预测点坐标
            max_speed: 最大移动速度(像素/帧)

        Returns:
            limited_prediction: 限速后的预测点
        """
        if new_prediction is None:
            self.last_prediction = None
            return None

        if max_speed is None:
            max_speed = self.config.get('filtering', 'prediction_max_speed')

        if self.last_prediction is None:
            self.last_prediction = new_prediction
            return new_prediction

        # 计算移动距离
        dx = new_prediction[0] - self.last_prediction[0]
        dy = new_prediction[1] - self.last_prediction[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # 如果超过最大速度，则限制
        if distance > max_speed:
            ratio = max_speed / distance
            limited_prediction = (
                int(self.last_prediction[0] + dx * ratio),
                int(self.last_prediction[1] + dy * ratio)
            )
        else:
            limited_prediction = new_prediction

        self.last_prediction = limited_prediction
        return limited_prediction