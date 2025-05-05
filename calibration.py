# -*- coding: utf-8 -*-
# @Time    : 2025/05/05
# @Author  : 技术导师 Copilot (@mosivez)
# @Module  : 校准流程

import numpy as np
import time
import cv2


class CalibrationManager:
    """校准流程管理器"""

    def __init__(self, config_manager):
        """
        初始化校准管理器

        Args:
            config_manager: 配置管理器实例
        """
        self.config = config_manager

        # 屏幕尺寸
        self.screen_width = self.config.get('camera', 'width')
        self.screen_height = self.config.get('camera', 'height')

        # 校准点配置
        point_positions = self.config.get('calibration', 'point_positions')
        self.calibration_points = [
            (int(self.screen_width * p[0]), int(self.screen_height * p[1]))
            for p in point_positions
        ]

        # 校准状态
        self.calibration_data = []
        self.current_calib_index = 0
        self.calib_confirmed = False
        self.last_calib_capture_time = 0
        self.point_delay = self.config.get('calibration', 'point_delay')

    def update_screen_size(self, width, height):
        """
        更新屏幕尺寸

        Args:
            width: 屏幕宽度
            height: 屏幕高度
        """
        self.screen_width = width
        self.screen_height = height

        # 重新计算校准点位置
        point_positions = self.config.get('calibration', 'point_positions')
        self.calibration_points = [
            (int(self.screen_width * p[0]), int(self.screen_height * p[1]))
            for p in point_positions
        ]

    def process_calibration(self, key, features):
        """
        处理校准步骤

        Args:
            key: 键盘输入
            features: 当前帧的特征向量

        Returns:
            is_complete: 校准是否完成
            message: 提示信息
        """
        message = ""

        # 检查是否完成所有校准点
        if self.current_calib_index >= len(self.calibration_points):
            return True, "校准完成"

        # 当前校准点
        current_point = self.calibration_points[self.current_calib_index]

        # 处理按键确认
        if key == ord(' ') and not self.calib_confirmed:
            if features is not None:
                message = f"正在采集第 {self.current_calib_index + 1} 个校准点数据..."
                self.calib_confirmed = True
                self.last_calib_capture_time = time.time()

                # 保存校准数据
                target = np.array(current_point)
                self.calibration_data.append({'features': features, 'target': target})
                message += f"\n数据已采集: 目标=({target[0]}, {target[1]})"
            else:
                message = "无法采集校准数据: 特征不可用"

        # 处理确认后的延迟
        if self.calib_confirmed and (time.time() - self.last_calib_capture_time > self.point_delay):
            self.current_calib_index += 1
            self.calib_confirmed = False

            if self.current_calib_index < len(self.calibration_points):
                message = f"请注视下一个校准点 ({self.current_calib_index + 1}/{len(self.calibration_points)}) 并按空格键"
            else:
                return True, "校准完成"

        # 默认消息
        if not message:
            message = f"校准点 {self.current_calib_index + 1}/{len(self.calibration_points)}: 请注视点并按空格键"
            if self.calib_confirmed:
                message = f"点 {self.current_calib_index + 1} 已采集! 请稍等..."

        return False, message

    def draw_calibration(self, frame):
        """
        绘制校准界面

        Args:
            frame: 显示画面

        Returns:
            frame: 更新后的画面
        """
        if self.current_calib_index < len(self.calibration_points):
            point = self.calibration_points[self.current_calib_index]

            # 确定颜色
            color = (0, 255, 0) if not self.calib_confirmed else (0, 0, 255)

            # 绘制校准点
            cv2.circle(frame, point, 15, color, -1)
            cv2.circle(frame, point, 20, (255, 255, 255), 1)

            # 绘制提示文字
            text = f"校准 ({self.current_calib_index + 1}/{len(self.calibration_points)}): 请注视点并按空格键"
            if self.calib_confirmed:
                text = f"点 {self.current_calib_index + 1} 已采集! 请稍等..."

            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def get_calibration_data(self):
        """
        获取校准数据

        Returns:
            X: 特征矩阵
            y: 目标矩阵
        """
        if not self.calibration_data:
            return None, None

        X = np.array([item['features'] for item in self.calibration_data])
        y = np.array([item['target'] for item in self.calibration_data])

        return X, y

    def reset(self):
        """重置校准状态"""
        self.calibration_data = []
        self.current_calib_index = 0
        self.calib_confirmed = False
        self.last_calib_capture_time = 0