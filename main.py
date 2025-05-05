# -*- coding: utf-8 -*-
# @Time    : 2025/05/05
# @Author  : 技术导师 Copilot (@mosivez)
# @System  : 基于MediaPipe的眼睛注视跟踪系统 - 模块化整合版

import cv2
import numpy as np
import time
import os
import json
import argparse

# 导入模块
from config import ConfigManager
from eye_detector import EyeDetector
from pose_estimator import PoseEstimator
from kalman_filter import KalmanFilterManager
from gaze_mapper import GazeMapper
from calibration import CalibrationManager
from evaluation import EvaluationManager
from visualization import Visualizer


class EyeTrackingSystem:
    """眼睛注视跟踪系统"""

    def __init__(self, config_file=None):
        """
        初始化眼动追踪系统

        Args:
            config_file: 配置文件路径
        """
        print("初始化眼动追踪系统...")

        # 加载配置
        self.config_manager = ConfigManager(config_file)

        # 初始化摄像头
        self.camera_id = self.config_manager.get('camera', 'id')
        self.cap = None

        # 初始化各个模块
        self.eye_detector = EyeDetector(self.config_manager)
        self.pose_estimator = PoseEstimator(self.config_manager)
        self.kalman_filter = KalmanFilterManager(self.config_manager)
        self.gaze_mapper = GazeMapper(self.config_manager)
        self.calibration_manager = CalibrationManager(self.config_manager)
        self.evaluation_manager = EvaluationManager(self.config_manager)
        self.visualizer = Visualizer(self.config_manager)

        # 系统状态
        self.mode = 'calibration'  # 'calibration', 'training', 'evaluation', 'prediction'
        self.running = True
        self.predicted_gaze_point = None

        print("系统初始化完成。")

    def _initialize_camera(self):
        """初始化摄像头并获取画面尺寸"""
        print(f"正在初始化摄像头 {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"错误: 无法打开摄像头 {self.camera_id}")
            self.running = False
            return False

        # 读取一帧以获取尺寸
        ret, frame = self.cap.read()
        if not ret:
            print("错误: 无法从摄像头读取帧")
            self.cap.release()
            self.running = False
            return False

        # 更新尺寸信息
        height, width, _ = frame.shape
        print(f"摄像头已初始化。分辨率: {width}x{height}")

        # 更新各模块的尺寸信息
        self.pose_estimator.update_camera_params(width, height)
        self.calibration_manager.update_screen_size(width, height)
        self.evaluation_manager.update_screen_size(width, height)

        return True

    def _process_frame(self, frame):
        """
        处理单帧

        Args:
            frame: 输入视频帧

        Returns:
            features: 提取的特征向量
            debug_data: 用于可视化的调试数据
        """
        # 1. 检测眼睛
        landmarks, eye_points = self.eye_detector.detect_eyes(frame)

        # 2. 估计头部姿态
        head_pose = self.pose_estimator.estimate_pose(
            landmarks,
            lambda lm, idx: self.eye_detector._get_landmark_coords(lm, idx)
        )

        # 3. 计算眼睛偏移
        eye_offset = None
        l_offset = eye_points.left_offset
        r_offset = eye_points.right_offset

        if l_offset and r_offset:
            eye_offset = np.array([[(l_offset[0] + r_offset[0]) / 2], [(l_offset[1] + r_offset[1]) / 2]],
                                  dtype=np.float32)
        elif l_offset:
            eye_offset = np.array([[l_offset[0]], [l_offset[1]]], dtype=np.float32)
        elif r_offset:
            eye_offset = np.array([[r_offset[0]], [r_offset[1]]], dtype=np.float32)

        # 4. 使用卡尔曼滤波平滑数据
        filtered_data = self.kalman_filter.filter_data(head_pose, eye_offset)

        # 返回可用于注视映射的特征向量和调试数据
        return filtered_data.feature_vector, {
            'landmarks': landmarks,
            'eye_points': eye_points,
            'head_pose': head_pose,
            'filtered_data': filtered_data
        }

    def _train_model(self):
        """训练注视映射模型"""
        print("\n--- 进入训练模式 ---")

        # 获取校准数据
        X, y = self.calibration_manager.get_calibration_data()

        if X is None or y is None:
            print("错误: 无有效校准数据用于训练")
            self.running = False
            return False

        # 可选: 数据增强
        if self.config_manager.get('model', {}).get('use_augmentation', False):
            print("应用数据增强...")
            augmented_data = self.gaze_mapper.augment_calibration_data(
                self.calibration_manager.calibration_data,
                augment_factor=2
            )
            X = np.array([item['features'] for item in augmented_data])
            y = np.array([item['target'] for item in augmented_data])
            print(f"数据量: 原始 {len(self.calibration_manager.calibration_data)} -> 增强后 {len(augmented_data)}")

        # 训练模型
        if self.gaze_mapper.train(X, y):
            self.mode = 'evaluation'
            print("\n--- 进入评估模式 ---")
            print("请注视第一个白点并按R键记录。")
            return True
        else:
            print("模型训练失败，即将退出。")
            self.running = False
            return False

    def _save_results(self):
        """保存模型和评估结果"""
        # 创建结果目录
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(results_dir, f"session_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)

        # 保存模型
        model_dir = os.path.join(session_dir, "model")
        self.gaze_mapper.save_model(model_dir)

        # 保存评估结果
        if hasattr(self.evaluation_manager, 'evaluation_data_raw') and self.evaluation_manager.evaluation_data_raw:
            # 计算并保存评估指标
            metrics = self.evaluation_manager.calculate_metrics()
            with open(os.path.join(session_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            # 生成并保存热图
            heatmap = self.evaluation_manager.generate_accuracy_heatmap()
            if heatmap is not None:
                self.visualizer.visualize_accuracy_heatmap(
                    heatmap,
                    os.path.join(session_dir, "accuracy_heatmap.png")
                )

            # 可视化评估结果
            self.visualizer.visualize_evaluation_results(
                self.evaluation_manager.evaluation_data_raw,
                os.path.join(session_dir, "visualizations")
            )

            print(f"结果已保存到: {session_dir}")
        else:
            print("无评估数据可保存")

    def run(self):
        """运行主循环"""
        if not self._initialize_camera():
            return

        # 设置窗口
        self.visualizer.setup_windows()

        # 主循环
        while self.running:
            # 1. 读取摄像头帧
            ret, frame = self.cap.read()
            if not ret:
                print("警告: 无法获取帧")
                continue

            # 是否翻转画面
            if self.config_manager.get('camera', 'flip'):
                frame = cv2.flip(frame, 1)

            # 2. 处理帧
            features, debug_data = self._process_frame(frame)

            # 3. 获取键盘输入
            key = cv2.waitKey(5) & 0xFF

            # 4. 根据当前模式执行相应逻辑
            if self.mode == 'calibration':
                # 校准模式
                calib_completed, calib_message = self.calibration_manager.process_calibration(key, features)
                if calib_completed:
                    self.mode = 'training'

            elif self.mode == 'training':
                # 训练模式
                self._train_model()

            elif self.mode == 'evaluation':
                # 评估模式

                # 在评估模式下也需要预测注视点
                if features is not None and self.gaze_mapper.trained:
                    predicted_gaze = self.gaze_mapper.predict(features)
                    if predicted_gaze:
                        predicted_gaze = (
                            np.clip(predicted_gaze[0], 0, self.calibration_manager.screen_width - 1),
                            np.clip(predicted_gaze[1], 0, self.calibration_manager.screen_height - 1)
                        )
                        # 应用速度限制
                        predicted_gaze = self.kalman_filter.limit_prediction_speed(predicted_gaze)
                else:
                    predicted_gaze = None

                self.predicted_gaze_point = predicted_gaze

                # 处理评估逻辑
                eval_completed, eval_message = self.evaluation_manager.process_evaluation(key, predicted_gaze)
                if eval_completed:
                    self.mode = 'prediction'
                    print("\n--- 评估完成，进入预测模式 ---")
                    self._save_results()

            elif self.mode == 'prediction':
                # 预测模式
                if features is not None and self.gaze_mapper.trained:
                    predicted_gaze = self.gaze_mapper.predict(features)
                    if predicted_gaze:
                        predicted_gaze = (
                            np.clip(predicted_gaze[0], 0, self.calibration_manager.screen_width - 1),
                            np.clip(predicted_gaze[1], 0, self.calibration_manager.screen_height - 1)
                        )
                        # 应用速度限制
                        predicted_gaze = self.kalman_filter.limit_prediction_speed(predicted_gaze)
                        self.predicted_gaze_point = predicted_gaze
                else:
                    self.predicted_gaze_point = None

            # 5. 更新显示
            debug_frame = self.visualizer.draw_debug_feed(
                frame,
                debug_data['landmarks'],
                debug_data['eye_points'],
                debug_data['head_pose'],
                debug_data['filtered_data']
            )

            main_screen = self.visualizer.create_main_screen(
                self.mode,
                self.calibration_manager,
                self.evaluation_manager,
                self.predicted_gaze_point
            )

            cv2.imshow(self.visualizer.feed_window_name, debug_frame)
            cv2.imshow(self.visualizer.main_window_name, main_screen)

            # 6. 处理退出
            if key == ord('q'):
                print("用户请求退出")
                self.running = False

            # 特殊命令
            if key == ord('s') and self.mode == 'prediction':
                self._save_results()

            if key == ord('r') and self.mode == 'prediction':
                print("重置校准...")
                self.calibration_manager.reset()
                self.mode = 'calibration'

        # 7. 清理资源
        self._cleanup()

    def _cleanup(self):
        """释放资源"""
        print("释放资源...")
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

        self.eye_detector.close()
        self.visualizer.close_windows()
        print("清理完成。")


# 配置文件模板
CONFIG_TEMPLATE = """# 眼睛注视跟踪系统配置文件

camera:
  id: 0           # 摄像头ID
  width: 640      # 宽度
  height: 480     # 高度
  flip: true      # 是否水平翻转

mediapipe:
  max_num_faces: 1
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5

calibration:
  points: 9       # 校准点数量
  point_delay: 0.5  # 采集点后等待时间(秒)
  point_positions:  # 校准点位置(相对屏幕比例)
    - [0.1, 0.1]
    - [0.5, 0.1]
    - [0.9, 0.1]
    - [0.1, 0.5]
    - [0.5, 0.5]
    - [0.9, 0.5]
    - [0.1, 0.9]
    - [0.5, 0.9]
    - [0.9, 0.9]

evaluation:
  frames_per_point: 30  # 每个点录制的帧数
  point_positions:      # 评估点位置(相对屏幕比例)
    - [0.2, 0.2]
    - [0.4, 0.2]
    - [0.6, 0.2]
    - [0.8, 0.2]
    - [0.2, 0.4]
    - [0.4, 0.4]
    - [0.6, 0.4]
    - [0.8, 0.4]
    - [0.2, 0.6]
    - [0.4, 0.6]
    - [0.6, 0.6]
    - [0.8, 0.6]
    - [0.2, 0.8]
    - [0.4, 0.8]
    - [0.6, 0.8]
    - [0.8, 0.8]

model:
  type: svr        # 'svr', 'random_forest', 'neural_network'
  use_augmentation: true
  svr:
    kernel: rbf
    C: 1.0
    epsilon: 0.2
    gamma: scale
  random_forest:
    n_estimators: 100
    max_depth: 10
  neural_network:
    hidden_layers: [50, 25]
    max_iter: 1000
    activation: relu

filtering:
  use_kalman: true
  pose_process_noise: 1.0e-4
  pose_measure_noise: 1.0e-2
  offset_process_noise: 1.0e-3
  offset_measure_noise: 5.0e-2
  prediction_max_speed: 100  # 防止大跳变

visualization:
  show_debug_feed: true
  show_landmarks: false
  show_iris: true
  show_gaze: true
  debug_window_name: 'Eye Gaze Detection Feed'
  main_window_name: 'Gaze Tracking Screen'

system:
  gaze_sensitivity_x: 0.03
  gaze_sensitivity_y: 0.03
  log_level: INFO
"""


def main():
    """程序入口点"""
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='基于MediaPipe的眼睛注视跟踪系统')
    parser.add_argument('-c', '--config', type=str, default='config.yml',
                        help='配置文件路径')
    parser.add_argument('--generate-config', action='store_true',
                        help='生成默认配置文件')
    args = parser.parse_args()

    # 生成默认配置文件
    if args.generate_config:
        if os.path.exists(args.config):
            print(f"警告: 配置文件 '{args.config}' 已存在，不会覆盖。")
        else:
            with open(args.config, 'w') as f:
                f.write(CONFIG_TEMPLATE)
            print(f"已生成默认配置文件: '{args.config}'")
        return

    # 检查配置文件是否存在
    config_file = args.config if os.path.exists(args.config) else None
    if config_file is None:
        print(f"警告: 配置文件 '{args.config}' 不存在，将使用默认配置。")
        print(f"提示: 使用 --generate-config 生成默认配置文件。")

    # 启动系统
    system = EyeTrackingSystem(config_file)
    system.run()


if __name__ == "__main__":
    main()