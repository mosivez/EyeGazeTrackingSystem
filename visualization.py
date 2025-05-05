# -*- coding: utf-8 -*-
# @Time    : 2025/05/05
# @Author  : 技术导师 Copilot (@mosivez)
# @Module  : 可视化工具

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime


class Visualizer:
    """可视化工具类"""

    def __init__(self, config_manager):
        """
        初始化可视化工具

        Args:
            config_manager: 配置管理器实例
        """
        self.config = config_manager

        # 窗口名称
        self.feed_window_name = self.config.get('visualization', 'debug_window_name')
        self.main_window_name = self.config.get('visualization', 'main_window_name')

        # 可视化选项
        self.show_landmarks = self.config.get('visualization', 'show_landmarks')
        self.show_iris = self.config.get('visualization', 'show_iris')
        self.show_gaze = self.config.get('visualization', 'show_gaze')

        # 屏幕尺寸
        self.screen_width = self.config.get('camera', 'width')
        self.screen_height = self.config.get('camera', 'height')

    def setup_windows(self):
        """设置显示窗口"""
        cv2.namedWindow(self.feed_window_name)
        cv2.namedWindow(self.main_window_name, cv2.WINDOW_NORMAL)

    def draw_debug_feed(self, frame, landmarks, eye_points, head_pose, filtered_data):
        """
        在调试画面上绘制信息

        Args:
            frame: 原始视频帧
            landmarks: 面部关键点
            eye_points: 眼睛关键点数据
            head_pose: 头部姿态数据
            filtered_data: 滤波后的数据

        Returns:
            debug_frame: 带有调试信息的帧
        """
        debug_frame = frame.copy()

        # 1. 绘制眼睛关键点
        if self.show_iris and eye_points:
            # 眼角
            if eye_points.left_eye_corner_left:
                cv2.circle(debug_frame, eye_points.left_eye_corner_left, 2, (0, 128, 255), -1)
            if eye_points.left_eye_corner_right:
                cv2.circle(debug_frame, eye_points.left_eye_corner_right, 2, (0, 128, 255), -1)
            if eye_points.right_eye_corner_left:
                cv2.circle(debug_frame, eye_points.right_eye_corner_left, 2, (0, 128, 255), -1)
            if eye_points.right_eye_corner_right:
                cv2.circle(debug_frame, eye_points.right_eye_corner_right, 2, (0, 128, 255), -1)

            # 眼睛中心
            if eye_points.left_eye_center:
                cv2.circle(debug_frame, eye_points.left_eye_center, 3, (0, 255, 0), -1)
            if eye_points.right_eye_center:
                cv2.circle(debug_frame, eye_points.right_eye_center, 3, (0, 255, 0), -1)

            # 虹膜中心
            if eye_points.left_iris_center:
                cv2.circle(debug_frame, eye_points.left_iris_center, 2, (0, 0, 255), -1)
            if eye_points.right_iris_center:
                cv2.circle(debug_frame, eye_points.right_iris_center, 2, (0, 0, 255), -1)

            # 虹膜偏移线
            if eye_points.left_eye_center and eye_points.left_iris_center:
                cv2.line(debug_frame, eye_points.left_eye_center, eye_points.left_iris_center, (255, 0, 128), 1)
            if eye_points.right_eye_center and eye_points.right_iris_center:
                cv2.line(debug_frame, eye_points.right_eye_center, eye_points.right_iris_center, (255, 0, 128), 1)

        # 2. 绘制所有面部关键点
        if self.show_landmarks and landmarks:
            for i in range(468):  # MediaPipe提供468个面部关键点
                pos = self._get_landmark_pos(landmarks, i, self.screen_width, self.screen_height)
                if pos:
                    cv2.circle(debug_frame, pos, 1, (80, 110, 10), -1)

        # 3. 绘制头部姿态和视线
        camera_matrix = np.array([
            [float(self.screen_width), 0, self.screen_width / 2],
            [0, float(self.screen_width), self.screen_height / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        if filtered_data and filtered_data.rvec is not None and filtered_data.tvec is not None:
            # 绘制头部姿态轴
            self._draw_head_pose_axes(debug_frame, filtered_data.rvec, filtered_data.tvec, camera_matrix, dist_coeffs)

            # 绘制视线方向
            if self.show_gaze and filtered_data.offset is not None:
                self._draw_gaze_direction(debug_frame, filtered_data.rvec, filtered_data.tvec,
                                          filtered_data.offset, camera_matrix, dist_coeffs)

        # 4. 绘制状态信息
        cv2.putText(debug_frame, "Eye Tracking Debug View", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return debug_frame

    def create_main_screen(self, mode, calibration_manager=None, evaluation_manager=None, predicted_gaze=None):
        """
        创建主显示屏

        Args:
            mode: 当前模式 ('calibration', 'training', 'evaluation', 'prediction')
            calibration_manager: 校准管理器实例
            evaluation_manager: 评估管理器实例
            predicted_gaze: 预测的注视点

        Returns:
            main_screen: 主显示画面
        """
        main_screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        if mode == 'calibration' and calibration_manager:
            return calibration_manager.draw_calibration(main_screen)

        elif mode == 'training':
            cv2.putText(main_screen, "训练模型中...",
                        (self.screen_width // 2 - 150, self.screen_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        elif mode == 'evaluation' and evaluation_manager:
            return evaluation_manager.draw_evaluation(main_screen, predicted_gaze)

        elif mode == 'prediction':
            # 在预测模式下绘制目标示例
            self._draw_target_examples(main_screen)

            # 绘制预测点
            if predicted_gaze:
                cv2.circle(main_screen, predicted_gaze, 10, (0, 0, 255), -1)
                cv2.circle(main_screen, predicted_gaze, 12, (255, 255, 255), 1)

            cv2.putText(main_screen, "预测模式", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return main_screen

    def _get_landmark_pos(self, landmarks, index, width, height):
        """获取特定关键点的屏幕位置"""
        if landmarks and 0 <= index < len(landmarks.landmark):
            lm = landmarks.landmark[index]
            if lm and 0 <= lm.x <= 1 and 0 <= lm.y <= 1:
                return (int(lm.x * width), int(lm.y * height))
        return None

    def _draw_head_pose_axes(self, frame, rvec, tvec, camera_matrix, dist_coeffs, length=50.0):
        """绘制头部姿态坐标轴"""
        try:
            # 定义坐标轴点
            axis_3d = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, -length]]).reshape(-1, 3)

            # 投影到2D
            axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)

            if axis_2d is not None and len(axis_2d) == 4:
                # 提取点
                origin, x_axis, y_axis, z_axis = [tuple(np.int32(pt.ravel())) for pt in axis_2d]

                # 绘制坐标轴
                cv2.line(frame, origin, x_axis, (0, 0, 255), 2)  # X轴: 红色
                cv2.line(frame, origin, y_axis, (0, 255, 0), 2)  # Y轴: 绿色
                cv2.line(frame, origin, z_axis, (255, 0, 0), 2)  # Z轴: 蓝色
        except Exception as e:
            print(f"绘制姿态轴失败: {e}")

    def _draw_gaze_direction(self, frame, rvec, tvec, offset, camera_matrix, dist_coeffs):
        """绘制视线方向"""
        try:
            # 计算旋转矩阵
            rmat, _ = cv2.Rodrigues(rvec)

            # 左眼中心的3D位置
            eye_center_3d_model = np.array([-120., 170., -140.], dtype=np.float64)
            l_eye_center_3d = rmat @ eye_center_3d_model.reshape(3, 1) + tvec

            # 基础视线方向
            forward = np.array([[0.], [0.], [1000.]])
            base_gaze_dir = rmat @ forward

            # 从偏移计算修正角度
            dx, dy = offset.flatten()
            sensitivity_x = self.config.get('system', 'gaze_sensitivity_x')
            sensitivity_y = self.config.get('system', 'gaze_sensitivity_y')

            ay = -dx * sensitivity_x * (np.pi / 180.)
            ax = dy * sensitivity_y * (np.pi / 180.)

            # 创建旋转矩阵
            cosx, sinx = np.cos(ax), np.sin(ax)
            rot_x = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])

            cosy, siny = np.cos(ay), np.sin(ay)
            rot_y = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])

            # 应用偏移旋转得到实际视线方向
            actual_gaze_dir = rmat @ rot_y @ rot_x @ forward

            # 计算视线终点
            gaze_end_base = l_eye_center_3d + base_gaze_dir
            gaze_end_actual = l_eye_center_3d + actual_gaze_dir

            # 投影到2D
            pts_base, _ = cv2.projectPoints(
                np.array([l_eye_center_3d.flatten(), gaze_end_base.flatten()]),
                np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
            )

            pts_actual, _ = cv2.projectPoints(
                np.array([l_eye_center_3d.flatten(), gaze_end_actual.flatten()]),
                np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
            )

            # 绘制视线
            if pts_base is not None and len(pts_base) == 2:
                p1, p2 = [tuple(np.int32(pt.ravel())) for pt in pts_base]
                cv2.line(frame, p1, p2, (255, 100, 0), 2)  # 蓝色(基础)

            if pts_actual is not None and len(pts_actual) == 2:
                p1, p2 = [tuple(np.int32(pt.ravel())) for pt in pts_actual]
                cv2.line(frame, p1, p2, (0, 255, 255), 2)  # 黄色(实际)

        except Exception as e:
            pass  # 忽略视线渲染错误

    def _draw_target_examples(self, frame):
        """在预测模式下绘制示例目标"""
        # 在屏幕上绘制一些目标，用户可以尝试注视这些目标验证系统准确性
        targets = [
            (int(self.screen_width * 0.2), int(self.screen_height * 0.2)),
            (int(self.screen_width * 0.8), int(self.screen_height * 0.2)),
            (int(self.screen_width * 0.2), int(self.screen_height * 0.8)),
            (int(self.screen_width * 0.8), int(self.screen_height * 0.8)),
            (int(self.screen_width * 0.5), int(self.screen_height * 0.5))
        ]

        for target in targets:
            cv2.circle(frame, target, 10, (0, 255, 0), -1)
            cv2.circle(frame, target, 15, (255, 255, 255), 1)

    def visualize_accuracy_heatmap(self, heatmap, save_path=None):
        """
        可视化准确度热图

        Args:
            heatmap: 热图数据
            save_path: 保存路径
        """
        if heatmap is None:
            return

        plt.figure(figsize=(10, 8))

        # 创建自定义颜色映射: 绿色(低误差)到红色(高误差)
        colors = [(0, 0.8, 0), (0.8, 0.8, 0), (0.8, 0, 0)]  # 绿、黄、红
        cmap = LinearSegmentedColormap.from_list("accuracy_cmap", colors, N=100)

        # 绘制热图
        img = plt.imshow(heatmap, cmap=cmap)
        plt.colorbar(img, label='平均误差 (像素)')

        plt.title('注视点预测误差分布热图')
        plt.xlabel('水平位置')
        plt.ylabel('垂直位置')

        # 设置刻度标签
        x_ticks = np.linspace(0, heatmap.shape[1] - 1, 5)
        y_ticks = np.linspace(0, heatmap.shape[0] - 1, 5)
        plt.xticks(x_ticks, [f"{int(i * 100 / (heatmap.shape[1] - 1))}%" for i in range(5)])
        plt.yticks(y_ticks, [f"{int(i * 100 / (heatmap.shape[0] - 1))}%" for i in range(5)])

        # 显示值
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                if heatmap[i, j] > 0:
                    plt.text(j, i, f"{heatmap[i, j]:.1f}",
                             ha="center", va="center",
                             color="white" if heatmap[i, j] > 30 else "black")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"热图已保存到: {save_path}")
        else:
            plt.show()

    def visualize_evaluation_results(self, evaluation_data, save_dir=None):
        """
        可视化评估结果

        Args:
            evaluation_data: 评估数据
            save_dir: 保存目录
        """
        if not evaluation_data:
            print("无评估数据可视化")
            return

        # 创建保存目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 创建准确度散点图
        plt.figure(figsize=(12, 10))

        # 准备数据
        targets = []
        predictions = []
        mean_errors = []

        for point_data in evaluation_data:
            target = point_data['target']
            preds = point_data['predictions']

            if not preds:
                continue

            targets.append(target)

            # 计算平均预测点
            avg_pred = np.mean(np.array(preds), axis=0)
            predictions.append(avg_pred)

            # 计算平均误差
            errors = np.sqrt(np.sum((np.array(preds) - target) ** 2, axis=1))
            mean_error = np.mean(errors)
            mean_errors.append(mean_error)

        # 转换为numpy数组
        targets = np.array(targets)
        predictions = np.array(predictions)
        mean_errors = np.array(mean_errors)

        # 绘制目标点
        plt.scatter(targets[:, 0], targets[:, 1],
                    color='green', s=100, marker='o', label='目标点')

        # 使用误差作为颜色的散点图
        scatter = plt.scatter(predictions[:, 0], predictions[:, 1],
                              c=mean_errors, cmap='YlOrRd',
                              s=80, marker='x', label='预测点')

        # 绘制从目标点到预测点的线
        for i in range(len(targets)):
            plt.plot([targets[i, 0], predictions[i, 0]],
                     [targets[i, 1], predictions[i, 1]],
                     'k-', alpha=0.3)

        plt.colorbar(scatter, label='平均误差 (像素)')
        plt.title('注视点预测结果分析')
        plt.xlabel('X坐标 (像素)')
        plt.ylabel('Y坐标 (像素)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 设置轴范围
        plt.xlim(0, self.screen_width)
        plt.ylim(0, self.screen_height)

        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{timestamp}_accuracy_scatter.png"))
        else:
            plt.show()

        # 2. 创建误差箱型图
        plt.figure(figsize=(10, 6))

        # 收集每个点的误差数据
        all_errors = []
        labels = []

        for i, point_data in enumerate(evaluation_data):
            target = point_data['target']
            preds = point_data['predictions']

            if not preds:
                continue

            errors = np.sqrt(np.sum((np.array(preds) - target) ** 2, axis=1))
            all_errors.append(errors)
            labels.append(f"点{i + 1}\n({target[0]},{target[1]})")

        plt.boxplot(all_errors, labels=labels)
        plt.title('各评估点预测误差分布')
        plt.ylabel('误差 (像素)')
        plt.xlabel('评估点')
        plt.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45)

        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{timestamp}_error_boxplot.png"))
        else:
            plt.show()

    def close_windows(self):
        """关闭所有窗口"""
        cv2.destroyAllWindows()