# -*- coding: utf-8 -*-
# @Time    : 2025/05/05
# @Author  : 技术导师 Copilot (@mosivez)
# @Module  : 评估模块

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


class EvaluationManager:
    """评估流程管理器"""

    def __init__(self, config_manager):
        """
        初始化评估管理器

        Args:
            config_manager: 配置管理器实例
        """
        self.config = config_manager

        # 屏幕尺寸
        self.screen_width = self.config.get('camera', 'width')
        self.screen_height = self.config.get('camera', 'height')

        # 评估点配置
        point_positions = self.config.get('evaluation', 'point_positions')
        self.evaluation_points = [
            (int(self.screen_width * p[0]), int(self.screen_height * p[1]))
            for p in point_positions
        ]

        # 评估状态
        self.evaluation_data_raw = []
        self.current_eval_index = 0
        self.is_recording_eval = False
        self.eval_frame_count = 0
        self.frames_per_point = self.config.get('evaluation', 'frames_per_point')
        self.current_eval_predictions = []

        # 评估报告相关
        self.metrics = None
        self.regional_analysis = None
        self.temporal_analysis = None

    def update_screen_size(self, width, height):
        """
        更新屏幕尺寸

        Args:
            width: 屏幕宽度
            height: 屏幕高度
        """
        self.screen_width = width
        self.screen_height = height

        # 重新计算评估点位置
        point_positions = self.config.get('evaluation', 'point_positions')
        self.evaluation_points = [
            (int(self.screen_width * p[0]), int(self.screen_height * p[1]))
            for p in point_positions
        ]

    def process_evaluation(self, key, predicted_gaze):
        """
        处理评估步骤

        Args:
            key: 键盘输入
            predicted_gaze: 预测的注视点

        Returns:
            is_complete: 评估是否完成
            message: 提示信息
        """
        message = ""

        # 检查是否完成所有评估点
        if self.current_eval_index >= len(self.evaluation_points):
            return True, "评估完成"

        # 当前评估点
        target_point = self.evaluation_points[self.current_eval_index]

        # 处理按键录制
        if key == ord('r') and not self.is_recording_eval:
            message = f"开始录制评估点 {self.current_eval_index + 1} 的数据..."
            self.is_recording_eval = True
            self.eval_frame_count = 0
            self.current_eval_predictions.clear()

        # 如果正在录制
        if self.is_recording_eval:
            if predicted_gaze is not None:
                # 存储带时间戳的预测，用于后续时间序列分析
                self.current_eval_predictions.append({
                    'position': predicted_gaze,
                    'timestamp': datetime.now().timestamp(),
                    'frame_index': self.eval_frame_count
                })

            self.eval_frame_count += 1

            if self.eval_frame_count >= self.frames_per_point:
                message = f"完成评估点 {self.current_eval_index + 1} 的录制"

                # 保存评估数据
                self.evaluation_data_raw.append({
                    'target': target_point,
                    'target_index': self.current_eval_index,
                    'predictions': self.current_eval_predictions,
                    'timestamp': datetime.now().timestamp()
                })

                self.is_recording_eval = False
                self.current_eval_index += 1

                if self.current_eval_index < len(self.evaluation_points):
                    message += f"\n请注视下一个评估点 ({self.current_eval_index + 1}) 并按R键录制"
                else:
                    return True, "评估完成"
            else:
                message = f"正在录制... {self.eval_frame_count}/{self.frames_per_point}"

        # 默认消息
        if not message:
            message = f"评估点 {self.current_eval_index + 1}/{len(self.evaluation_points)}: 请注视点并按R键开始录制"

        return False, message

    def draw_evaluation(self, frame, predicted_gaze=None):
        """
        绘制评估界面

        Args:
            frame: 显示画面
            predicted_gaze: 预测的注视点

        Returns:
            frame: 更新后的画面
        """
        if self.current_eval_index < len(self.evaluation_points):
            target = self.evaluation_points[self.current_eval_index]

            # 确定颜色
            color = (255, 255, 255)  # 白色目标
            if self.is_recording_eval:
                color = (0, 255, 255)  # 黄色表示正在录制

            # 绘制评估点
            cv2.circle(frame, target, 15, color, -1)
            cv2.circle(frame, target, 20, (0, 255, 0), 1)

            # 绘制提示文字
            text = f"评估 ({self.current_eval_index + 1}/{len(self.evaluation_points)}): 请注视点并按R键"
            if self.is_recording_eval:
                text = f"录制中... {self.eval_frame_count}/{self.frames_per_point}"

            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 绘制预测点
            if predicted_gaze:
                cv2.circle(frame, predicted_gaze, 10, (0, 0, 255), -1)  # 红色预测点

        return frame

    def calculate_metrics(self):
        """
        计算评估指标

        Returns:
            metrics: 评估指标字典
        """
        print("\n--- 评估指标计算 ---")

        metrics = {
            'per_point': [],
            'overall': {}
        }

        total_errors = []
        total_points = 0
        valid_points = 0

        for point_data in self.evaluation_data_raw:
            target = np.array(point_data['target'])
            predictions = [pred['position'] for pred in point_data['predictions']]

            point_metrics = {
                'target': target.tolist(),
                'target_index': point_data.get('target_index', 0),
                'mean_error': 0,
                'std_error': 0,
                'completeness': 0,
                'predictions_count': 0
            }

            if not predictions:
                print(f"目标点 {target}: 无有效预测")
                metrics['per_point'].append(point_metrics)
                total_points += 1
                continue

            pred_array = np.array(predictions)
            errors = np.sqrt(np.sum((pred_array - target) ** 2, axis=1))  # 欧氏距离

            mean_error = np.mean(errors)
            std_error = np.std(errors)
            completeness = len(predictions) / self.frames_per_point

            print(f"目标点 ({target[0]}, {target[1]}):")
            print(f"  - 平均误差: {mean_error:.2f} 像素")
            print(f"  - 标准差: {std_error:.2f} 像素")
            print(f"  - 完整度: {completeness:.2%}")
            print(f"  - 有效预测数: {len(predictions)}")

            point_metrics['mean_error'] = float(mean_error)
            point_metrics['std_error'] = float(std_error)
            point_metrics['completeness'] = float(completeness)
            point_metrics['predictions_count'] = len(predictions)

            metrics['per_point'].append(point_metrics)
            total_errors.extend(errors)
            total_points += 1
            valid_points += (len(predictions) > 0)

        if total_errors:
            overall_mean_error = np.mean(total_errors)
            overall_std_error = np.std(total_errors)
            overall_completeness = valid_points / total_points if total_points > 0 else 0

            print("\n整体评估:")
            print(f"  - 平均误差: {overall_mean_error:.2f} 像素")
            print(f"  - 标准差: {overall_std_error:.2f} 像素")
            print(f"  - 完整度: {overall_completeness:.2%}")
            print(f"  - 有效预测点数: {valid_points}/{total_points}")

            metrics['overall']['mean_error'] = float(overall_mean_error)
            metrics['overall']['std_error'] = float(overall_std_error)
            metrics['overall']['completeness'] = float(overall_completeness)
            metrics['overall']['valid_points'] = valid_points
            metrics['overall']['total_points'] = total_points
            metrics['overall']['total_predictions_count'] = len(total_errors)

            # 添加角度及区域相关指标
            metrics['regional'] = self._calculate_regional_metrics()

        else:
            print("没有足够数据计算整体评估指标")

        # 保存计算结果以便后续使用
        self.metrics = metrics
        return metrics

    def _calculate_regional_metrics(self):
        """计算不同屏幕区域的性能指标"""
        # 将屏幕分为9个区域
        regions = {
            'top_left': [], 'top_center': [], 'top_right': [],
            'middle_left': [], 'middle_center': [], 'middle_right': [],
            'bottom_left': [], 'bottom_center': [], 'bottom_right': []
        }

        # 区域边界
        w_thirds = [self.screen_width / 3, 2 * self.screen_width / 3]
        h_thirds = [self.screen_height / 3, 2 * self.screen_height / 3]

        # 分类每个评估点
        for point_data in self.evaluation_data_raw:
            target = point_data['target']
            predictions = [pred['position'] for pred in point_data['predictions']]

            if not predictions:
                continue

            # 确定区域
            region_name = ''
            if target[1] < h_thirds[0]:
                region_name = 'top_'
            elif target[1] < h_thirds[1]:
                region_name = 'middle_'
            else:
                region_name = 'bottom_'

            if target[0] < w_thirds[0]:
                region_name += 'left'
            elif target[0] < w_thirds[1]:
                region_name += 'center'
            else:
                region_name += 'right'

            # 计算误差
            pred_array = np.array(predictions)
            errors = np.sqrt(np.sum((pred_array - target) ** 2, axis=1))

            # 添加到区域列表
            regions[region_name].extend(errors)

        # 计算每个区域的指标
        regional_metrics = {}
        for region, errors in regions.items():
            if not errors:
                regional_metrics[region] = {
                    'mean_error': 0,
                    'std_error': 0,
                    'samples': 0
                }
                continue

            regional_metrics[region] = {
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'samples': len(errors)
            }

        return regional_metrics

    def generate_accuracy_heatmap(self):
        """
        生成准确度热力图

        Returns:
            heatmap: 热力图矩阵
        """
        if not self.evaluation_data_raw:
            print("没有评估数据，无法生成热力图")
            return None

        # 将屏幕划分为网格
        grid_size = 5  # 5x5网格
        heatmap = np.zeros((grid_size, grid_size))
        counts = np.zeros((grid_size, grid_size))

        for point_data in self.evaluation_data_raw:
            target = np.array(point_data['target'])
            predictions = [pred['position'] for pred in point_data['predictions']]

            if not predictions:
                continue

            # 计算目标点所在的网格位置
            grid_x = min(int(target[0] / self.screen_width * grid_size), grid_size - 1)
            grid_y = min(int(target[1] / self.screen_height * grid_size), grid_size - 1)

            # 计算平均误差
            pred_array = np.array(predictions)
            errors = np.sqrt(np.sum((pred_array - target) ** 2, axis=1))
            mean_error = np.mean(errors)

            heatmap[grid_y, grid_x] += mean_error
            counts[grid_y, grid_x] += 1

        # 计算平均值
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_heatmap = np.divide(heatmap, counts)
            avg_heatmap[np.isnan(avg_heatmap)] = 0

        return avg_heatmap

    def analyze_by_time(self):
        """
        分析时间序列变化

        Returns:
            temporal_analysis: 时间序列分析结果
        """
        if not self.evaluation_data_raw:
            return None

        # 分析每个目标点的时间序列误差变化
        temporal_data = {}

        for point_idx, point_data in enumerate(self.evaluation_data_raw):
            target = np.array(point_data['target'])
            predictions = point_data['predictions']

            if not predictions:
                continue

            # 提取时间序列数据
            time_series = []
            for pred in predictions:
                position = np.array(pred['position'])
                error = np.sqrt(np.sum((position - target) ** 2))
                frame_idx = pred['frame_index']

                time_series.append({
                    'frame': frame_idx,
                    'error': float(error),
                    'x_error': float(abs(position[0] - target[0])),
                    'y_error': float(abs(position[1] - target[1]))
                })

            if time_series:
                # 计算误差随时间的稳定性
                errors = [point['error'] for point in time_series]
                x_errors = [point['x_error'] for point in time_series]
                y_errors = [point['y_error'] for point in time_series]

                first_half_errors = errors[:len(errors) // 2]
                second_half_errors = errors[len(errors) // 2:]

                temporal_data[f"point_{point_idx}"] = {
                    'target': target.tolist(),
                    'time_series': time_series,
                    'stability': {
                        'error_trend': float(np.mean(second_half_errors) - np.mean(first_half_errors)),
                        'x_std': float(np.std(x_errors)),
                        'y_std': float(np.std(y_errors)),
                        'error_std': float(np.std(errors))
                    }
                }

        # 计算整体时间趋势
        all_first_half = []
        all_second_half = []

        for point_id, data in temporal_data.items():
            time_series = data['time_series']
            errors = [point['error'] for point in time_series]
            all_first_half.extend(errors[:len(errors) // 2])
            all_second_half.extend(errors[len(errors) // 2:])

        # 保存分析结果
        self.temporal_analysis = {
            'per_point': temporal_data,
            'overall': {
                'error_trend': float(
                    np.mean(all_second_half) - np.mean(all_first_half)) if all_first_half and all_second_half else 0,
                'first_half_mean': float(np.mean(all_first_half)) if all_first_half else 0,
                'second_half_mean': float(np.mean(all_second_half)) if all_second_half else 0
            }
        }

        # 输出总体时间趋势
        if all_first_half and all_second_half:
            trend = np.mean(all_second_half) - np.mean(all_first_half)
            print(f"\n时间序列分析:")
            print(f"  - 前半段平均误差: {np.mean(all_first_half):.2f} 像素")
            print(f"  - 后半段平均误差: {np.mean(all_second_half):.2f} 像素")
            print(f"  - 误差趋势: {trend:.2f} 像素 ({'增加' if trend > 0 else '减少' if trend < 0 else '无变化'})")

        return self.temporal_analysis

    def analyze_directional_error(self):
        """
        分析不同方向的误差

        Returns:
            directional_analysis: 方向误差分析结果
        """
        if not self.evaluation_data_raw:
            return None

        # 收集X方向和Y方向的误差
        x_errors = []
        y_errors = []

        for point_data in self.evaluation_data_raw:
            target = np.array(point_data['target'])
            predictions = [pred['position'] for pred in point_data['predictions']]

            if not predictions:
                continue

            pred_array = np.array(predictions)
            # X方向误差
            x_err = np.abs(pred_array[:, 0] - target[0])
            # Y方向误差
            y_err = np.abs(pred_array[:, 1] - target[1])

            x_errors.extend(x_err)
            y_errors.extend(y_err)

        # 计算方向误差统计
        directional_analysis = {
            'x_direction': {
                'mean_error': float(np.mean(x_errors)) if x_errors else 0,
                'std_error': float(np.std(x_errors)) if x_errors else 0,
                'samples': len(x_errors)
            },
            'y_direction': {
                'mean_error': float(np.mean(y_errors)) if y_errors else 0,
                'std_error': float(np.std(y_errors)) if y_errors else 0,
                'samples': len(y_errors)
            }
        }

        # 添加方向比较
        if x_errors and y_errors:
            x_mean = np.mean(x_errors)
            y_mean = np.mean(y_errors)
            ratio = x_mean / y_mean if y_mean > 0 else 0

            directional_analysis['comparison'] = {
                'x_to_y_ratio': float(ratio),
                'dominant_direction': 'horizontal' if x_mean > y_mean else 'vertical'
            }

            print("\n方向误差分析:")
            print(f"  - X方向平均误差: {x_mean:.2f} 像素")
            print(f"  - Y方向平均误差: {y_mean:.2f} 像素")
            print(f"  - 主导误差方向: {directional_analysis['comparison']['dominant_direction']}")
            print(f"  - 水平:垂直误差比: {ratio:.2f}")

        return directional_analysis

    def save_results(self, directory="evaluation_results"):
        """
        保存评估结果

        Args:
            directory: 保存目录

        Returns:
            success: 是否保存成功, 保存路径
        """
        try:
            # 确保目录存在
            os.makedirs(directory, exist_ok=True)

            # 当前时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = os.path.join(directory, f"eval_{timestamp}")
            os.makedirs(result_dir, exist_ok=True)

            # 1. 保存原始评估数据
            raw_data_simplified = []
            for point_data in self.evaluation_data_raw:
                # 简化预测数据以便JSON序列化
                simplified_predictions = [
                    {
                        'position': list(pred['position']),
                        'frame_index': pred['frame_index']
                    }
                    for pred in point_data['predictions']
                ]

                raw_data_simplified.append({
                    'target': point_data['target'],
                    'target_index': point_data.get('target_index', 0),
                    'predictions': simplified_predictions
                })

            with open(os.path.join(result_dir, 'raw_data.json'), 'w') as f:
                json.dump(raw_data_simplified, f, indent=2)

            # 2. 计算并保存评估指标
            if not self.metrics:
                self.metrics = self.calculate_metrics()

            with open(os.path.join(result_dir, 'metrics.json'), 'w') as f:
                json.dump(self.metrics, f, indent=2)

            # 3. 时间序列分析
            if not self.temporal_analysis:
                self.temporal_analysis = self.analyze_by_time()

            if self.temporal_analysis:
                with open(os.path.join(result_dir, 'temporal_analysis.json'), 'w') as f:
                    json.dump(self.temporal_analysis, f, indent=2)

            # 4. 方向误差分析
            directional_analysis = self.analyze_directional_error()
            if directional_analysis:
                with open(os.path.join(result_dir, 'directional_analysis.json'), 'w') as f:
                    json.dump(directional_analysis, f, indent=2)

            # 5. 生成可视化
            self._generate_visualizations(result_dir)

            # 6. 生成HTML报告
            self._generate_html_report(result_dir)

            print(f"评估结果已保存到: {result_dir}")
            return True, result_dir

        except Exception as e:
            print(f"保存评估结果时出错: {e}")
            return False, None

    def _generate_visualizations(self, result_dir):
        """生成评估可视化图"""
        viz_dir = os.path.join(result_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # 1. 生成热力图
        heatmap = self.generate_accuracy_heatmap()
        if heatmap is not None:
            plt.figure(figsize=(10, 8))

            # 创建自定义颜色映射: 绿色(低误差)到红色(高误差)
            colors = [(0, 0.8, 0), (0.8, 0.8, 0), (0.8, 0, 0)]  # 绿、黄、红
            cmap = LinearSegmentedColormap.from_list("accuracy_cmap", colors, N=100)

            img = plt.imshow(heatmap, cmap=cmap)
            plt.colorbar(img, label='平均误差 (像素)')

            plt.title('注视点预测误差分布热力图')
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
            plt.savefig(os.path.join(viz_dir, 'accuracy_heatmap.png'), dpi=300)
            plt.close()

        # 2. 准确度散点图
        if self.evaluation_data_raw:
            plt.figure(figsize=(12, 10))

            # 准备数据
            targets = []
            predictions = []
            mean_errors = []

            for point_data in self.evaluation_data_raw:
                target = point_data['target']
                preds = [pred['position'] for pred in point_data['predictions']]

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
            plt.gca().invert_yaxis()  # 图像坐标系Y轴向下

            plt.savefig(os.path.join(viz_dir, 'accuracy_scatter.png'), dpi=300)
            plt.close()

        # 3. 误差箱型图
        if self.evaluation_data_raw:
            plt.figure(figsize=(14, 8))

            # 收集每个点的误差数据
            all_errors = []
            labels = []

            for i, point_data in enumerate(self.evaluation_data_raw):
                target = point_data['target']
                preds = [pred['position'] for pred in point_data['predictions']]

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

            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'error_boxplot.png'), dpi=300)
            plt.close()

        # 4. 水平vs垂直误差条形图
        if self.evaluation_data_raw:
            # 收集X和Y方向误差
            x_errors_by_point = []
            y_errors_by_point = []
            point_labels = []

            for i, point_data in enumerate(self.evaluation_data_raw):
                target = np.array(point_data['target'])
                preds = [pred['position'] for pred in point_data['predictions']]

                if not preds:
                    continue

                pred_array = np.array(preds)
                x_errors_by_point.append(np.mean(np.abs(pred_array[:, 0] - target[0])))
                y_errors_by_point.append(np.mean(np.abs(pred_array[:, 1] - target[1])))
                point_labels.append(f"点{i + 1}")

            # 绘制条形图
            plt.figure(figsize=(12, 8))

            x = np.arange(len(point_labels))
            width = 0.35

            plt.bar(x - width / 2, x_errors_by_point, width, label='X方向误差')
            plt.bar(x + width / 2, y_errors_by_point, width, label='Y方向误差')

            plt.title('水平vs垂直方向预测误差对比')
            plt.xlabel('评估点')
            plt.ylabel('平均误差 (像素)')
            plt.xticks(x, point_labels)
            plt.grid(True, axis='y', alpha=0.3)
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'xy_error_comparison.png'), dpi=300)
            plt.close()

        # 5. 时间序列稳定性分析图
        if self.temporal_analysis:
            for point_id, data in self.temporal_analysis['per_point'].items():
                plt.figure(figsize=(10, 6))

                time_series = data['time_series']
                frames = [point['frame'] for point in time_series]
                errors = [point['error'] for point in time_series]
                x_errors = [point['x_error'] for point in time_series]
                y_errors = [point['y_error'] for point in time_series]

                plt.plot(frames, errors, 'b-', label='总误差')
                plt.plot(frames, x_errors, 'r--', label='X方向误差')
                plt.plot(frames, y_errors, 'g--', label='Y方向误差')

                # 添加均值线
                plt.axhline(y=np.mean(errors), color='b', linestyle='-', alpha=0.3)

                plt.title(f'评估点 {point_id} 的预测稳定性')
                plt.xlabel('帧索引')
                plt.ylabel('误差 (像素)')
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f'{point_id}_stability.png'), dpi=300)
                plt.close()

    def _generate_html_report(self, result_dir):
        """生成HTML评估报告"""
        html_path = os.path.join(result_dir, 'report.html')
        viz_dir = os.path.join(result_dir, 'visualizations')

        if not self.metrics:
            return

        # 创建HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>眼动追踪系统评估报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .metric-box {{ background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                .metric-row {{ display: flex; flex-wrap: wrap; margin: 0 -10px; }}
                .metric-col {{ flex: 1; padding: 0 10px; min-width: 200px; margin-bottom: 15px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                .vis-container {{ margin: 30px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                .good {{ color: #27ae60; }}
                .medium {{ color: #f39c12; }}
                .bad {{ color: #e74c3c; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>眼动追踪系统评估报告</h1>
                <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

                <div class="summary">
                    <h2>总体评估结果</h2>
                    <div class="metric-row">
                        <div class="metric-col">
                            <div class="metric-value">
                                {self.metrics['overall']['mean_error']:.2f} 像素
                            </div>
                            <div class="metric-label">平均误差</div>
                        </div>
                        <div class="metric-col">
                            <div class="metric-value">
                                {self.metrics['overall']['std_error']:.2f} 像素
                            </div>
                            <div class="metric-label">标准差</div>
                        </div>
                        <div class="metric-col">
                            <div class="metric-value">
                                {self.metrics['overall']['completeness'] * 100:.1f}%
                            </div>
                            <div class="metric-label">完整度</div>
                        </div>
                        <div class="metric-col">
                            <div class="metric-value">
                                {self.metrics['overall']['valid_points']}/{self.metrics['overall']['total_points']}
                            </div>
                            <div class="metric-label">有效评估点</div>
                        </div>
                    </div>
                </div>

                <div class="vis-container">
                    <h2>预测准确度分布</h2>
                    <img src="visualizations/accuracy_scatter.png" alt="准确度散点图">
                    <p>绿色点为目标注视点，彩色叉为预测点的平均位置。颜色表示误差大小，连线表示预测偏移方向。</p>
                </div>

                <div class="vis-container">
                    <h2>空间误差分布</h2>
                    <img src="visualizations/accuracy_heatmap.png" alt="准确度热力图">
                    <p>热力图显示不同屏幕区域的预测误差大小。绿色表示误差小，红色表示误差大。</p>
                </div>

                <div class="vis-container">
                    <h2>评估点误差分布</h2>
                    <img src="visualizations/error_boxplot.png" alt="误差箱型图">
                    <p>箱型图显示每个评估点的误差分布情况，包括中位数、四分位数和异常值。</p>
                </div>

                <div class="vis-container">
                    <h2>方向误差对比</h2>
                    <img src="visualizations/xy_error_comparison.png" alt="方向误差对比图">
                    <p>条形图对比每个评估点在水平(X)和垂直(Y)方向的误差大小。</p>
                </div>

                <h2>详细评估指标</h2>
                <table>
                    <tr>
                        <th>评估点</th>
                        <th>目标坐标</th>
                        <th>平均误差</th>
                        <th>标准差</th>
                        <th>完整度</th>
                        <th>预测样本数</th>
                    </tr>
        """

        # 添加每个评估点的指标
        for point in self.metrics['per_point']:
            # 确定误差等级
            error_class = ""
            if point['mean_error'] < 30:
                error_class = "good"
            elif point['mean_error'] < 60:
                error_class = "medium"
            else:
                error_class = "bad"

            html_content += f"""
                <tr>
                    <td>点{point['target_index'] + 1}</td>
                    <td>({point['target'][0]}, {point['target'][1]})</td>
                    <td class="{error_class}">{point['mean_error']:.2f} 像素</td>
                    <td>{point['std_error']:.2f} 像素</td>
                    <td>{point['completeness'] * 100:.1f}%</td>
                    <td>{point['predictions_count']}</td>
                </tr>
            """

        # 添加区域分析
        if 'regional' in self.metrics:
            html_content += """
                </table>

                <h2>区域分析</h2>
                <table>
                    <tr>
                        <th>区域</th>
                        <th>平均误差</th>
                        <th>标准差</th>
                        <th>样本数</th>
                    </tr>
            """

            for region, metrics in self.metrics['regional'].items():
                # 只显示有数据的区域
                if metrics['samples'] > 0:
                    # 确定误差等级
                    error_class = ""
                    if metrics['mean_error'] < 30:
                        error_class = "good"
                    elif metrics['mean_error'] < 60:
                        error_class = "medium"
                    else:
                        error_class = "bad"

                    html_content += f"""
                        <tr>
                            <td>{region.replace('_', ' ').title()}</td>
                            <td class="{error_class}">{metrics['mean_error']:.2f} 像素</td>
                            <td>{metrics['std_error']:.2f} 像素</td>
                            <td>{metrics['samples']}</td>
                        </tr>
                    """

        # 添加时间分析
        if self.temporal_analysis and self.temporal_analysis['overall']:
            html_content += """
                </table>

                <h2>时间序列分析</h2>
                <div class="metric-box">
                    <div class="metric-row">
            """

            trend = self.temporal_analysis['overall']['error_trend']
            trend_class = "good" if trend < -1 else "bad" if trend > 1 else "medium"
            trend_desc = "改善" if trend < 0 else "恶化" if trend > 0 else "保持稳定"

            html_content += f"""
                        <div class="metric-col">
                            <div class="metric-value">
                                {self.temporal_analysis['overall']['first_half_mean']:.2f} 像素
                            </div>
                            <div class="metric-label">前半段平均误差</div>
                        </div>
                        <div class="metric-col">
                            <div class="metric-value">
                                {self.temporal_analysis['overall']['second_half_mean']:.2f} 像素
                            </div>
                            <div class="metric-label">后半段平均误差</div>
                        </div>
                        <div class="metric-col">
                            <div class="metric-value {trend_class}">
                                {trend:.2f} 像素
                            </div>
                            <div class="metric-label">误差趋势 ({trend_desc})</div>
                        </div>
                    </div>
                </div>

                <p>时间序列分析显示系统在整个评估过程中的稳定性。负趋势表示系统性能改善，正趋势表示性能随时间恶化。</p>

                <h3>各评估点稳定性分析</h3>
                <div class="metric-row">
            """

            # 添加各点的稳定性图
            for point_id in self.temporal_analysis['per_point']:
                point_num = point_id.split('_')[1]
                html_content += f"""
                    <div class="metric-col" style="max-width: 50%;">
                        <img src="visualizations/{point_id}_stability.png" alt="点{point_num}稳定性分析">
                    </div>
                """

        # 结束HTML
        html_content += """
                </div>
            </div>
        </body>
        </html>
        """

        # 写入文件
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def export_evaluation_report(self, output_path=None):
        """
        导出完整评估报告

        Args:
            output_path: 保存路径

        Returns:
            success: 是否导出成功
        """
        if not self.evaluation_data_raw:
            print("无评估数据，无法生成报告")
            return False

        # 如果没有提供路径，自动生成
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"evaluation_report_{timestamp}.pdf"

        # 确保有指标数据
        if not self.metrics:
            self.metrics = self.calculate_metrics()

        # 保存结果，并获取结果目录
        success, result_dir = self.save_results()
        if not success:
            return False

        print(f"评估报告已保存: {result_dir}/report.html")
        return True