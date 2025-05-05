# -*- coding: utf-8 -*-
# @Time    : 2025/05/05
# @Author  : 技术导师 Copilot (@mosivez)
# @Module  : 配置管理模块

import os
import yaml
import numpy as np


class ConfigManager:
    """配置管理类，用于加载和访问系统配置"""

    DEFAULT_CONFIG = {
        'camera': {
            'id': 0,
            'width': 640,
            'height': 480,
            'flip': True
        },
        'mediapipe': {
            'max_num_faces': 1,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5
        },
        'calibration': {
            'points': 9,  # 3x3网格
            'point_delay': 0.5,  # 采集点后等待时间(秒)
            'point_positions': [(0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
                                (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
                                (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)]
        },
        'evaluation': {
            'points': 16,  # 4x4网格
            'frames_per_point': 30,
            'point_positions': [(0.2, 0.2), (0.4, 0.2), (0.6, 0.2), (0.8, 0.2),
                                (0.2, 0.4), (0.4, 0.4), (0.6, 0.4), (0.8, 0.4),
                                (0.2, 0.6), (0.4, 0.6), (0.6, 0.6), (0.8, 0.6),
                                (0.2, 0.8), (0.4, 0.8), (0.6, 0.8), (0.8, 0.8)]
        },
        'model': {
            'type': 'svr',  # 'svr', 'random_forest', 'neural_network'
            'svr': {
                'kernel': 'rbf',
                'C': 1.0,
                'epsilon': 0.2,
                'gamma': 'scale'
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10
            },
            'neural_network': {
                'hidden_layers': [50, 25],
                'max_iter': 1000,
                'activation': 'relu'
            }
        },
        'filtering': {
            'use_kalman': True,
            'pose_process_noise': 1e-4,
            'pose_measure_noise': 1e-2,
            'offset_process_noise': 1e-3,
            'offset_measure_noise': 5e-2,
            'prediction_max_speed': 100  # 防止大跳变
        },
        'visualization': {
            'show_debug_feed': True,
            'show_landmarks': False,
            'show_iris': True,
            'show_gaze': True,
            'debug_window_name': 'Eye Gaze Detection Feed',
            'main_window_name': 'Gaze Tracking Screen'
        },
        'face_model': {
            '3d_points': [
                [0., 0., 0.],  # 鼻尖
                [-225., 170., -135.],  # 左眼角
                [225., 170., -135.],  # 右眼角
                [-150., -150., -125.],  # 左嘴角
                [150., -150., -125.],  # 右嘴角
                [0., -330., -65.],  # 下巴
                [-120., 170., -140.],  # 左眼中心 (近似)
                [120., 170., -140.]  # 右眼中心 (近似)
            ]
        },
        'system': {
            'gaze_sensitivity_x': 0.03,
            'gaze_sensitivity_y': 0.03,
            'log_level': 'INFO'
        }
    }

    def __init__(self, config_file=None):
        self.config = self.DEFAULT_CONFIG.copy()

        if config_file and os.path.exists(config_file):
            self._load_file(config_file)

        # 转换特定配置项为NumPy数组
        self._convert_to_numpy()

    def _load_file(self, config_file):
        """从文件加载配置"""
        try:
            with open(config_file, 'r') as f:
                custom_config = yaml.safe_load(f)

            # 递归更新配置
            self._update_dict(self.config, custom_config)
            print(f"配置已从 {config_file} 加载")
        except Exception as e:
            print(f"加载配置文件出错: {e}")

    def _update_dict(self, d, u):
        """递归地更新字典"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d:
                self._update_dict(d[k], v)
            else:
                d[k] = v

    def _convert_to_numpy(self):
        """转换需要NumPy格式的配置项"""
        self.config['face_model']['3d_points'] = np.array(
            self.config['face_model']['3d_points'], dtype=np.float64)

    def get(self, section, key=None):
        """获取配置项"""
        if key:
            if section in self.config and key in self.config[section]:
                return self.config[section][key]
            return None
        else:
            return self.config.get(section, None)

    def save(self, file_path):
        """保存当前配置到文件"""
        # 转换NumPy数组为列表
        config_for_save = self.config.copy()
        config_for_save['face_model']['3d_points'] = config_for_save['face_model']['3d_points'].tolist()

        try:
            with open(file_path, 'w') as f:
                yaml.dump(config_for_save, f, default_flow_style=False)
            print(f"配置已保存到 {file_path}")
            return True
        except Exception as e:
            print(f"保存配置文件出错: {e}")
            return False