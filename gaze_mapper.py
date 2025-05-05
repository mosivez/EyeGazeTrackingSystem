# -*- coding: utf-8 -*-
# @Time    : 2025/05/05
# @Author  : 技术导师 Copilot (@mosivez)
# @Module  : 注视点映射

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import joblib
import os


class GazeMapper:
    """注视点映射模型"""

    def __init__(self, config_manager):
        """
        初始化注视映射模型

        Args:
            config_manager: 配置管理器实例
        """
        self.config = config_manager
        self.scaler = StandardScaler()
        self.trained = False

        # 创建模型
        self._create_models()

    def _create_models(self):
        """创建预测模型"""
        model_type = self.config.get('model', 'type')

        if model_type == 'svr':
            svr_params = self.config.get('model', 'svr')
            self.model_x = SVR(
                kernel=svr_params['kernel'],
                C=svr_params['C'],
                epsilon=svr_params['epsilon'],
                gamma=svr_params['gamma']
            )
            self.model_y = SVR(
                kernel=svr_params['kernel'],
                C=svr_params['C'],
                epsilon=svr_params['epsilon'],
                gamma=svr_params['gamma']
            )

        elif model_type == 'random_forest':
            rf_params = self.config.get('model', 'random_forest')
            self.model_x = RandomForestRegressor(
                n_estimators=rf_params['n_estimators'],
                max_depth=rf_params['max_depth']
            )
            self.model_y = RandomForestRegressor(
                n_estimators=rf_params['n_estimators'],
                max_depth=rf_params['max_depth']
            )

        elif model_type == 'neural_network':
            nn_params = self.config.get('model', 'neural_network')
            self.model_x = MLPRegressor(
                hidden_layer_sizes=tuple(nn_params['hidden_layers']),
                max_iter=nn_params['max_iter'],
                activation=nn_params['activation']
            )
            self.model_y = MLPRegressor(
                hidden_layer_sizes=tuple(nn_params['hidden_layers']),
                max_iter=nn_params['max_iter'],
                activation=nn_params['activation']
            )

        else:
            print(f"不支持的模型类型 '{model_type}'，使用默认SVR模型")
            self.model_x = SVR(kernel='rbf', C=1.0, epsilon=0.2, gamma='scale')
            self.model_y = SVR(kernel='rbf', C=1.0, epsilon=0.2, gamma='scale')

    def train(self, X_train, y_train):
        """
        训练模型

        Args:
            X_train: 特征矩阵 (n_samples, n_features)
            y_train: 目标坐标 (n_samples, 2)

        Returns:
            success: 训练是否成功
        """
        if len(X_train) == 0 or len(y_train) == 0 or len(X_train) != len(y_train):
            print("错误: 无效的训练数据")
            self.trained = False
            return False

        print(f"训练{self.config.get('model', 'type')}模型...")
        try:
            y_train_x = y_train[:, 0]
            y_train_y = y_train[:, 1]

            # 特征缩放
            X_train_scaled = self.scaler.fit_transform(X_train)

            print("训练X坐标模型...")
            self.model_x.fit(X_train_scaled, y_train_x)
            print("训练Y坐标模型...")
            self.model_y.fit(X_train_scaled, y_train_y)

            self.trained = True
            print("注视映射模型训练完成")
            return True
        except Exception as e:
            print(f"模型训练错误: {e}")
            self.trained = False
            return False

    def predict(self, features):
        """
        预测注视点

        Args:
            features: 特征向量

        Returns:
            gaze_point: (x, y)坐标
        """
        if not self.trained:
            return None

        if features is None or features.ndim == 0:
            return None

        try:
            # 确保features是2D数组
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # 特征缩放
            features_scaled = self.scaler.transform(features)

            # 预测坐标
            pred_x = self.model_x.predict(features_scaled)[0]
            pred_y = self.model_y.predict(features_scaled)[0]

            return int(pred_x), int(pred_y)
        except Exception as e:
            print(f"预测错误: {e}")
            return None

    def save_model(self, directory="models"):
        """
        保存模型

        Args:
            directory: 保存目录

        Returns:
            success: 是否保存成功
        """
        if not self.trained:
            print("模型未训练，无法保存")
            return False

        try:
            # 确保目录存在
            os.makedirs(directory, exist_ok=True)

            # 保存模型
            joblib.dump(self.model_x, os.path.join(directory, "model_x.pkl"))
            joblib.dump(self.model_y, os.path.join(directory, "model_y.pkl"))
            joblib.dump(self.scaler, os.path.join(directory, "scaler.pkl"))

            # 保存模型类型信息
            with open(os.path.join(directory, "model_info.txt"), "w") as f:
                f.write(f"Model type: {self.config.get('model', 'type')}")

            print(f"模型已保存到 {directory}")
            return True
        except Exception as e:
            print(f"保存模型失败: {e}")
            return False

    def load_model(self, directory="models"):
        """
        加载模型

        Args:
            directory: 模型目录

        Returns:
            success: 是否加载成功
        """
        try:
            self.model_x = joblib.load(os.path.join(directory, "model_x.pkl"))
            self.model_y = joblib.load(os.path.join(directory, "model_y.pkl"))
            self.scaler = joblib.load(os.path.join(directory, "scaler.pkl"))
            self.trained = True
            print(f"从 {directory} 加载模型成功")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            self.trained = False
            return False

    def augment_calibration_data(self, calibration_data, augment_factor=2):
        """
        通过插值增强校准数据

        Args:
            calibration_data: 原始校准数据
            augment_factor: 每对点之间插值的点数

        Returns:
            augmented_data: 增强后的数据
        """
        augmented_data = []
        for i in range(len(calibration_data)):
            for j in range(i + 1, len(calibration_data)):
                for factor in np.linspace(0.3, 0.7, augment_factor):
                    # 插值特征和目标
                    feat_i = calibration_data[i]['features']
                    feat_j = calibration_data[j]['features']
                    target_i = calibration_data[i]['target']
                    target_j = calibration_data[j]['target']

                    # 线性插值
                    new_feat = feat_i * (1 - factor) + feat_j * factor
                    new_target = target_i * (1 - factor) + target_j * factor

                    augmented_data.append({
                        'features': new_feat,
                        'target': new_target
                    })

        return calibration_data + augmented_data