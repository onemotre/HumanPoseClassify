import numpy as np

class KF2D:
    def __init__(self, x0, P0, F, H, Q, R):
        """
        初始化二维 Kalman 滤波器类。

        参数：
        x0: 初始状态向量 (2x1) numpy.ndarray
        P0: 初始协方差矩阵 (2x2) numpy.ndarray
        F: 状态转移矩阵 (2x2) numpy.ndarray
        H: 观测矩阵 (2x2) numpy.ndarray
        Q: 过程噪声协方差矩阵 (2x2) numpy.ndarray
        R: 测量噪声协方差矩阵 (2x2) numpy.ndarray
        """
        self.x = x0  # 状态估计
        self.P = P0  # 估计协方差
        self.F = F   # 状态转移矩阵
        self.H = H   # 观测矩阵
        self.Q = Q   # 过程噪声协方差
        self.R = R   # 测量噪声协方差

    def predict(self):
        """
        预测步骤。
        """
        # 预测状态
        self.x = np.dot(self.F, self.x)
        # 预测协方差
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        更新步骤。

        参数：
        z: 测量值向量 (2x1) numpy.ndarray
        """
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态估计
        y = z - self.H @ self.x  # 创新（测量残差）
        self.x = self.x + K @ y
        
        # 更新协方差矩阵
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        """
        获取当前状态估计。

        返回值：
        x: 当前状态向量 numpy.ndarray
        """
        return self.x

# Kalman Filter
F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
Q = np.eye(4) * 0.1
R = np.eye(2) * 0.0001

pointKF = {}


def kalman_filter(idx, x0, y0):
    if idx not in pointKF.keys():
        print("init")
        pointKF[idx] = KF2D (
            x0 = np.array([x0, y0, 0, 0]),
            P0 = np.eye(4) * 0.01,
            F = F,
            H = H,
            Q = Q,
            R = R
        )
    pointKF[idx].predict()
    pointKF[idx].update(np.array([x0, y0]))
    return pointKF[idx].get_state()


def test_kalman_filter():
    data = [
        [0.0, 0],
        [1.0, 1],
        [2.0, 2],
        [3.0, 3],
        [4.0, 4],
        [5.0, 5]
    ]
    for turn in range(len(data)):
        kalman_filter(0, data[turn][0], data[turn][1])
        print(pointKF[0].get_state())

if __name__ == "__main__":
    test_kalman_filter()