import torch
import numpy as np


# Non-linearity
def apply_nonlinear_response(sensor_value):
    threshold = 0.5  # 비선형 응답을 적용할 기준값
    if sensor_value < threshold:
        # 저압에서 비선형성 적용 (예: 로그)
        return np.log(sensor_value + 1e-6)  # 로그함수는 0에 가까운 값을 피하기 위해 작은 값 추가
    else:
        # 고압에서 포화 상태로 비선형성 적용
        return np.sqrt(sensor_value) + 0.1
