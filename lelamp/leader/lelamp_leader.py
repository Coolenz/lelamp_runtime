#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from lerobot.teleoperators import Teleoperator
from .config_lelamp_leader import LeLampLeaderConfig

from collections import deque
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

class LeLampLeader(Teleoperator):
    """
    LeLamp Leader Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = LeLampLeaderConfig
    name = "lelamp_leader"

    def __init__(self, config: LeLampLeaderConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "base_yaw": Motor(1, "sts3215", norm_mode_body),
                "base_pitch": Motor(2, "sts3215", norm_mode_body),
                "elbow_pitch": Motor(3, "sts3215", norm_mode_body),
                "wrist_roll": Motor(4, "sts3215", norm_mode_body),
                "wrist_pitch": Motor(5, "sts3215", norm_mode_body),
            },
            calibration=self.calibration,
        )
        # SG滤波历史缓存
        self._action_history = {motor: deque(maxlen=7) for motor in self.bus.motors}
        # S曲线当前输出值
        self._s_curve_output = {motor: None for motor in self.bus.motors}
        # S曲线速度缓存
        self._s_curve_speed = {motor: 0.0 for motor in self.bus.motors}

        # 你可以调整以下参数
        self._max_speed = 2.0   # 单步最大速度（单位与动作一致，如度/步进）
        self._max_accel = 1.0   # 单步最大加速度（单位与动作一致）
        self._dt = 0.02         # 每次get_action的时间间隔（秒），用于速度/加速度计算

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")

        # Step 1: SG三阶滤波
        window_length = 7
        polyorder = 3
        sg_filtered = {}

        for motor, val in action.items():
            hist = self._action_history[motor]
            hist.append(val)
            if len(hist) >= window_length:
                filt_val = savgol_filter(list(hist), window_length, polyorder)[-1]
            else:
                filt_val = val
            sg_filtered[motor] = filt_val

        # Step 2: S型速度曲线处理
        s_curve_action = {}
        for motor, target in sg_filtered.items():
            if self._s_curve_output[motor] is None:
                # 初始化
                self._s_curve_output[motor] = target
                self._s_curve_speed[motor] = 0.0

            current = self._s_curve_output[motor]
            speed = self._s_curve_speed[motor]
            diff = target - current

            # 计算理想速度，方向正确
            desired_speed = diff / self._dt if self._dt > 0 else diff

            # 限制加速度
            delta_speed = desired_speed - speed
            if delta_speed > self._max_accel:
                delta_speed = self._max_accel
            elif delta_speed < -self._max_accel:
                delta_speed = -self._max_accel

            speed += delta_speed

            # 限制速度
            if speed > self._max_speed:
                speed = self._max_speed
            elif speed < -self._max_speed:
                speed = -self._max_speed

            # 计算新输出（S型曲线近似：加速度-速度-位移三步限制）
            new_output = current + speed * self._dt

            # 如果接近目标，直接锁定，防止抖动
            if abs(new_output - target) < 1e-3:
                new_output = target
                speed = 0.0

            # 保存状态
            self._s_curve_output[motor] = new_output
            self._s_curve_speed[motor] = speed

            s_curve_action[f"{motor}.pos"] = new_output

        return s_curve_action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
