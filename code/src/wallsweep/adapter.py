"""Velocity-style command adapter on top of DSLPIDControl, with a load-aware vertical loop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
from scipy.spatial.transform import Rotation

PHYSICS_HZ = 240
PID_HZ = 48
SWEEP_HZ = 24
RANGE_HZ = 12

MAX_SPEED_XY = 0.5
MAX_SPEED_Z = 0.3
MAX_YAW_RATE = math.radians(60.0)
MAX_ACCEL_XY = 0.5
TARGET_LEAD_MAX = 0.15

LOAD_I_LIMIT_Z = 3.0
HOSE_FF_G_PER_M = 3.0
G = 9.8


@dataclass
class Command:
    """Body-frame velocity (m/s), yaw rate (rad/s) and target height (m)."""

    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    z: float = 0.0


class Schedule(NamedTuple):
    run_sweep: bool
    new_range: bool


def schedule(step: int) -> Schedule:
    """Whether the sweep controller runs and a new range reading arrives on this PID step."""
    return Schedule(step % (PID_HZ // SWEEP_HZ) == 0, step % (PID_HZ // RANGE_HZ) == 0)


class LoadPID(DSLPIDControl):
    """DSLPIDControl with a wider vertical integral clamp and a hose weight feed-forward."""

    def __init__(self, drone_model: DroneModel = DroneModel.CF2X, g: float = G,
                 hose_ff_g_per_m: float = HOSE_FF_G_PER_M):
        self.hose_ff_g_per_m = hose_ff_g_per_m
        super().__init__(drone_model=drone_model, g=g)

    def hose_feedforward(self, z: float) -> float:
        """Extra vertical thrust (N) for the hose hanging below height z."""
        return 0.001 * self.hose_ff_g_per_m * max(float(z), 0.0) * G

    def _dslPIDPositionControl(self, control_timestep, cur_pos, cur_quat, cur_vel,
                               target_pos, target_rpy, target_vel):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = self.integral_pos_e + pos_e * control_timestep
        self.integral_pos_e[0:2] = np.clip(self.integral_pos_e[0:2], -2.0, 2.0)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -LOAD_I_LIMIT_Z, LOAD_I_LIMIT_Z)
        target_thrust = (
            np.multiply(self.P_COEFF_FOR, pos_e)
            + np.multiply(self.I_COEFF_FOR, self.integral_pos_e)
            + np.multiply(self.D_COEFF_FOR, vel_e)
            + np.array([0.0, 0.0, self.GRAVITY + self.hose_feedforward(cur_pos[2])])
        )
        scalar_thrust = max(0.0, np.dot(target_thrust, cur_rotation[:, 2]))
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0.0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = np.vstack([target_x_ax, target_y_ax, target_z_ax]).transpose()
        target_euler = Rotation.from_matrix(target_rotation).as_euler("XYZ", degrees=False)
        return thrust, target_euler, pos_e


class Adapter:
    """Turns the latest Command and the drone state into 4 motor RPMs at PID_HZ."""

    def __init__(self, ctrl: DSLPIDControl | None = None):
        self.ctrl = ctrl if ctrl is not None else LoadPID()
        self.cmd = Command()
        self.target_xy = np.zeros(2)
        self.target_z = 0.0
        self.target_yaw = 0.0
        self.v_world = np.zeros(2)
        self.commands = 0
        self.clip_counts = {"horizontal": 0, "yaw_rate": 0, "z": 0, "accel": 0}

    def reset(self, state: np.ndarray) -> None:
        """Hold the current pose and clear the controller and the clip counters."""
        self.ctrl.reset()
        self.target_xy = np.array(state[0:2], dtype=float)
        self.target_z = float(state[2])
        self.target_yaw = float(state[9])
        self.cmd = Command(z=self.target_z)
        self.v_world = np.zeros(2)
        self.commands = 0
        self.clip_counts = {"horizontal": 0, "yaw_rate": 0, "z": 0, "accel": 0}

    def set_command(self, cmd: Command) -> Command:
        """Store a clipped copy of cmd and return it."""
        self.commands += 1
        vx, vy = float(cmd.vx), float(cmd.vy)
        speed = math.hypot(vx, vy)
        if speed > MAX_SPEED_XY:
            vx, vy = vx * MAX_SPEED_XY / speed, vy * MAX_SPEED_XY / speed
            self.clip_counts["horizontal"] += 1
        yaw_rate = float(np.clip(cmd.yaw_rate, -MAX_YAW_RATE, MAX_YAW_RATE))
        if yaw_rate != cmd.yaw_rate:
            self.clip_counts["yaw_rate"] += 1
        self.cmd = Command(vx, vy, yaw_rate, float(cmd.z))
        return self.cmd

    def compute(self, state: np.ndarray, dt: float = 1.0 / PID_HZ) -> np.ndarray:
        """Motor RPMs for this PID step."""
        pos = np.asarray(state[0:3], dtype=float)
        yaw = float(state[9])
        c, s = math.cos(yaw), math.sin(yaw)
        v_cmd = np.array([c * self.cmd.vx - s * self.cmd.vy, s * self.cmd.vx + c * self.cmd.vy])
        dv = v_cmd - self.v_world
        dv_norm = float(np.linalg.norm(dv))
        max_dv = MAX_ACCEL_XY * dt
        if dv_norm > max_dv:
            dv = dv * max_dv / dv_norm
            self.clip_counts["accel"] += 1
        self.v_world = self.v_world + dv
        v_world = self.v_world

        self.target_xy = self.target_xy + v_world * dt
        lead = self.target_xy - pos[0:2]
        dist = float(np.linalg.norm(lead))
        if dist > TARGET_LEAD_MAX:
            self.target_xy = pos[0:2] + lead * TARGET_LEAD_MAX / dist

        dz = self.cmd.z - self.target_z
        max_dz = MAX_SPEED_Z * dt
        if abs(dz) > max_dz:
            dz = math.copysign(max_dz, dz)
            self.clip_counts["z"] += 1
        self.target_z += dz

        self.target_yaw = math.remainder(self.target_yaw + self.cmd.yaw_rate * dt, math.tau)

        rpm, _, _ = self.ctrl.computeControlFromState(
            control_timestep=dt,
            state=state,
            target_pos=np.array([self.target_xy[0], self.target_xy[1], self.target_z]),
            target_rpy=np.array([0.0, 0.0, self.target_yaw]),
            target_vel=np.array([v_world[0], v_world[1], dz / dt]),
            target_rpy_rates=np.array([0.0, 0.0, self.cmd.yaw_rate]),
        )
        return rpm
