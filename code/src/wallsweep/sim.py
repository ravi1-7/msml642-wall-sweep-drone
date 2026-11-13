"""Thin wrapper around gym-pybullet-drones: one Crazyflie, no window."""

import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

PYB_HZ = 240  # physics steps per second
CTRL_HZ = 48  # flight controller (DSLPIDControl) updates per second


def make_env() -> CtrlAviary:
    return CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 0.05]]),
        physics=Physics.PYB,
        pyb_freq=PYB_HZ,
        ctrl_freq=CTRL_HZ,
        gui=False,
        record=False,
        user_debug_gui=False,
    )


def hover(seconds: float, height: float = 1.0) -> np.ndarray:
    """Take off and hold `height`. Returns the (steps, 3) position trace."""
    env = make_env()
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    obs, _ = env.reset()
    target = np.array([0.0, 0.0, height])
    action = np.zeros((1, 4))
    trace = []
    for _ in range(int(seconds * CTRL_HZ)):
        obs, *_ = env.step(action)
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP, state=obs[0], target_pos=target
        )
        trace.append(obs[0][0:3].copy())
    env.close()
    return np.array(trace)
