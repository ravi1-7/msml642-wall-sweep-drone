"""Watch the simulated world: a random facade, the drone's sensor rays, gusts and the hose.

The drone follows a fixed, scripted side-to-side path 30 cm from the wall. This is NOT the
sweep controller (that is phase 2); it just shows the world the controller will face.
Green trail = inside the 25-35 cm band (spraying), red = outside it. The band is judged from
the Kalman filter's wall-distance estimate, which rides through glass dropouts.

Examples (from the code/ folder):
    uv run python scripts/demo.py                       # GUI window, wall seed 0
    uv run python scripts/demo.py --seed 7 --gust strong
    uv run python scripts/demo.py --harder --hose 3
    uv run python scripts/demo.py --glass-dropout 0.4    # glass reads badly
    uv run python scripts/demo.py --set dev_test --index 4   # a wall from a fixed set
    uv run python scripts/demo.py --flat
    uv run python scripts/demo.py --snapshot results/wall.png   # no window, save a picture
"""

import argparse
import time

import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.utils.utils import sync

from wallsweep.conditions import SETS, TEST_SET, VAL_SET, conditions
from wallsweep.estimator import StandoffKF
from wallsweep.hose import Hose
from wallsweep.world import CTRL_HZ, STANDOFF, FacadeAviary, Gusts, sample_facade

BAND = (0.25, 0.35)
SPEED = 0.25  # m/s along the scripted path
STEP_UP = 0.30  # m between passes
HELDOUT_SETS = (VAL_SET, TEST_SET, "harder")


def add_heldout_flag(ap):
    ap.add_argument("--allow-heldout", action="store_true",
                    help=f"allow --set {', '.join(HELDOUT_SETS)} (held out for validation and the "
                         "final test)")


def check_heldout(ap, a):
    if a.set in HELDOUT_SETS and not a.allow_heldout:
        ap.error(f"--set {a.set} is held out for validation or the final test; "
                 "pass --allow-heldout to fly it anyway")


def scripted_path(facade):
    """Waypoints (facade frame): take off, then sweep left-right, stepping up each pass."""
    left, right = -facade.width / 2 + 0.30, facade.width / 2 - 0.30
    z, pts, go_right = 0.30, [], True
    pts.append((-STANDOFF, left, 0.05))
    while z < facade.height - 0.15:
        a, b = (left, right) if go_right else (right, left)
        pts += [(-STANDOFF, a, z), (-STANDOFF, b, z)]
        z += STEP_UP
        go_right = not go_right
    return np.array(pts)


def target_at(pts, s):
    """Point at arc length s along the polyline."""
    seg = np.diff(pts, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(lens)])
    if s >= cum[-1]:
        return pts[-1], True
    i = np.searchsorted(cum, s, side="right") - 1
    return pts[i] + seg[i] * (s - cum[i]) / lens[i], False


def camera(facade):
    """Eye behind and to the left of the drone, looking at the middle of the wall."""
    target = facade.to_world([0, 0, facade.height / 2])
    eye = target + 3.2 * facade.outward_normal + facade.to_world([0, -1.4, 0]) + [0, 0, 0.5]
    d = target - eye
    yaw = np.degrees(np.arctan2(-d[0], d[1]))
    pitch = np.degrees(np.arctan2(d[2], np.linalg.norm(d[:2])))
    return target, eye, float(np.linalg.norm(d)), yaw, pitch


def main():
    try:
        run()
    except p.error as e:  # window closed mid-flight
        print(f"stopped: {e} (was the window closed?)")


def run():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="wall seed")
    ap.add_argument("--gust", choices=["none", "moderate", "strong"], default="moderate")
    ap.add_argument("--hose", type=float, default=3.0, help="hose grams per metre (0 = no hose)")
    ap.add_argument("--harder", action="store_true", help="deeper sills and ledges")
    ap.add_argument("--flat", action="store_true", help="plain flat wall")
    ap.add_argument("--glass-dropout", type=float, default=0.15,
                    help="chance a head-on reading off glass is lost (0 to 1)")
    ap.add_argument("--set", choices=list(SETS),
                    help="load a wall from a fixed set (train or a dev set) with its own gusts, "
                         "glass and hose; ignores the options above")
    ap.add_argument("--index", type=int, default=0, help="wall number within --set")
    ap.add_argument("--snapshot", help="save a picture of the scene here instead of a window")
    add_heldout_flag(ap)
    a = ap.parse_args()
    check_heldout(ap, a)

    gui = a.snapshot is None
    if a.set:
        cond = conditions(a.set, a.index)
        print(cond)
        facade, a.seed, a.gust = cond.facade, cond.seed, cond.gust_level
        env = cond.make_env(gui=gui)
    else:
        facade = sample_facade(a.seed, harder=a.harder, flat=a.flat)
        env = FacadeAviary(
            facade,
            gusts=Gusts(a.gust, seed=a.seed),
            hose=Hose(grams_per_m=a.hose) if a.hose > 0 else None,
            gui=gui,
            noise_seed=a.seed,
            glass_dropout=a.glass_dropout,
        )
    obs, _ = env.reset()
    print(
        f"wall seed {a.seed}: {facade.width:.2f} x {facade.height:.2f} m, "
        f"{facade.floors} floors, yaw {np.degrees(facade.yaw):+.1f} deg, "
        f"{facade.count('glass')} windows, {facade.count('sill')} sills, "
        f"{facade.count('hood')} hoods, "
        f"{facade.count('ledge')} ledges, gusts: {a.gust}"
    )

    target, eye, dist, cam_yaw, cam_pitch = camera(facade)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.CLIENT)
        p.resetDebugVisualizerCamera(dist, cam_yaw, cam_pitch, target.tolist(),
                                     physicsClientId=env.CLIENT)

    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    kf = StandoffKF(dt=env.CTRL_TIMESTEP)
    raw_err, kf_err = [], []
    pts = scripted_path(facade)
    action = np.zeros((1, 4))
    ray_ids = {k: -1 for k in ("front", "left", "right", "up")}
    text_id, last_pos = -1, env.pos[0].copy()
    contacts = sprayed = steps = 0
    min_front = np.inf
    start = time.time()

    for i in range(int(180 * CTRL_HZ)):
        local, done = target_at(pts, SPEED * i / CTRL_HZ)
        obs, *_ = env.step(action)
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP, state=obs[0],
            target_pos=facade.to_world(local), target_rpy=np.array([0, 0, facade.yaw]),
        )
        sense = env.sense()
        est = kf.update(sense.front, sense.vx)
        env.spraying = BAND[0] <= est <= BAND[1]
        if env.pos[0][2] > 0.2:
            true_front = env.true_ranges()["front"]
            raw_err.append(sense.front - true_front)
            kf_err.append(est - true_front)
        contacts += env.in_contact()
        sprayed += env.spraying
        steps += 1
        if env.pos[0][2] > 0.2:
            min_front = min(min_front, env.true_ranges()["front"])

        if gui and i % 2 == 0:
            pos = env.pos[0]
            rot = np.array(p.getMatrixFromQuaternion(env.quat[0])).reshape(3, 3)
            rng = env.true_ranges()
            for k, d in zip(ray_ids, ([1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1])):
                end = pos + rot @ np.array(d) * min(rng[k], 1.0)
                ray_ids[k] = p.addUserDebugLine(
                    pos, end, [1, 0.6, 0], 1.5, replaceItemUniqueId=ray_ids[k],
                    physicsClientId=env.CLIENT,
                )
            if i % 12 == 0:  # trail segment every 0.25 s; too many drawn items stall the GUI
                color = [0.1, 0.8, 0.2] if env.spraying else [0.9, 0.2, 0.2]
                p.addUserDebugLine(last_pos, pos, color, 3, physicsClientId=env.CLIENT)
                last_pos = pos.copy()
            label = (f"sensor {sense.front * 100:4.0f} cm   filter {est * 100:4.0f} cm   "
                     f"mass {env.mass * 1000:4.1f} g"
                     + ("   CONTACT" if env.in_contact() else ""))
            text_id = p.addUserDebugText(
                label, (facade.to_world([-0.6, -facade.width / 2, facade.height + 0.25])).tolist(),
                [0, 0, 0], 1.3, replaceItemUniqueId=text_id, physicsClientId=env.CLIENT,
            )
            sync(i, start, env.CTRL_TIMESTEP)
        if done and i > CTRL_HZ:
            break

    print(
        f"flight: {steps / CTRL_HZ:.0f} s, in band {100 * sprayed / steps:.0f}% of the time, "
        f"closest approach {min_front * 100:.0f} cm, contact steps {contacts}, "
        f"mass at the end {env.mass * 1000:.1f} g (dry {env.dry_mass * 1000:.0f} g)"
    )
    rms = lambda e: np.sqrt(np.mean(np.square(e))) * 100
    print(
        f"wall distance: {env.dropouts} glass dropouts; RMS error raw sensor {rms(raw_err):.1f} cm, "
        f"Kalman filter {rms(kf_err):.1f} cm"
    )

    if a.snapshot:
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(50, 4 / 3, 0.05, 20)
        w, h, rgb, *_ = p.getCameraImage(1200, 900, view, proj, renderer=p.ER_TINY_RENDERER,
                                         physicsClientId=env.CLIENT)
        import matplotlib.pyplot as plt

        plt.imsave(a.snapshot, np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8))
        print(f"saved {a.snapshot}")
    env.close()


if __name__ == "__main__":
    main()
