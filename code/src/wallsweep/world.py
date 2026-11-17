"""The simulated world: a random building facade, wind gusts, spray push, sensors, contact check.

Facade frame (local), used everywhere else in the code:
    u: distance into the wall. The wall's front face is u = 0; the drone flies at u < 0.
    v: along the wall, left (-) to right (+), centred on the wall.
    z: up, ground at z = 0.
The facade is placed in the world by a small yaw rotation about the world origin, so a drone
that starts with yaw 0 is not perfectly square to the wall.

The style (windows set into a masonry wall, with sills, hoods and floor ledges) matches the
mid-rise commercial buildings that hose-fed cleaning drones such as the Lucid Bots Sherpa work on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math

import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from wallsweep.hose import Hose

PYB_HZ = 240
CTRL_HZ = 48

WALL_THICKNESS = 0.10
STANDOFF = 0.30
RANGE_MAX = 4.0  # Multi-ranger maximum range (m)
RAY_START = 0.065  # rays start just outside the drone's 6 cm collision disc
SPRAY_PUSH_N = 0.003  # spray jet pushes the drone away from the wall while spraying
GROUND_ARMED_Z = 0.15
TUMBLE_RAD = math.radians(60.0)
OOB_MARGIN = 1.0
OOB_WALL_DIST = 1.5

# Glass and time-of-flight sensors: most infrared light passes through glass, and what reflects
# bounces off like a mirror, so readings are reliable only nearly head-on (Hopkins et al. 2025,
# arXiv 2510.06518; Cartographer_glass, arXiv 2212.08633). A reading that misses the glass
# "sees through" it and reports nothing in range. These numbers are placeholders until we
# measure the Multi-ranger against acrylic.
GLASS_NOISE_FACTOR = 2.0  # glass readings are noisier than readings off the wall
GLASS_BLIND_DEG = 35.0  # past this angle from head-on, glass is always missed

COLORS = {
    "wall": [0.85, 0.80, 0.70, 1.0],
    "glass": [0.35, 0.55, 0.80, 0.9],
    "sill": [0.45, 0.45, 0.48, 1.0],
    "hood": [0.45, 0.45, 0.48, 1.0],
    "ledge": [0.30, 0.30, 0.33, 1.0],
}

# Body-frame ray directions: Crazyflie x is forward (towards the wall at the start).
RAY_DIRS = {
    "front": np.array([1.0, 0.0, 0.0]),
    "left": np.array([0.0, 1.0, 0.0]),
    "right": np.array([0.0, -1.0, 0.0]),
    "up": np.array([0.0, 0.0, 1.0]),
}


# --------------------------------------------------------------------------- facade


@dataclass(frozen=True)
class Box:
    kind: str  # wall | glass | sill | hood | ledge
    center: tuple[float, float, float]  # facade frame (u, v, z)
    half: tuple[float, float, float]

    @staticmethod
    def from_bounds(kind, u, v, z) -> Box:
        c = ((u[0] + u[1]) / 2, (v[0] + v[1]) / 2, (z[0] + z[1]) / 2)
        h = ((u[1] - u[0]) / 2, (v[1] - v[0]) / 2, (z[1] - z[0]) / 2)
        return Box(kind, c, h)


@dataclass
class Facade:
    seed: int
    width: float
    height: float
    yaw: float  # radians
    floors: int
    boxes: list[Box] = field(default_factory=list)

    @property
    def rotation(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def to_world(self, local) -> np.ndarray:
        return self.rotation @ np.asarray(local, dtype=float)

    def to_local(self, world) -> np.ndarray:
        return self.rotation.T @ np.asarray(world, dtype=float)

    @property
    def outward_normal(self) -> np.ndarray:
        """World-frame unit vector pointing out of the wall, towards the drone."""
        return self.to_world([-1.0, 0.0, 0.0])

    def start_position(self) -> np.ndarray:
        """Bottom-left start, 30 cm in front of the wall, just above the ground."""
        return self.to_world([-STANDOFF, -self.width / 2 + 0.30, 0.05])

    def count(self, kind: str) -> int:
        return sum(b.kind == kind for b in self.boxes)


def sample_facade(seed: int, harder: bool = False, flat: bool = False) -> Facade:
    """Random facade from a seed. `flat` gives a plain square wall; `harder` deeper sills,
    hoods and ledges, present more often."""
    rng = np.random.default_rng(seed)
    W = 2.0 * rng.uniform(0.8, 1.2)
    H = 2.0 * rng.uniform(0.8, 1.2)
    T = WALL_THICKNESS

    if flat:
        f = Facade(seed, W, H, 0.0, 1)
        f.boxes.append(Box.from_bounds("wall", (0, T), (-W / 2, W / 2), (0, H)))
        return f

    yaw = np.deg2rad(rng.uniform(-10, 10))
    nf = int(rng.integers(2, 4))  # 2 or 3 floors
    sill_depth = (0.08, 0.12) if harder else (0.03, 0.08)
    hood_depth = (0.10, 0.15) if harder else (0.05, 0.12)
    ledge_depth = (0.20, 0.22) if harder else (0.15, 0.20)
    p_sill, p_hood, p_ledge = (0.9, 0.6, 1.0) if harder else (0.6, 0.3, 0.6)

    f = Facade(seed, W, H, yaw, nf)
    fh = H / nf
    for k in range(nf):
        z0, z1 = k * fh, (k + 1) * fh
        n = int(rng.integers(2, 5))  # 2 to 4 windows
        pitch = W / n
        ww = min(rng.uniform(0.25, 0.40), pitch - 0.10)
        zb = z0 + rng.uniform(0.25, 0.35) * fh
        zt = zb + rng.uniform(0.35, 0.50) * fh
        recess = rng.uniform(0.03, 0.05)

        f.boxes.append(Box.from_bounds("wall", (0, T), (-W / 2, W / 2), (z0, zb)))
        f.boxes.append(Box.from_bounds("wall", (0, T), (-W / 2, W / 2), (zt, z1)))
        edges = [-W / 2]
        for i in range(n):
            c = -W / 2 + (i + 0.5) * pitch
            left, right = c - ww / 2, c + ww / 2
            edges += [left, right]
            f.boxes.append(Box.from_bounds("glass", (recess, T), (left, right), (zb, zt)))
            if rng.random() < p_sill:
                d = rng.uniform(*sill_depth)
                f.boxes.append(
                    Box.from_bounds("sill", (-d, 0), (left - 0.02, right + 0.02), (zb - 0.03, zb))
                )
            if rng.random() < p_hood:  # small overhang above the window
                d = rng.uniform(*hood_depth)
                f.boxes.append(
                    Box.from_bounds("hood", (-d, 0), (left - 0.03, right + 0.03), (zt, zt + 0.03))
                )
        edges.append(W / 2)
        for a, b in zip(edges[0::2], edges[1::2]):  # pillars between windows
            f.boxes.append(Box.from_bounds("wall", (0, T), (a, b), (zb, zt)))

    for k in range(1, nf):  # ledges on floor lines
        if rng.random() < p_ledge:
            d = rng.uniform(*ledge_depth)
            zl = k * fh
            f.boxes.append(Box.from_bounds("ledge", (-d, 0), (-W / 2, W / 2), (zl - 0.02, zl + 0.02)))
    return f


def build_facade(facade: Facade, client: int) -> list[int]:
    """Create the facade's boxes in PyBullet. Returns their body ids."""
    quat = p.getQuaternionFromEuler([0, 0, facade.yaw])
    ids = []
    for b in facade.boxes:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=b.half, physicsClientId=client)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=b.half, rgbaColor=COLORS[b.kind], physicsClientId=client
        )
        ids.append(
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=facade.to_world(b.center).tolist(),
                baseOrientation=quat,
                physicsClientId=client,
            )
        )
    return ids


# --------------------------------------------------------------------------- wind


GUST_LEVELS = {"none": (0.0, 0.0), "moderate": (0.01, 0.02), "strong": (0.03, 0.05)}  # newtons
GUST_RATE = 0.3  # gusts per second on average
GUST_DURATION = (0.5, 2.0)  # seconds


class Gusts:
    """Random wind gusts: smooth force bumps with random timing, direction and strength.

    For scale, the Crazyflie weighs about 0.26 N, so a strong gust is up to ~20% of its weight.
    """

    def __init__(self, level: str = "none", seed: int = 0, t_max: float = 300.0):
        lo, hi = GUST_LEVELS[level]
        self.level = level
        rng = np.random.default_rng(seed)
        n = 0 if hi == 0 else rng.poisson(GUST_RATE * t_max)
        self.t0 = np.sort(rng.uniform(0, t_max, n))
        self.dur = rng.uniform(*GUST_DURATION, n)
        heading = rng.uniform(0, 2 * np.pi, n)
        vertical = rng.uniform(-0.2, 0.2, n)
        d = np.stack([np.cos(heading), np.sin(heading), vertical], axis=1)
        self.vec = d / np.linalg.norm(d, axis=1, keepdims=True) * rng.uniform(lo, hi, n)[:, None]

    def force(self, t: float) -> np.ndarray:
        if len(self.t0) == 0:
            return np.zeros(3)
        s = (t - self.t0) / self.dur
        active = (s >= 0) & (s < 1)
        if not active.any():
            return np.zeros(3)
        shape = 0.5 * (1 - np.cos(2 * np.pi * s[active]))  # smooth rise and fall
        return (shape[:, None] * self.vec[active]).sum(axis=0)


# --------------------------------------------------------------------------- environment


@dataclass
class Observation:
    """What the sweep controller sees: noisy Multi-ranger and Flow-deck style readings."""

    front: float
    left: float
    right: float
    up: float
    height: float
    vx: float  # body-frame velocity (forward)
    vy: float  # body-frame velocity (left)
    vz: float
    yaw: float


class FacadeAviary(CtrlAviary):
    """One Crazyflie in front of a facade, with gusts, spray push and an optional hose.

    Step it like CtrlAviary (motor RPMs from DSLPIDControl); read `sense()`, `true_ranges()`
    and `in_contact()` after each step. Set `spraying = True` to apply the spray push.
    """

    def __init__(
        self,
        facade: Facade,
        gusts: Gusts | None = None,
        hose: Hose | None = None,
        gui: bool = False,
        noise_seed: int = 0,
        glass_dropout: float = 0.0,
    ):
        """`glass_dropout`: chance a head-on reading off glass is lost (0 to 1). It rises
        towards 1 as the angle to the glass grows; see `glass_miss_probability`."""
        self.facade = facade
        self.glass_dropout = glass_dropout
        self.dropouts = 0  # glass readings lost so far
        self.failure: str | None = None
        self.failure_time: float | None = None
        self._airborne = False
        self._contact_this_step = False
        self.gusts = gusts or Gusts("none")
        self.hose = hose
        self.spraying = False
        self.sim_time = 0.0
        self.facade_ids: list[int] = []
        self._noise = np.random.default_rng(noise_seed)
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=facade.start_position()[None, :],
            initial_rpys=np.zeros((1, 3)),
            physics=Physics.PYB,
            pyb_freq=PYB_HZ,
            ctrl_freq=CTRL_HZ,
            gui=gui,
            record=False,
            obstacles=True,
            user_debug_gui=False,
            output_folder="results",
        )
        self.dry_mass = self.M
        self.mass = self.M  # current mass (kg): dry drone + lifted hose
        self.dry_inertia = p.getDynamicsInfo(self.DRONE_IDS[0], -1, physicsClientId=self.CLIENT)[2]

    # -- hooks into the simulator

    def _addObstacles(self):
        self.facade_ids = build_facade(self.facade, self.CLIENT)
        self._facade_set = set(self.facade_ids)
        self._kind = {i: b.kind for i, b in zip(self.facade_ids, self.facade.boxes)}

    def reset(self, seed=None, options=None):
        self.sim_time = 0.0
        self.spraying = False
        self.dropouts = 0
        self.failure = None
        self.failure_time = None
        self._airborne = False
        self._contact_this_step = False
        out = super().reset(seed=seed, options=options)
        if hasattr(self, "dry_mass"):
            self.mass = self.dry_mass
            self._set_dynamics()
        return out

    def _set_dynamics(self):
        p.changeDynamics(
            self.DRONE_IDS[0], -1, mass=self.mass, localInertiaDiagonal=self.dry_inertia,
            physicsClientId=self.CLIENT,
        )

    def step(self, action):
        self._contact_this_step = False
        if self.hose is not None:  # more hose is lifted as the drone climbs
            self.mass = self.dry_mass + self.hose.lifted_mass(self.pos[0][2])
            self._set_dynamics()
        return super().step(action)

    def _physics(self, rpm, nth_drone):
        super()._physics(rpm, nth_drone)
        # Forces are cleared after every physics sub-step, so extra forces go here.
        drone = self.DRONE_IDS[nth_drone]
        pos, _ = p.getBasePositionAndOrientation(drone, physicsClientId=self.CLIENT)
        f = self.gusts.force(self.sim_time)
        if self.spraying:
            f = f + SPRAY_PUSH_N * self.facade.outward_normal
        if self.hose is not None:
            f = f + self.hose.pull(np.array(pos), self.facade)
        if np.any(f):
            p.applyExternalForce(
                drone, -1, forceObj=f.tolist(), posObj=pos, flags=p.WORLD_FRAME,
                physicsClientId=self.CLIENT,
            )
        self._check_failure(drone, np.array(pos))
        self.sim_time += self.PYB_TIMESTEP

    def _check_failure(self, drone, pos):
        pts = p.getContactPoints(bodyA=int(drone), physicsClientId=self.CLIENT) or ()
        touching = any(c[2] in self._facade_set and (c[9] > 0 or c[8] < 0) for c in pts)
        if touching:
            self._contact_this_step = True
        if self.failure is not None:
            return
        self._airborne = self._airborne or pos[2] > GROUND_ARMED_Z
        failure = None
        if touching:
            failure = "facade_contact"
        elif self._airborne and any(c[2] not in self._facade_set for c in pts):
            failure = "ground"
        else:
            _, quat = p.getBasePositionAndOrientation(drone, physicsClientId=self.CLIENT)
            roll, pitch, _ = p.getEulerFromQuaternion(quat)
            local = self.facade.to_local(pos)
            if abs(roll) > TUMBLE_RAD or abs(pitch) > TUMBLE_RAD:
                failure = "tumble"
            elif (
                abs(local[1]) > self.facade.width / 2 + OOB_MARGIN
                or local[2] < -OOB_MARGIN
                or local[2] > self.facade.height + OOB_MARGIN
                or abs(local[0]) > OOB_WALL_DIST
            ):
                failure = "out_of_bounds"
        if failure is not None:
            self.failure = failure
            self.failure_time = self.sim_time

    # -- what the controller and the scorer can read

    def cast_rays(self) -> dict[str, tuple[float, str, float]]:
        """For each sensor: (exact distance capped at 4 m, kind of surface hit, angle in degrees
        between the ray and the surface's normal). Kind is "" when nothing is hit."""
        pos = self.pos[0]
        rot = np.array(p.getMatrixFromQuaternion(self.quat[0])).reshape(3, 3)
        dirs = [rot @ d for d in RAY_DIRS.values()]
        starts = [(pos + RAY_START * d).tolist() for d in dirs]
        ends = [(pos + RANGE_MAX * d).tolist() for d in dirs]
        hits = p.rayTestBatch(starts, ends, physicsClientId=self.CLIENT)
        out = {}
        for name, d, h in zip(RAY_DIRS, dirs, hits):
            hit_id, frac, normal = h[0], h[2], np.array(h[4])
            if hit_id < 0:
                out[name] = (RANGE_MAX, "", 0.0)
                continue
            dist = RAY_START + frac * (RANGE_MAX - RAY_START)
            angle = np.degrees(np.arccos(np.clip(abs(d @ normal), 0.0, 1.0)))
            out[name] = (dist, self._kind.get(hit_id, "ground"), float(angle))
        return out

    def true_ranges(self) -> dict[str, float]:
        """Exact distances from the drone centre along front/left/right/up, capped at 4 m."""
        return {k: v[0] for k, v in self.cast_rays().items()}

    def glass_miss_probability(self, angle_deg: float) -> float:
        """Chance a reading off glass is lost: `glass_dropout` head-on, rising to 1 at
        GLASS_BLIND_DEG."""
        x = min(angle_deg / GLASS_BLIND_DEG, 1.0)
        return self.glass_dropout + (1.0 - self.glass_dropout) * x**2

    def sense(self) -> Observation:
        """Noisy readings, as the real Multi-ranger and Flow deck would give them."""
        n = self._noise
        r = {}
        for k, (dist, kind, angle) in self.cast_rays().items():
            sd = 0.005 + 0.01 * dist
            if kind == "glass":
                if n.random() < self.glass_miss_probability(angle):
                    self.dropouts += 1
                    r[k] = RANGE_MAX  # beam went through the glass: nothing in range
                    continue
                sd *= GLASS_NOISE_FACTOR
            r[k] = float(np.clip(dist + n.normal(0, sd), 0.0, RANGE_MAX))
        yaw = float(self.rpy[0][2])
        c, s = np.cos(yaw), np.sin(yaw)
        vw = self.vel[0]
        vb = np.array([c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1], vw[2]])
        vb = vb + n.normal(0, 0.02, 3)
        return Observation(
            front=r["front"], left=r["left"], right=r["right"], up=r["up"],
            height=float(self.pos[0][2] + n.normal(0, 0.005)),
            vx=float(vb[0]), vy=float(vb[1]), vz=float(vb[2]), yaw=yaw,
        )

    def in_contact(self) -> bool:
        """True if the drone touching any part of the facade (wall, glass, sill, ledge)."""
        return self._contact_this_step
