"""
P-MCP Physics Engine Integration
==================================
Pluggable physics backend for shadow simulation.
Supports PyBullet, MuJoCo, and a built-in analytical solver.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("pmcp.physics")


# ─────────────────────────────────────────────────────────────────────────────
#  DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Pose3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def distance_to(self, other: "Pose3D") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "Pose3D":
        return cls(
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            z=d.get("z", 0.0),
            roll=d.get("roll", 0.0),
            pitch=d.get("pitch", 0.0),
            yaw=d.get("yaw", 0.0),
        )


@dataclass
class Trajectory:
    waypoints: List[Pose3D] = field(default_factory=list)
    timestamps_ms: List[int] = field(default_factory=list)
    speed_m_s: float = 0.3
    robot_id: str = ""

    def total_distance(self) -> float:
        total = 0.0
        for i in range(1, len(self.waypoints)):
            total += self.waypoints[i - 1].distance_to(self.waypoints[i])
        return total

    def estimated_duration_ms(self) -> int:
        if self.speed_m_s <= 0:
            return 0
        dist = self.total_distance()
        return int((dist / self.speed_m_s) * 1000)


@dataclass
class CollisionResult:
    collision: bool
    colliding_objects: List[str] = field(default_factory=list)
    min_distance_m: float = float("inf")
    penetration_depth_m: float = 0.0
    contact_point: Optional[Pose3D] = None


@dataclass
class PhysicsSimResult:
    feasible: bool
    trajectory: Trajectory
    collision: CollisionResult
    final_pose: Pose3D
    energy_j: float
    duration_ms: int
    violations: List[str] = field(default_factory=list)
    sim_time_ms: int = 0


@dataclass
class WorkspaceObject:
    object_id: str
    shape: str  # "box" | "sphere" | "cylinder" | "mesh"
    pose: Pose3D
    dimensions: Dict[str, float]  # e.g. {"width": 0.1, "height": 0.2, "depth": 0.1}
    static: bool = True
    mass_kg: float = 1.0
    friction: float = 0.5


# ─────────────────────────────────────────────────────────────────────────────
#  PHYSICS BACKEND INTERFACE
# ─────────────────────────────────────────────────────────────────────────────


class PhysicsBackend(abc.ABC):
    """Abstract physics simulation backend."""

    @abc.abstractmethod
    async def simulate(
        self,
        trajectory: Trajectory,
        robot_geometry: Dict[str, Any],
        workspace_objects: List[WorkspaceObject],
        floor_z: float = 0.0,
    ) -> PhysicsSimResult: ...

    @abc.abstractmethod
    async def check_collision(
        self,
        pose: Pose3D,
        robot_geometry: Dict[str, Any],
        workspace_objects: List[WorkspaceObject],
    ) -> CollisionResult: ...

    @abc.abstractmethod
    def reset(self) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYTICAL SOLVER  (no external dependencies)
# ─────────────────────────────────────────────────────────────────────────────


class AnalyticalPhysicsBackend(PhysicsBackend):
    """
    Lightweight analytical backend for fast pre-flight checks.
    Uses bounding-sphere collision detection and linear motion model.
    """

    def __init__(
        self,
        gravity_m_s2: float = 9.81,
        default_robot_radius_m: float = 0.15,
        human_proximity_radius_m: float = 0.5,
    ):
        self.gravity = gravity_m_s2
        self.robot_radius = default_robot_radius_m
        self.human_proximity_radius = human_proximity_radius_m
        self._workspace_objects: List[WorkspaceObject] = []

    async def simulate(
        self,
        trajectory: Trajectory,
        robot_geometry: Dict[str, Any],
        workspace_objects: List[WorkspaceObject],
        floor_z: float = 0.0,
    ) -> PhysicsSimResult:
        t0 = time.perf_counter()
        violations: List[str] = []
        overall_collision = CollisionResult(collision=False)

        robot_r = robot_geometry.get("radius_m", self.robot_radius)

        for i, wp in enumerate(trajectory.waypoints):
            # Floor check
            if wp.z < floor_z:
                violations.append(f"waypoint[{i}]: z={wp.z:.3f} below floor {floor_z}")

            # Collision check at each waypoint
            col = await self.check_collision(wp, {"radius_m": robot_r}, workspace_objects)
            if col.collision:
                overall_collision = col
                for obj in col.colliding_objects:
                    violations.append(f"waypoint[{i}]: collision with {obj}")

        # Estimate energy: simplified model
        # E = integral of F * ds ≈ mass * gravity * delta_z + 0.5 * mass * v^2 * samples
        mass_kg = robot_geometry.get("mass_kg", 20.0)
        dist = trajectory.total_distance()
        delta_z = 0.0
        if len(trajectory.waypoints) >= 2:
            delta_z = max(0.0, trajectory.waypoints[-1].z - trajectory.waypoints[0].z)
        energy_j = (
            mass_kg * self.gravity * delta_z
            + 0.5 * mass_kg * trajectory.speed_m_s**2
            + dist * 5.0  # friction/motor coefficient
        )

        final_pose = trajectory.waypoints[-1] if trajectory.waypoints else Pose3D()
        duration_ms = trajectory.estimated_duration_ms()
        sim_time_ms = int((time.perf_counter() - t0) * 1000)

        return PhysicsSimResult(
            feasible=len(violations) == 0,
            trajectory=trajectory,
            collision=overall_collision,
            final_pose=final_pose,
            energy_j=energy_j,
            duration_ms=duration_ms,
            violations=violations,
            sim_time_ms=sim_time_ms,
        )

    async def check_collision(
        self,
        pose: Pose3D,
        robot_geometry: Dict[str, Any],
        workspace_objects: List[WorkspaceObject],
    ) -> CollisionResult:
        robot_r = robot_geometry.get("radius_m", self.robot_radius)
        colliding = []
        min_dist = float("inf")
        penetration = 0.0
        contact = None

        for obj in workspace_objects:
            obj_r = self._object_radius(obj)
            dx = pose.x - obj.pose.x
            dy = pose.y - obj.pose.y
            dz = pose.z - obj.pose.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            clearance = dist - (robot_r + obj_r)

            if clearance < min_dist:
                min_dist = clearance

            if clearance <= 0:
                colliding.append(obj.object_id)
                if -clearance > penetration:
                    penetration = -clearance
                    contact = Pose3D(
                        x=(pose.x + obj.pose.x) / 2,
                        y=(pose.y + obj.pose.y) / 2,
                        z=(pose.z + obj.pose.z) / 2,
                    )

        return CollisionResult(
            collision=bool(colliding),
            colliding_objects=colliding,
            min_distance_m=min_dist,
            penetration_depth_m=penetration,
            contact_point=contact,
        )

    def _object_radius(self, obj: WorkspaceObject) -> float:
        d = obj.dimensions
        if obj.shape == "sphere":
            return d.get("radius", 0.1)
        elif obj.shape == "cylinder":
            return max(d.get("radius", 0.1), d.get("height", 0.2) / 2)
        else:  # box / default
            return (
                math.sqrt(
                    d.get("width", 0.1) ** 2 + d.get("depth", 0.1) ** 2 + d.get("height", 0.1) ** 2
                )
                / 2
            )

    def reset(self) -> None:
        self._workspace_objects = []


# ─────────────────────────────────────────────────────────────────────────────
#  PYBULLET BACKEND
# ─────────────────────────────────────────────────────────────────────────────


class PyBulletBackend(PhysicsBackend):
    """
    PyBullet-backed physics simulation.
    Falls back to AnalyticalPhysicsBackend if pybullet is not installed.
    """

    def __init__(
        self,
        use_gui: bool = False,
        gravity_m_s2: float = -9.81,
        time_step_s: float = 1.0 / 240.0,
    ):
        self._use_gui = use_gui
        self._gravity = gravity_m_s2
        self._time_step = time_step_s
        self._client_id: Optional[int] = None
        self._pb = None
        self._fallback = AnalyticalPhysicsBackend()

    def _ensure_client(self):
        if self._pb is not None:
            return
        try:
            import pybullet as pb
            import pybullet_data

            self._pb = pb
            mode = pb.GUI if self._use_gui else pb.DIRECT
            self._client_id = pb.connect(mode)
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            pb.setGravity(0, 0, self._gravity, physicsClientId=self._client_id)
            pb.setTimeStep(self._time_step, physicsClientId=self._client_id)
            pb.loadURDF("plane.urdf", physicsClientId=self._client_id)
        except ImportError:
            log.warning("pybullet not installed; using analytical fallback")
            self._pb = None

    async def simulate(
        self,
        trajectory: Trajectory,
        robot_geometry: Dict[str, Any],
        workspace_objects: List[WorkspaceObject],
        floor_z: float = 0.0,
    ) -> PhysicsSimResult:
        self._ensure_client()
        if self._pb is None:
            return await self._fallback.simulate(
                trajectory, robot_geometry, workspace_objects, floor_z
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._simulate_sync,
            trajectory,
            robot_geometry,
            workspace_objects,
            floor_z,
        )

    def _simulate_sync(
        self,
        trajectory: Trajectory,
        robot_geometry: Dict[str, Any],
        workspace_objects: List[WorkspaceObject],
        floor_z: float,
    ) -> PhysicsSimResult:
        pb = self._pb
        cid = self._client_id
        t0 = time.perf_counter()
        violations = []

        # Load robot sphere
        robot_r = robot_geometry.get("radius_m", 0.15)
        mass_kg = robot_geometry.get("mass_kg", 20.0)
        col_shape = pb.createCollisionShape(pb.GEOM_SPHERE, radius=robot_r, physicsClientId=cid)
        viz_shape = pb.createVisualShape(
            pb.GEOM_SPHERE,
            radius=robot_r,
            rgbaColor=[0.2, 0.5, 1.0, 0.8],
            physicsClientId=cid,
        )

        start = trajectory.waypoints[0] if trajectory.waypoints else Pose3D()
        robot_id = pb.createMultiBody(
            baseMass=mass_kg,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=viz_shape,
            basePosition=[start.x, start.y, start.z],
            physicsClientId=cid,
        )

        # Load workspace objects
        obj_ids = []
        for obj in workspace_objects:
            oid = self._load_object(obj)
            if oid is not None:
                obj_ids.append((obj.object_id, oid))

        # Simulate trajectory
        overall_collision = CollisionResult(collision=False)
        for wp in trajectory.waypoints:
            pb.resetBasePositionAndOrientation(
                robot_id,
                [wp.x, wp.y, wp.z],
                [0, 0, 0, 1],
                physicsClientId=cid,
            )
            pb.stepSimulation(physicsClientId=cid)

            if wp.z < floor_z:
                violations.append(f"z={wp.z:.3f} below floor")

            # Check collisions
            for obj_name, obj_id in obj_ids:
                pts = pb.getContactPoints(robot_id, obj_id, physicsClientId=cid)
                if pts:
                    overall_collision = CollisionResult(
                        collision=True,
                        colliding_objects=[obj_name],
                        min_distance_m=0.0,
                        penetration_depth_m=abs(pts[0][8]),
                    )
                    violations.append(f"collision with {obj_name}")

        # Cleanup
        pb.removeBody(robot_id, physicsClientId=cid)
        for _, oid in obj_ids:
            pb.removeBody(oid, physicsClientId=cid)

        final = trajectory.waypoints[-1] if trajectory.waypoints else Pose3D()
        energy_j = (
            mass_kg * abs(self._gravity) * max(0, final.z - start.z)
            + trajectory.total_distance() * 5.0
        )
        duration_ms = trajectory.estimated_duration_ms()
        sim_time_ms = int((time.perf_counter() - t0) * 1000)

        return PhysicsSimResult(
            feasible=not violations,
            trajectory=trajectory,
            collision=overall_collision,
            final_pose=final,
            energy_j=energy_j,
            duration_ms=duration_ms,
            violations=violations,
            sim_time_ms=sim_time_ms,
        )

    def _load_object(self, obj: WorkspaceObject) -> Optional[int]:
        pb = self._pb
        cid = self._client_id
        d = obj.dimensions
        pos = [obj.pose.x, obj.pose.y, obj.pose.z]
        orn = [0, 0, 0, 1]

        if obj.shape == "box":
            half = [d.get("width", 0.1) / 2, d.get("depth", 0.1) / 2, d.get("height", 0.1) / 2]
            col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half, physicsClientId=cid)
        elif obj.shape == "sphere":
            col = pb.createCollisionShape(
                pb.GEOM_SPHERE, radius=d.get("radius", 0.1), physicsClientId=cid
            )
        elif obj.shape == "cylinder":
            col = pb.createCollisionShape(
                pb.GEOM_CYLINDER,
                radius=d.get("radius", 0.1),
                height=d.get("height", 0.2),
                physicsClientId=cid,
            )
        else:
            return None

        mass = 0 if obj.static else obj.mass_kg
        return pb.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col,
            basePosition=pos,
            baseOrientation=orn,
            physicsClientId=cid,
        )

    async def check_collision(
        self,
        pose: Pose3D,
        robot_geometry: Dict[str, Any],
        workspace_objects: List[WorkspaceObject],
    ) -> CollisionResult:
        return await self._fallback.check_collision(pose, robot_geometry, workspace_objects)

    def reset(self) -> None:
        if self._pb and self._client_id is not None:
            self._pb.resetSimulation(physicsClientId=self._client_id)


# ─────────────────────────────────────────────────────────────────────────────
#  SHADOW ENGINE  (thin orchestrator over a physics backend)
# ─────────────────────────────────────────────────────────────────────────────


class ShadowEngine:
    """
    Orchestrates pre-flight physics simulation (shadow preview) before
    any actuation is executed.
    """

    def __init__(
        self,
        backend: Optional[PhysicsBackend] = None,
        max_sim_time_ms: int = 500,
        default_floor_z: float = 0.0,
    ):
        self._backend = backend or AnalyticalPhysicsBackend()
        self._max_sim_time_ms = max_sim_time_ms
        self._floor_z = default_floor_z
        self._workspace_objects: List[WorkspaceObject] = []
        self._robot_geometries: Dict[str, Dict[str, Any]] = {}

    def register_robot(self, robot_id: str, geometry: Dict[str, Any]) -> None:
        self._robot_geometries[robot_id] = geometry

    def update_workspace(self, objects: List[WorkspaceObject]) -> None:
        self._workspace_objects = objects

    def add_object(self, obj: WorkspaceObject) -> None:
        self._workspace_objects = [
            o for o in self._workspace_objects if o.object_id != obj.object_id
        ]
        self._workspace_objects.append(obj)

    def remove_object(self, object_id: str) -> None:
        self._workspace_objects = [o for o in self._workspace_objects if o.object_id != object_id]

    async def preview(
        self,
        robot_id: str,
        trajectory: Trajectory,
    ) -> PhysicsSimResult:
        """Run a shadow simulation. Raises on timeout."""
        geometry = self._robot_geometries.get(robot_id, {"radius_m": 0.15, "mass_kg": 20.0})

        try:
            result = await asyncio.wait_for(
                self._backend.simulate(
                    trajectory,
                    geometry,
                    self._workspace_objects,
                    self._floor_z,
                ),
                timeout=self._max_sim_time_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            log.warning(
                "Shadow simulation timed out after %dms for robot %s",
                self._max_sim_time_ms,
                robot_id,
            )
            return PhysicsSimResult(
                feasible=False,
                trajectory=trajectory,
                collision=CollisionResult(collision=False),
                final_pose=Pose3D(),
                energy_j=0.0,
                duration_ms=0,
                violations=["Shadow simulation timeout"],
                sim_time_ms=self._max_sim_time_ms,
            )

        return result

    async def is_feasible(
        self,
        robot_id: str,
        trajectory: Trajectory,
    ) -> Tuple[bool, List[str]]:
        """Quick check: return (feasible, violations)."""
        result = await self.preview(robot_id, trajectory)
        return result.feasible, result.violations
