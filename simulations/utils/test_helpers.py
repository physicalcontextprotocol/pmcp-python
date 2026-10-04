#!/usr/bin/env python3
"""
PCP Test Utilities
====================

Common test utilities and fixtures.

Usage:
    python -m tests.utils.test_helpers
"""

import asyncio
import random
import time
import unittest
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class TestRobot:
    """Test robot fixture."""
    robot_id: str
    name: str
    robot_type: str
    status: str
    zone: str
    health_score: float


@dataclass
class TestAlert:
    """Test alert fixture."""
    alert_id: str
    severity: str
    title: str
    message: str
    robot_id: Optional[str]
    created_at: float


@dataclass
class TestTask:
    """Test task fixture."""
    task_id: str
    task_type: str
    robot_id: str
    status: str
    priority: int


def generate_test_robots(count: int = 10) -> List[TestRobot]:
    """Generate test robot fixtures."""
    robots = []
    robot_types = ["turtlebot3", "ur5", "otto100", "fetch", " Sawyer"]
    statuses = ["online", "offline", "error", "maintenance"]
    zones = ["warehouse-a", "warehouse-b", "loading-dock", "maintenance"]

    for i in range(count):
        robots.append(TestRobot(
            robot_id=f"robot-{i:03d}",
            name=f"Test Robot {i}",
            robot_type=random.choice(robot_types),
            status=random.choice(statuses),
            zone=random.choice(zones),
            health_score=random.uniform(50, 100),
        ))
    return robots


def generate_test_alerts(count: int = 20) -> List[TestAlert]:
    """Generate test alert fixtures."""
    alerts = []
    severities = ["critical", "high", "medium", "low", "info"]
    titles = [
        "High CPU Usage",
        "Low Battery",
        "Network Disconnection",
        "Sensor Failure",
        "Collision Detected",
        "Task Timeout",
        "Authentication Failed",
    ]

    for i in range(count):
        alerts.append(TestAlert(
            alert_id=f"alert-{i:03d}",
            severity=random.choice(severities),
            title=random.choice(titles),
            message=f"Test alert message {i}",
            robot_id=f"robot-{random.randint(0, 9):03d}" if random.random() > 0.2 else None,
            created_at=time.time() - random.randint(0, 3600),
        ))
    return alerts


def generate_test_tasks(count: int = 15) -> List[TestTask]:
    """Generate test task fixtures."""
    tasks = []
    task_types = ["navigation", "inspection", "manipulation", "transport", "maintenance"]
    statuses = ["pending", "running", "completed", "failed", "cancelled"]

    for i in range(count):
        tasks.append(TestTask(
            task_id=f"task-{i:03d}",
            task_type=random.choice(task_types),
            robot_id=f"robot-{random.randint(0, 9):03d}",
            status=random.choice(statuses),
            priority=random.randint(1, 10),
        ))
    return tasks


class MockAPIClient:
    """Mock API client for testing."""

    def __init__(self):
        self._robots = generate_test_robots()
        self._alerts = generate_test_alerts()
        self._tasks = generate_test_tasks()

    async def get_robots(self) -> List[Dict[str, Any]]:
        """Get all robots."""
        await asyncio.sleep(0.01)
        return [r.__dict__ for r in self._robots]

    async def get_robot(self, robot_id: str) -> Optional[Dict[str, Any]]:
        """Get robot by ID."""
        await asyncio.sleep(0.01)
        for robot in self._robots:
            if robot.robot_id == robot_id:
                return robot.__dict__
        return None

    async def get_alerts(self) -> List[Dict[str, Any]]:
        """Get all alerts."""
        await asyncio.sleep(0.01)
        return [a.__dict__ for a in self._alerts]

    async def get_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks."""
        await asyncio.sleep(0.01)
        return [t.__dict__ for t in self._tasks]


class TestHelpers:
    """Test helper functions."""

    @staticmethod
    def assert_valid_robot(robot: Dict[str, Any]):
        """Assert robot has valid structure."""
        assert "robot_id" in robot
        assert "name" in robot
        assert "robot_type" in robot
        assert "status" in robot

    @staticmethod
    def assert_valid_alert(alert: Dict[str, Any]):
        """Assert alert has valid structure."""
        assert "alert_id" in alert
        assert "severity" in alert
        assert "title" in alert
        assert "message" in alert
        assert alert["severity"] in ["critical", "high", "medium", "low", "info"]

    @staticmethod
    def assert_valid_task(task: Dict[str, Any]):
        """Assert task has valid structure."""
        assert "task_id" in task
        assert "task_type" in task
        assert "robot_id" in task
        assert "status" in task


class TestFixtures(unittest.TestCase):
    """Test fixture generation."""

    def test_generate_robots(self):
        robots = generate_test_robots(10)
        self.assertEqual(len(robots), 10)
        self.assertTrue(all(hasattr(r, "robot_id") for r in robots))

    def test_generate_alerts(self):
        alerts = generate_test_alerts(20)
        self.assertEqual(len(alerts), 20)
        self.assertTrue(all(hasattr(a, "alert_id") for a in alerts))

    def test_generate_tasks(self):
        tasks = generate_test_tasks(15)
        self.assertEqual(len(tasks), 15)
        self.assertTrue(all(hasattr(t, "task_id") for t in tasks))

    def test_mock_api_client(self):
        client = MockAPIClient()

        async def test():
            robots = await client.get_robots()
            self.assertEqual(len(robots), 10)

            alerts = await client.get_alerts()
            self.assertEqual(len(alerts), 20)

            tasks = await client.get_tasks()
            self.assertEqual(len(tasks), 15)

        asyncio.run(test())


if __name__ == "__main__":
    unittest.main()