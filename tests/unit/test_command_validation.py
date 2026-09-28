#!/usr/bin/env python3
"""
P-MCP Command Validation Tests
===============================

Unit tests for command validation and safety checks.

Usage:
    python -m tests.unit.test_command_validation
"""

import unittest
import time
import json
import threading
from typing import List, Dict, Any
from dataclasses import dataclass, field


class CommandValidator:
    """Validates robot commands before execution."""

    def __init__(self):
        self._validation_rules: Dict[str, Any] = {}
        self._lock = threading.RLock()

    KNOWN_COMMANDS = ("move", "gripper", "navigation")

    def validate_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a command against all rules."""
        errors = []
        warnings = []

        if not command.get("command"):
            errors.append("Missing command")

        if not command.get("robot_id"):
            errors.append("Missing robot_id")

        command_type = command.get("command")
        if command_type == "move":
            result = self._validate_move_command(command)
            errors.extend(result.get("errors", []))
            warnings.extend(result.get("warnings", []))
        elif command_type == "gripper":
            result = self._validate_gripper_command(command)
            errors.extend(result.get("errors", []))
            warnings.extend(result.get("warnings", []))
        elif command_type == "navigation":
            result = self._validate_navigation_command(command)
            errors.extend(result.get("errors", []))
            warnings.extend(result.get("warnings", []))
        elif command_type and command_type not in self.KNOWN_COMMANDS:
            errors.append(f"Unknown command: {command_type}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def _validate_move_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Validate movement commands."""
        errors = []
        warnings = []

        position = command.get("position", {})
        if not isinstance(position, dict):
            errors.append("Position must be a dictionary")
            return {"errors": errors, "warnings": warnings}

        for axis in ["x", "y", "z"]:
            if axis in position:
                value = position[axis]
                if not isinstance(value, (int, float)):
                    errors.append(f"Position {axis} must be a number")
                elif value > 100 or value < -100:
                    warnings.append(f"Position {axis} is outside typical range")

        velocity = command.get("velocity", {})
        if isinstance(velocity, dict):
            speed = 0
            for axis in ["x", "y", "z"]:
                if axis in velocity:
                    speed += velocity.get(axis, 0) ** 2
            speed = speed ** 0.5
            if speed > 2.0:
                errors.append(f"Velocity {speed:.2f} m/s exceeds safety limit")
            elif speed > 1.0:
                warnings.append(f"Velocity {speed:.2f} m/s is high")

        return {"errors": errors, "warnings": warnings}

    def _validate_gripper_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Validate gripper commands."""
        errors = []
        warnings = []

        gripper_action = command.get("action")
        if gripper_action not in ["open", "close", "set_position"]:
            errors.append(f"Invalid gripper action: {gripper_action}")

        if gripper_action == "set_position":
            position = command.get("position", 0)
            if not isinstance(position, (int, float)):
                errors.append("Gripper position must be a number")
            elif position < 0 or position > 100:
                errors.append("Gripper position must be between 0 and 100")

        return {"errors": errors, "warnings": warnings}

    def _validate_navigation_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Validate navigation commands."""
        errors = []
        warnings = []

        target = command.get("target", {})
        if not target:
            errors.append("Missing navigation target")

        if "position" in target:
            pos = target["position"]
            for axis in ["x", "y"]:
                if axis in pos:
                    if pos[axis] > 1000 or pos[axis] < -1000:
                        warnings.append(f"Navigation target {axis} is very far")

        tolerance = command.get("tolerance", 0.1)
        if tolerance <= 0 or tolerance > 1.0:
            warnings.append(f"Tolerance {tolerance} may be too large")

        return {"errors": errors, "warnings": warnings}


class TestCommandValidation(unittest.TestCase):

    def setUp(self):
        self.validator = CommandValidator()

    def test_valid_move_command(self):
        command = {
            "command": "move",
            "robot_id": "robot-001",
            "position": {"x": 1.0, "y": 2.0, "z": 0.0},
            "velocity": {"x": 0.5, "y": 0.0, "z": 0.0},
        }
        result = self.validator.validate_command(command)
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)

    def test_missing_robot_id(self):
        command = {
            "command": "move",
            "position": {"x": 1.0, "y": 2.0},
        }
        result = self.validator.validate_command(command)
        self.assertFalse(result["valid"])
        self.assertIn("Missing robot_id", result["errors"])

    def test_missing_command(self):
        command = {
            "robot_id": "robot-001",
            "position": {"x": 1.0},
        }
        result = self.validator.validate_command(command)
        self.assertFalse(result["valid"])
        self.assertIn("Missing command", result["errors"])

    def test_excessive_velocity(self):
        command = {
            "command": "move",
            "robot_id": "robot-001",
            "position": {"x": 1.0, "y": 0.0},
            "velocity": {"x": 5.0, "y": 0.0},
        }
        result = self.validator.validate_command(command)
        self.assertFalse(result["valid"])
        self.assertTrue(any("exceeds safety limit" in e for e in result["errors"]))

    def test_high_velocity_warning(self):
        command = {
            "command": "move",
            "robot_id": "robot-001",
            "position": {"x": 1.0, "y": 0.0},
            "velocity": {"x": 1.5, "y": 0.0},
        }
        result = self.validator.validate_command(command)
        self.assertTrue(result["valid"])
        self.assertTrue(any("high" in w.lower() for w in result["warnings"]))

    def test_valid_gripper_command(self):
        command = {
            "command": "gripper",
            "robot_id": "robot-001",
            "action": "open",
        }
        result = self.validator.validate_command(command)
        self.assertTrue(result["valid"])

    def test_invalid_gripper_action(self):
        command = {
            "command": "gripper",
            "robot_id": "robot-001",
            "action": "invalid_action",
        }
        result = self.validator.validate_command(command)
        self.assertFalse(result["valid"])

    def test_gripper_position_out_of_range(self):
        command = {
            "command": "gripper",
            "robot_id": "robot-001",
            "action": "set_position",
            "position": 150,
        }
        result = self.validator.validate_command(command)
        self.assertFalse(result["valid"])
        self.assertTrue(any("between 0 and 100" in e for e in result["errors"]))

    def test_valid_navigation_command(self):
        command = {
            "command": "navigation",
            "robot_id": "robot-001",
            "target": {"position": {"x": 5.0, "y": 10.0}},
            "tolerance": 0.2,
        }
        result = self.validator.validate_command(command)
        self.assertTrue(result["valid"])

    def test_navigation_missing_target(self):
        command = {
            "command": "navigation",
            "robot_id": "robot-001",
        }
        result = self.validator.validate_command(command)
        self.assertFalse(result["valid"])
        self.assertIn("Missing navigation target", result["errors"])

    def test_navigation_far_target_warning(self):
        command = {
            "command": "navigation",
            "robot_id": "robot-001",
            "target": {"position": {"x": 5000.0, "y": 0.0}},
        }
        result = self.validator.validate_command(command)
        self.assertTrue(result["valid"])
        self.assertTrue(any("very far" in w.lower() for w in result["warnings"]))


class TestBatchValidation(unittest.TestCase):
    """Test batch command validation."""

    def setUp(self):
        self.validator = CommandValidator()

    def test_batch_validation(self):
        commands = [
            {"command": "move", "robot_id": "robot-001", "position": {"x": 1.0}},
            {"command": "gripper", "robot_id": "robot-002", "action": "open"},
            {"command": "navigation", "robot_id": "robot-003", "target": {"position": {"x": 5.0}}},
        ]

        results = []
        for cmd in commands:
            results.append(self.validator.validate_command(cmd))

        self.assertEqual(len(results), 3)
        self.assertTrue(all(r["valid"] for r in results))

    def test_batch_with_invalid_commands(self):
        commands = [
            {"command": "move", "robot_id": "robot-001", "position": {"x": 1.0}},
            {"command": "invalid", "robot_id": "robot-002"},
            {"command": "gripper", "robot_id": "robot-003", "action": "open"},
        ]

        results = []
        for cmd in commands:
            results.append(self.validator.validate_command(cmd))

        self.assertEqual(len(results), 3)
        self.assertTrue(results[0]["valid"])
        self.assertFalse(results[1]["valid"])
        self.assertTrue(results[2]["valid"])


class TestConcurrentValidation(unittest.TestCase):
    """Test thread-safe command validation."""

    def setUp(self):
        self.validator = CommandValidator()
        self.results = []
        self.lock = threading.Lock()

    def test_concurrent_validation(self):
        def validate_commands():
            for i in range(100):
                command = {
                    "command": "move",
                    "robot_id": f"robot-{i % 10}",
                    "position": {"x": float(i)},
                }
                result = self.validator.validate_command(command)
                with self.lock:
                    self.results.append(result)

        threads = [threading.Thread(target=validate_commands) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(self.results), 500)
        self.assertTrue(all(r["valid"] for r in self.results))


class TestPerformanceMetrics(unittest.TestCase):
    """Test validation performance."""

    def setUp(self):
        self.validator = CommandValidator()

    def test_validation_performance(self):
        iterations = 1000
        start_time = time.time()

        for i in range(iterations):
            command = {
                "command": "move",
                "robot_id": "robot-001",
                "position": {"x": float(i), "y": float(i), "z": 0.0},
                "velocity": {"x": 0.5, "y": 0.3},
            }
            self.validator.validate_command(command)

        elapsed = time.time() - start_time
        ops_per_sec = iterations / elapsed

        self.assertGreater(ops_per_sec, 100)
        print(f"Validation performance: {ops_per_sec:.2f} ops/sec")


if __name__ == "__main__":
    unittest.main()