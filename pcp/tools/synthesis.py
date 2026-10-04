"""
PCP Dynamic Tool Synthesis v2
=================================
Given a robot's capability profile and a natural-language task description,
automatically compose and register new PCP tools by:

  1. Parsing the task with an LLM (or heuristic parser)
  2. Mapping required capabilities to existing tools/actuations
  3. Synthesising a new `SynthesisedTool` that chains them
  4. Registering the tool in the runtime tool-registry
  5. Optionally persisting the new tool to the marketplace

Design
------
- `CapabilityProfile` — set of actuations/sensors a robot exposes
- `ToolTemplate`      — parameterised recipe with slot variables
- `Synthesiser`       — orchestrates LLM call → template selection → code gen
- `SynthesisedTool`   — generated callable implementing the PCP tool protocol
- `ToolRegistry`      — runtime registry keyed by tool_id
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("pcp.synthesis")


# ─────────────────────────────────────────────────────────────────────────────
#  Capability Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

MOTION_ACTUATIONS = {"moveJ", "moveL", "cmd_vel", "navigate_to", "walk_to", "forward", "rotate"}
GRIPPER_ACTUATIONS = {"open_gripper", "close_gripper", "set_gripper_width"}
SENSOR_TYPES = {"joints", "tcp_pose", "joint_temps", "lidar", "camera", "imu", "battery", "odom"}


@dataclass
class CapabilityProfile:
    robot_id: str
    robot_class: str
    actuations: List[str]
    sensors: List[str]

    def has_motion(self) -> bool:
        return bool(set(self.actuations) & MOTION_ACTUATIONS)

    def has_gripper(self) -> bool:
        return bool(set(self.actuations) & GRIPPER_ACTUATIONS)

    def has_sensor(self, sensor: str) -> bool:
        return sensor in self.sensors


# ─────────────────────────────────────────────────────────────────────────────
#  Tool Templates
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ToolTemplate:
    """A parameterised recipe for composing a tool from sub-operations."""

    name: str
    description: str
    required_actuations: List[str]
    required_sensors: List[str]
    steps: List[Dict]  # ordered list of {op, params_template}
    parameters: List[Dict]  # JSON-Schema-style parameter descriptors

    def matches(self, profile: CapabilityProfile) -> bool:
        for act in self.required_actuations:
            if act not in profile.actuations:
                return False
        for sen in self.required_sensors:
            if sen not in profile.sensors:
                return False
        return True


# Built-in template library
BUILTIN_TEMPLATES: List[ToolTemplate] = [
    ToolTemplate(
        name="pick_and_place",
        description="Move to pick pose, grasp object, move to place pose, release",
        required_actuations=["moveL", "open_gripper", "close_gripper"],
        required_sensors=["tcp_pose"],
        parameters=[
            {
                "name": "pick_pose",
                "type": "array",
                "items": {"type": "number"},
                "minItems": 6,
                "maxItems": 6,
            },
            {
                "name": "place_pose",
                "type": "array",
                "items": {"type": "number"},
                "minItems": 6,
                "maxItems": 6,
            },
            {"name": "grasp_width", "type": "number", "default": 0.02},
        ],
        steps=[
            {"op": "actuation", "name": "open_gripper", "params": {}},
            {"op": "actuation", "name": "moveL", "params": {"pose": "{pick_pose}"}},
            {"op": "actuation", "name": "close_gripper", "params": {"width": "{grasp_width}"}},
            {"op": "actuation", "name": "moveL", "params": {"pose": "{place_pose}"}},
            {"op": "actuation", "name": "open_gripper", "params": {}},
        ],
    ),
    ToolTemplate(
        name="patrol_waypoints",
        description="Navigate a robot through a list of waypoints in sequence",
        required_actuations=["navigate_to"],
        required_sensors=["odom"],
        parameters=[
            {
                "name": "waypoints",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                },
            },
            {"name": "loop", "type": "boolean", "default": False},
        ],
        steps=[
            {
                "op": "for_each",
                "var": "wp",
                "in": "{waypoints}",
                "body": [
                    {
                        "op": "actuation",
                        "name": "navigate_to",
                        "params": {"x": "{wp.x}", "y": "{wp.y}"},
                    }
                ],
            },
        ],
    ),
    ToolTemplate(
        name="inspect_joints",
        description="Read all joint positions and temperatures, return structured report",
        required_actuations=[],
        required_sensors=["joints", "joint_temps"],
        parameters=[],
        steps=[
            {"op": "sensor", "name": "joints", "bind": "joint_data"},
            {"op": "sensor", "name": "joint_temps", "bind": "temp_data"},
            {"op": "return", "value": {"joints": "{joint_data}", "temperatures": "{temp_data}"}},
        ],
    ),
    ToolTemplate(
        name="charge_and_resume",
        description="Drive to nearest charging station, charge until level, return to prior pose",
        required_actuations=["navigate_to"],
        required_sensors=["odom", "battery"],
        parameters=[
            {"name": "target_charge", "type": "number", "default": 0.9},
            {
                "name": "charger_pose",
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            },
        ],
        steps=[
            {"op": "sensor", "name": "odom", "bind": "start_pose"},
            {
                "op": "actuation",
                "name": "navigate_to",
                "params": {"x": "{charger_pose.x}", "y": "{charger_pose.y}"},
            },
            {
                "op": "wait_condition",
                "sensor": "battery",
                "field": "level",
                "value": "{target_charge}",
            },
            {
                "op": "actuation",
                "name": "navigate_to",
                "params": {"x": "{start_pose.x}", "y": "{start_pose.y}"},
            },
        ],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Synthesised Tool
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SynthesisedTool:
    tool_id: str
    name: str
    description: str
    source_template: str
    parameters: List[Dict]
    code: str  # Python source generated for this tool
    created_at: float = field(default_factory=time.time)
    robot_class: str = ""

    def to_pcp_schema(self) -> Dict:
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "robot_class": self.robot_class,
            "source_template": self.source_template,
        }


def _render(template_str: str, bindings: Dict[str, Any]) -> Any:
    """Recursively resolve {var} placeholders in template strings/dicts/lists."""
    if isinstance(template_str, str):
        pattern = re.compile(r"\{([^}]+)\}")

        def replacer(m: re.Match) -> str:
            key = m.group(1)
            parts = key.split(".")
            val = bindings
            for p in parts:
                if isinstance(val, dict):
                    val = val.get(p, "")
                else:
                    val = getattr(val, p, "")
            return json.dumps(val) if not isinstance(val, str) else val

        return pattern.sub(replacer, template_str)
    if isinstance(template_str, dict):
        return {k: _render(v, bindings) for k, v in template_str.items()}
    if isinstance(template_str, list):
        return [_render(item, bindings) for item in template_str]
    return template_str


def _generate_tool_code(template: ToolTemplate, profile: CapabilityProfile) -> str:
    """Generate Python source that implements the tool as an async coroutine."""
    param_names = [p["name"] for p in template.parameters]
    param_str = ", ".join(param_names) if param_names else ""

    lines = [
        f'async def {template.name}(client, {param_str}{"," if param_str else ""}**kwargs):',
        f'    """Auto-synthesised: {template.description}"""',
        "    bindings = dict(locals())",
        "    results = {}",
    ]

    def emit_step(step: Dict, indent: int) -> List[str]:
        pad = "    " * indent
        op = step.get("op")
        if op == "actuation":
            return [
                f'{pad}await client.actuation_execute("{step["name"]}", {json.dumps(step.get("params", {}))})',
            ]
        if op == "sensor":
            bind = step.get("bind", "_")
            return [
                f'{pad}{bind} = await client.sensor_read("{step["name"]}")',
                f'{pad}bindings["{bind}"] = {bind}',
            ]
        if op == "for_each":
            sub = []
            sub.append(f'{pad}for {step["var"]} in {step["in"].strip("{}")}:')
            for s in step.get("body", []):
                sub.extend(emit_step(s, indent + 1))
            return sub
        if op == "return":
            return [f'{pad}return {json.dumps(step.get("value", {}))}']
        if op == "wait_condition":
            return [
                f"{pad}while True:",
                f'{pad}    _d = await client.sensor_read("{step["sensor"]}")',
                f'{pad}    if _d.get("{step["field"]}", 0) >= {step.get("value", 0)}: break',
                f"{pad}    await asyncio.sleep(1)",
            ]
        return [f"{pad}pass  # unknown op: {op}"]

    for step in template.steps:
        lines.extend(emit_step(step, 1))

    lines.append("    return results")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Tool Registry
# ─────────────────────────────────────────────────────────────────────────────


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, SynthesisedTool] = {}

    def register(self, tool: SynthesisedTool) -> None:
        self._tools[tool.tool_id] = tool
        log.info("Registered synthesised tool: %s (%s)", tool.name, tool.tool_id)

    def get(self, tool_id: str) -> Optional[SynthesisedTool]:
        return self._tools.get(tool_id)

    def list_tools(self) -> List[Dict]:
        return [t.to_pcp_schema() for t in self._tools.values()]

    def find_for_class(self, robot_class: str) -> List[SynthesisedTool]:
        return [
            t for t in self._tools.values() if t.robot_class == robot_class or not t.robot_class
        ]


# ─────────────────────────────────────────────────────────────────────────────
#  Synthesiser
# ─────────────────────────────────────────────────────────────────────────────


class Synthesiser:
    """
    Synthesise new tools from natural-language intent + capability profile.

    In production, replace `_llm_match` with a real LLM call (OpenAI, Claude, etc.).
    The heuristic fallback is used when no LLM key is configured.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        templates: Optional[List[ToolTemplate]] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        self.registry = registry
        self.templates = templates or BUILTIN_TEMPLATES
        self.llm = llm_client

    def _heuristic_match(self, intent: str, profile: CapabilityProfile) -> Optional[ToolTemplate]:
        """Simple keyword matching fallback."""
        intent_lower = intent.lower()
        kw_map = {
            "pick": "pick_and_place",
            "place": "pick_and_place",
            "grasp": "pick_and_place",
            "patrol": "patrol_waypoints",
            "waypoint": "patrol_waypoints",
            "navigate": "patrol_waypoints",
            "inspect": "inspect_joints",
            "joint": "inspect_joints",
            "temperature": "inspect_joints",
            "charge": "charge_and_resume",
            "battery": "charge_and_resume",
        }
        for kw, tmpl_name in kw_map.items():
            if kw in intent_lower:
                tmpl = next((t for t in self.templates if t.name == tmpl_name), None)
                if tmpl and tmpl.matches(profile):
                    return tmpl
        # Fallback: return first matching template
        for tmpl in self.templates:
            if tmpl.matches(profile):
                return tmpl
        return None

    async def _llm_match(self, intent: str, profile: CapabilityProfile) -> Optional[ToolTemplate]:
        """Use LLM to select the best template. Falls back to heuristic if no client."""
        if self.llm is None:
            return self._heuristic_match(intent, profile)

        prompt = (
            f"Given a robot with actuations {profile.actuations} and sensors {profile.sensors}, "
            f"select the best tool template name for: '{intent}'. "
            f"Available templates: {[t.name for t in self.templates]}. "
            "Reply with just the template name."
        )
        try:
            response = await self.llm.complete(prompt)
            name = response.strip().lower()
            return next((t for t in self.templates if t.name == name and t.matches(profile)), None)
        except Exception:
            return self._heuristic_match(intent, profile)

    async def synthesise(
        self, intent: str, profile: CapabilityProfile
    ) -> Optional[SynthesisedTool]:
        """Synthesise a tool for the given intent and robot profile."""
        template = await self._llm_match(intent, profile)
        if template is None:
            log.warning("No matching template for intent: %s", intent)
            return None

        code = _generate_tool_code(template, profile)
        tool = SynthesisedTool(
            tool_id=str(uuid.uuid4()),
            name=f"{template.name}_{profile.robot_id[:8]}",
            description=f"{template.description} (synthesised for {profile.robot_id})",
            source_template=template.name,
            parameters=template.parameters,
            code=code,
            robot_class=profile.robot_class,
        )
        self.registry.register(tool)
        return tool

    def add_template(self, template: ToolTemplate) -> None:
        """Add a custom template to the synthesiser's library."""
        self.templates.append(template)
        log.info("Added template: %s", template.name)
