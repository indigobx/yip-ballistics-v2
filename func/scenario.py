# func/scenario.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


def _req(d: dict[str, Any], key: str) -> Any:
  if key not in d:
    raise KeyError(f"Missing required key: {key}")
  return d[key]


def _as_str(x: Any, *, name: str) -> str:
  if not isinstance(x, str):
    raise TypeError(f"Field {name} must be str, got: {type(x).__name__}")
  s = x.strip()
  if not s:
    raise ValueError(f"Field {name} must be non-empty")
  return s


def _as_float(x: Any, *, name: str) -> float:
  try:
    v = float(x)
  except Exception as e:
    raise TypeError(f"Field {name} must be a number, got: {type(x).__name__}") from e
  if not np.isfinite(v):
    raise ValueError(f"Field {name} must be finite, got: {v}")
  return v


def _as_opt_float(x: Any, *, name: str) -> Optional[float]:
  if x is None:
    return None
  return _as_float(x, name=name)


@dataclass(slots=True, frozen=True)
class ScenarioRef:
  id: str

  @staticmethod
  def from_dict(d: dict[str, Any], *, name: str) -> "ScenarioRef":
    return ScenarioRef(id=_as_str(_req(d, 'id'), name=f"{name}.id"))


@dataclass(slots=True, frozen=True)
class WindSpec:
  direction_deg: float = 0.0
  speed_mps: float = 0.0

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "WindSpec":
    return WindSpec(
      direction_deg=_as_float(d.get('direction_deg', 0.0), name='environment.wind.direction_deg'),
      speed_mps=_as_float(d.get('speed_mps', 0.0), name='environment.wind.speed_mps'),
    )


@dataclass(slots=True, frozen=True)
class GroundSpec:
  medium: str = 'air'

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "GroundSpec":
    return GroundSpec(medium=_as_str(d.get('medium', 'air'), name='environment.ground.medium'))


@dataclass(slots=True, frozen=True)
class EnvironmentSpec:
  medium: str
  overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
  wind: WindSpec = field(default_factory=WindSpec)
  ground: GroundSpec = field(default_factory=GroundSpec)

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "EnvironmentSpec":
    medium = _as_str(_req(d, 'medium'), name='environment.medium')
    overrides = d.get('overrides', {}) or {}
    wind = WindSpec.from_dict(d.get('wind', {}) or {})
    ground = GroundSpec.from_dict(d.get('ground', {}) or {})
    return EnvironmentSpec(medium=medium, overrides=overrides, wind=wind, ground=ground)


@dataclass(slots=True, frozen=True)
class ShotSpec:
  elevation_angle_deg: float = 0.0
  sight_offset_angle_deg: float = 0.0

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "ShotSpec":
    return ShotSpec(
      elevation_angle_deg=_as_float(d.get('elevation_angle_deg', 0.0),
                                    name='shot.elevation_angle_deg'),
      sight_offset_angle_deg=_as_float(d.get('sight_offset_angle_deg', 0.0),
                                       name='shot.sight_offset_angle_deg'),
    )


@dataclass(slots=True, frozen=True)
class LimitsSpec:
  max_distance_m: float = 1000.0
  max_drop_m: float = 100.0
  max_height_m: float = 100.0
  max_drift_m: float = 100.0
  min_energy_j: float = 0.0

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "LimitsSpec":
    return LimitsSpec(
      max_distance_m=_as_float(d.get('max_distance_m', 1000.0), name='limits.max_distance_m'),
      max_drop_m=_as_float(d.get('max_drop_m', 100.0), name='limits.max_drop_m'),
      max_height_m=_as_float(d.get('max_height_m', 100.0), name='limits.max_height_m'),
      max_drift_m=_as_float(d.get('max_drift_m', 100.0), name='limits.max_drift_m'),
      min_energy_j=_as_float(d.get('min_energy_j', 0.0), name='limits.min_energy_j'),
    )


@dataclass(slots=True, frozen=True)
class TargetLayerSpec:
  name: str
  medium: str
  thickness_m: float

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "TargetLayerSpec":
    name = _as_str(d.get('name', d.get('medium', 'layer')), name='target.layer.name')
    medium = _as_str(_req(d, 'medium'), name='target.layer.medium')
    thickness_mm = _as_float(_req(d, 'thickness_mm'), name='target.layer.thickness_mm')
    thickness_m = thickness_mm / 1000.0
    if thickness_m <= 0.0:
      raise ValueError(f"target.layer.thickness_m must be > 0, got: {thickness_m}")
    return TargetLayerSpec(name=name, medium=medium, thickness_m=thickness_m)


@dataclass(slots=True, frozen=True)
class TargetSpec:
  name: str
  position_m: float
  layers: list[TargetLayerSpec] = field(default_factory=list)

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "TargetSpec":
    name = _as_str(_req(d, 'name'), name='target.name')
    position_m = _as_float(_req(d, 'position_m'), name='target.position_m')
    layers_in = d.get('layers', []) or []
    layers = [TargetLayerSpec.from_dict(x) for x in layers_in]
    return TargetSpec(name=name, position_m=position_m, layers=layers)


@dataclass(slots=True, frozen=True)
class Scenario:
  schema: str
  id: str
  description: str

  weapon: ScenarioRef
  ammo: ScenarioRef
  environment: EnvironmentSpec
  shot: ShotSpec
  limits: LimitsSpec
  target: list[TargetSpec] = field(default_factory=list)

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "Scenario":
    schema = _as_str(_req(d, 'schema'), name='schema')
    sid = _as_str(_req(d, 'id'), name='id')
    desc = _as_str(d.get('description', ''), name='description') if 'description' in d else ''

    weapon = ScenarioRef.from_dict(_req(d, 'weapon'), name='weapon')
    ammo = ScenarioRef.from_dict(_req(d, 'ammo'), name='ammo')
    env = EnvironmentSpec.from_dict(_req(d, 'environment'))
    shot = ShotSpec.from_dict(d.get('shot', {}) or {})
    limits = LimitsSpec.from_dict(d.get('limits', {}) or {})

    targets_in = d.get('target', []) or []
    targets = [TargetSpec.from_dict(x) for x in targets_in]

    return Scenario(
      schema=schema,
      id=sid,
      description=desc,
      weapon=weapon,
      ammo=ammo,
      environment=env,
      shot=shot,
      limits=limits,
      target=targets,
    )
