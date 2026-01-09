# func/classes.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

DragModel = Literal["G1", "G2", "G7", "Rod", "Sphere"]


def _req(d: dict[str, Any], key: str) -> Any:
  if key not in d:
    raise KeyError(f"Missing required key: {key}")
  return d[key]


def _as_float(x: Any, *, name: str) -> float:
  try:
    v = float(x)
  except Exception as e:
    raise TypeError(f"Field {name} must be a number, got: {type(x).__name__}") from e
  if not np.isfinite(v):
    raise ValueError(f"Field {name} must be finite, got: {v}")
  return v


def _as_str(x: Any, *, name: str) -> str:
  if not isinstance(x, str):
    raise TypeError(f"Field {name} must be str, got: {type(x).__name__}")
  s = x.strip()
  if not s:
    raise ValueError(f"Field {name} must be non-empty")
  return s


def _as_drag_model(x: Any, *, name: str) -> DragModel:
  s = _as_str(x, name=name)
  allowed = ("G1", "G2", "G7", "Rod", "Sphere")
  if s not in allowed:
    raise ValueError(f"Field {name} must be one of {allowed}, got: {s}")
  return s  # type: ignore[return-value]


@dataclass(slots=True, frozen=True)
class ProjectileDrag:
  model: DragModel
  bc: float

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "ProjectileDrag":
    model = _as_drag_model(_req(d, 'model'), name='projectile.drag.model')
    bc = _as_float(_req(d, 'bc'), name='projectile.drag.bc')
    if bc <= 0.0:
      raise ValueError(f"projectile.drag.bc must be > 0, got: {bc}")
    return ProjectileDrag(model=model, bc=bc)


@dataclass(slots=True, frozen=True)
class ProjectileSpec:
  mass_kg: float
  diameter_m: float
  length_m: float
  com_m: np.ndarray = field(repr=False)
  drag: Optional[ProjectileDrag] = None

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "ProjectileSpec":
    mass_g = _as_float(_req(d, 'mass_g'), name='projectile.mass_g')
    diameter_mm = _as_float(_req(d, 'diameter_mm'), name='projectile.diameter_mm')
    length_mm = _as_float(_req(d, 'length_mm'), name='projectile.length_mm')

    com = d.get('center_of_mass', {}) or {}
    x_mm = _as_float(com.get('x_mm', 0.0), name='projectile.center_of_mass.x_mm')
    y_mm = _as_float(com.get('y_mm', 0.0), name='projectile.center_of_mass.y_mm')
    z_mm = _as_float(com.get('z_mm', 0.0), name='projectile.center_of_mass.z_mm')

    drag_in = d.get('drag', None)
    drag = ProjectileDrag.from_dict(drag_in) if isinstance(drag_in, dict) else None

    mass_kg = mass_g / 1000.0
    diameter_m = diameter_mm / 1000.0
    length_m = length_mm / 1000.0
    com_m = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0

    if mass_kg <= 0.0:
      raise ValueError(f"projectile.mass_kg must be > 0, got: {mass_kg}")
    if diameter_m <= 0.0:
      raise ValueError(f"projectile.diameter_m must be > 0, got: {diameter_m}")
    if length_m <= 0.0:
      raise ValueError(f"projectile.length_m must be > 0, got: {length_m}")

    return ProjectileSpec(
      mass_kg=mass_kg,
      diameter_m=diameter_m,
      length_m=length_m,
      com_m=com_m,
      drag=drag,
    )

  @property
  def area_m2(self) -> float:
    r = 0.5 * self.diameter_m
    return float(np.pi * r * r)


@dataclass(slots=True, frozen=True)
class Ammo:
  id: str
  name: str
  mass_kg: float
  v0_mps: float
  projectile: ProjectileSpec

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "Ammo":
    ammo_id = _as_str(_req(d, 'id'), name='ammo.id')
    name = _as_str(_req(d, 'name'), name='ammo.name')
    mass_g = _as_float(_req(d, 'mass_g'), name='ammo.mass_g')

    vel = d.get('velocity', {}) or {}
    v0_mps = _as_float(_req(vel, 'v0_mps'), name='ammo.velocity.v0_mps')

    proj = ProjectileSpec.from_dict(_req(d, 'projectile'))

    mass_kg = mass_g / 1000.0
    if mass_kg <= 0.0:
      raise ValueError(f"ammo.mass_kg must be > 0, got: {mass_kg}")
    if v0_mps <= 0.0:
      raise ValueError(f"ammo.v0_mps must be > 0, got: {v0_mps}")

    return Ammo(
      id=ammo_id,
      name=name,
      mass_kg=mass_kg,
      v0_mps=v0_mps,
      projectile=proj,
    )


@dataclass(slots=True, frozen=True)
class BarrelSpec:
  length_m: float
  twist_rate_in_per_turn: float

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "BarrelSpec":
    length_mm = _as_float(_req(d, 'length_mm'), name='weapon.barrel.length_mm')
    twist_rate = _as_float(_req(d, 'twist_rate'), name='weapon.barrel.twist_rate')

    length_m = length_mm / 1000.0
    if length_m <= 0.0:
      raise ValueError(f"barrel.length_m must be > 0, got: {length_m}")
    if twist_rate <= 0.0:
      raise ValueError(f"barrel.twist_rate must be > 0, got: {twist_rate}")

    return BarrelSpec(length_m=length_m, twist_rate_in_per_turn=twist_rate)

  @property
  def twist_m_per_turn(self) -> float:
    return float(self.twist_rate_in_per_turn * 0.0254)

  @property
  def turns_per_meter(self) -> float:
    return float(1.0 / self.twist_m_per_turn)


@dataclass(slots=True, frozen=True)
class Weapon:
  id: str
  name: str
  mass_kg: float
  ammo_capacity: int
  barrel: BarrelSpec

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "Weapon":
    weapon_id = _as_str(_req(d, 'id'), name='weapon.id')
    name = _as_str(_req(d, 'name'), name='weapon.name')
    mass_kg = _as_float(_req(d, 'mass_kg'), name='weapon.mass_kg')
    ammo_capacity = int(_req(d, 'ammo_capacity'))
    barrel = BarrelSpec.from_dict(_req(d, 'barrel'))

    if mass_kg <= 0.0:
      raise ValueError(f"weapon.mass_kg must be > 0, got: {mass_kg}")
    if ammo_capacity <= 0:
      raise ValueError(f"weapon.ammo_capacity must be > 0, got: {ammo_capacity}")

    return Weapon(
      id=weapon_id,
      name=name,
      mass_kg=mass_kg,
      ammo_capacity=ammo_capacity,
      barrel=barrel,
    )


@dataclass(slots=True, frozen=True)
class Medium:
  id: str
  name: str
  model: str
  density_kg_m3: float
  temperature_C: float
  speed_of_sound_m_s: float

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "Medium":
    mid = _as_str(_req(d, 'id'), name='medium.id')
    name = _as_str(_req(d, 'name'), name='medium.name')
    model = _as_str(_req(d, 'model'), name='medium.model')

    rho = _as_float(_req(d, 'density_kg_m3'), name='medium.density_kg_m3')
    temp = _as_float(_req(d, 'temperature_C'), name='medium.temperature_C')
    sos = _as_float(_req(d, 'speed_of_sound_m_s'), name='medium.speed_of_sound_m_s')

    if rho <= 0.0:
      raise ValueError(f"medium.density_kg_m3 must be > 0, got: {rho}")
    if sos <= 0.0:
      raise ValueError(f"medium.speed_of_sound_m_s must be > 0, got: {sos}")

    return Medium(
      id=mid,
      name=name,
      model=model,
      density_kg_m3=rho,
      temperature_C=temp,
      speed_of_sound_m_s=sos,
    )


@dataclass(slots=True)
class ProjectileState:
  t_s: float
  pos_m: np.ndarray = field(repr=False)
  vel_mps: np.ndarray = field(repr=False)

  def __post_init__(self) -> None:
    self.t_s = float(self.t_s)

    self.pos_m = np.asarray(self.pos_m, dtype=np.float64).reshape(3)
    self.vel_mps = np.asarray(self.vel_mps, dtype=np.float64).reshape(3)

    if not np.isfinite(self.t_s):
      raise ValueError(f"state.t_s must be finite, got: {self.t_s}")
    if not np.all(np.isfinite(self.pos_m)):
      raise ValueError(f"state.pos_m must be finite, got: {self.pos_m}")
    if not np.all(np.isfinite(self.vel_mps)):
      raise ValueError(f"state.vel_mps must be finite, got: {self.vel_mps}")

  @property
  def speed_mps(self) -> float:
    return float(np.linalg.norm(self.vel_mps))

  def copy(self) -> "ProjectileState":
    return ProjectileState(
      t_s=float(self.t_s),
      pos_m=self.pos_m.copy(),
      vel_mps=self.vel_mps.copy(),
    )
