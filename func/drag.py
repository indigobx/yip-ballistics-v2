# func/drag.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from func.ballistic_tables import DragTable


@dataclass(slots=True, frozen=True)
class DragConfig:
  enabled: bool = True
  default_speed_of_sound_m_s: float = 343.26
  scale: float = 1.0


_DRAG_TABLES: Dict[str, DragTable] = {}

# G1/G7 BC are typically specified in lb/in^2; convert to kg/m^2 for SI math.
_LB_PER_IN2_TO_KG_PER_M2 = 0.45359237 / (0.0254 * 0.0254)


def init_drag_tables(tables: Dict[str, DragTable]) -> None:
  _DRAG_TABLES.clear()
  _DRAG_TABLES.update(tables)


def available_drag_models() -> list[str]:
  return sorted(_DRAG_TABLES.keys())


def _get(obj: Any, name: str, default: Any = None) -> Any:
  return getattr(obj, name, default)


def _get_drag(projectile: Any) -> Optional[Any]:
  return _get(projectile, 'drag', None)


def _get_area_m2(projectile: Any) -> float:
  area = _get(projectile, 'area_m2', None)
  if area is None:
    raise AttributeError('Projectile must provide area_m2')
  return float(area)


def _get_mass_kg(projectile: Any) -> float:
  mass = _get(projectile, 'mass_kg', None)
  if mass is None:
    raise AttributeError('Projectile must provide mass_kg')
  return float(mass)


def _interp_cd(model_id: str, mach: float) -> float:
  if model_id not in _DRAG_TABLES:
    avail = ', '.join(available_drag_models())
    raise KeyError(f"Drag model not loaded: {model_id}. Available: {avail}")

  t = _DRAG_TABLES[model_id]
  m = float(np.clip(float(mach), float(t.mach[0]), float(t.mach[-1])))
  return float(np.interp(m, t.mach, t.cd))


def _a_mag_from_cd(
  *,
  rho_kg_m3: float,
  v2_m2ps2: float,
  cd: float,
  area_m2: float,
  mass_kg: float,
) -> float:
  return 0.5 * float(rho_kg_m3) * cd * area_m2 * v2_m2ps2 / max(mass_kg, 1e-9)


def drag_accel(
  *,
  v_xyz_mps: tuple[float, float, float],
  projectile: Any,
  rho_kg_m3: float,
  cfg: DragConfig,
  speed_of_sound_m_s: Optional[float] = None,
) -> tuple[float, float, float]:
  if not cfg.enabled:
    return (0.0, 0.0, 0.0)

  vx, vy, vz = v_xyz_mps
  v2 = vx * vx + vy * vy + vz * vz
  if v2 <= 0.0:
    return (0.0, 0.0, 0.0)

  drag = _get_drag(projectile)
  if drag is None:
    return (0.0, 0.0, 0.0)

  kind = str(_get(drag, 'kind', 'standard')).strip()
  model_id = str(_get(drag, 'model', 'G7')).strip()

  v = v2 ** 0.5
  inv_v = 1.0 / v

  sos = float(speed_of_sound_m_s) if speed_of_sound_m_s is not None else cfg.default_speed_of_sound_m_s
  mach = v / max(sos, 1e-9)

  a_mag = 0.0

  if kind == 'standard':
    bc = _get(drag, 'bc', None)
    if bc is None:
      raise ValueError('projectile.drag.bc is required for kind=standard')
    bc_f = float(bc)
    if bc_f <= 0.0:
      raise ValueError(f"projectile.drag.bc must be > 0, got: {bc_f}")

    cd_ref = _interp_cd(model_id, mach)
    bc_si = bc_f * _LB_PER_IN2_TO_KG_PER_M2
    a_mag = 0.5 * float(rho_kg_m3) * cd_ref * v2 / max(bc_si, 1e-9)

  elif kind == 'primitive':
    # primitive использует те же таблицы по model_id (Rod/Sphere),
    # но без BC-масштабирования
    cd = _interp_cd(model_id, mach)
    area_m2 = _get_area_m2(projectile)
    mass_kg = _get_mass_kg(projectile)
    if mass_kg <= 0.0 or not (cd > 0.0):
      return (0.0, 0.0, 0.0)
    a_mag = _a_mag_from_cd(
      rho_kg_m3=rho_kg_m3,
      v2_m2ps2=v2,
      cd=cd,
      area_m2=area_m2,
      mass_kg=mass_kg,
    )

  elif kind == 'cd_const':
    cd_val = _get(drag, 'cd', None)
    if cd_val is None:
      raise ValueError('projectile.drag.cd is required for kind=cd_const')
    cd = float(cd_val)
    area_m2 = _get_area_m2(projectile)
    mass_kg = _get_mass_kg(projectile)
    if mass_kg <= 0.0 or not (cd > 0.0):
      return (0.0, 0.0, 0.0)
    a_mag = _a_mag_from_cd(
      rho_kg_m3=rho_kg_m3,
      v2_m2ps2=v2,
      cd=cd,
      area_m2=area_m2,
      mass_kg=mass_kg,
    )

  elif kind == 'cd_table':
    table = _get(drag, 'cd_vs_mach', None)
    if not isinstance(table, list) or len(table) < 2:
      raise ValueError('projectile.drag.cd_vs_mach must be list[[mach, cd], ...]')

    xs: list[float] = []
    ys: list[float] = []
    for row in table:
      if not isinstance(row, (list, tuple)) or len(row) != 2:
        raise ValueError('Each cd_vs_mach row must be [mach, cd]')
      xs.append(float(row[0]))
      ys.append(float(row[1]))

    pairs = sorted(zip(xs, ys), key=lambda p: p[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    m = float(np.clip(float(mach), xs[0], xs[-1]))
    cd = float(np.interp(m, xs, ys))

    area_m2 = _get_area_m2(projectile)
    mass_kg = _get_mass_kg(projectile)
    if mass_kg <= 0.0 or not (cd > 0.0):
      return (0.0, 0.0, 0.0)
    a_mag = _a_mag_from_cd(
      rho_kg_m3=rho_kg_m3,
      v2_m2ps2=v2,
      cd=cd,
      area_m2=area_m2,
      mass_kg=mass_kg,
    )

  else:
    raise ValueError(f"Unknown projectile.drag.kind: {kind}")

  if not (a_mag > 0.0):
    return (0.0, 0.0, 0.0)

  a_mag = a_mag * float(cfg.scale)

  # a направлена против скорости: -a_mag * v_hat
  ax = -a_mag * vx * inv_v
  ay = -a_mag * vy * inv_v
  az = -a_mag * vz * inv_v
  return (ax, ay, az)
