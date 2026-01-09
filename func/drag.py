# func/drag.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(slots=True, frozen=True)
class DragConfig:
  enabled: bool = True
  # Плейсхолдер для MVP: насколько сильно BC влияет на эффективный Cd.
  # Когда подключишь "настоящую" G7-таблицу, это поле можно будет убрать.
  bc_to_cd_scale: float = 1.0


def _cd_reference(model: str) -> float:
  # Плейсхолдеры. Задача сейчас — получить стабильный векторный drag и интерфейс.
  # Позже это заменится на табличную G-функцию.
  if model == 'Sphere':
    return 0.47
  if model == 'Rod':
    return 1.00
  if model in ('G1', 'G2', 'G7'):
    return 0.30
  return 0.30


def _get_drag(projectile: Any) -> Optional[Any]:
  # Поддерживаем текущую ситуацию, где в Ammo.projectile лежит ProjectileSpec,
  # а отдельный класс Projectile может существовать параллельно.
  # Если у объекта есть .drag — используем его, иначе drag отсутствует.
  return getattr(projectile, 'drag', None)


def _get_area_m2(projectile: Any) -> float:
  # ProjectileSpec.area_m2 — property, у Projectile тоже есть area_m2.
  area = getattr(projectile, 'area_m2', None)
  if area is None:
    raise AttributeError("Projectile object must provide area_m2")
  return float(area)


def _get_mass_kg(projectile: Any) -> float:
  mass = getattr(projectile, 'mass_kg', None)
  if mass is None:
    raise AttributeError("Projectile object must provide mass_kg")
  return float(mass)


def drag_accel(
  *,
  v_xyz_mps: tuple[float, float, float],
  projectile: Any,
  rho_kg_m3: float,
  cfg: DragConfig,
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

  model = getattr(drag, 'model', None)
  bc = getattr(drag, 'bc', None)
  if model is None or bc is None:
    return (0.0, 0.0, 0.0)

  model_s = str(model)
  bc_f = float(bc)
  if bc_f <= 0.0:
    return (0.0, 0.0, 0.0)

  cd_ref = _cd_reference(model_s)
  cd = cd_ref * (cfg.bc_to_cd_scale / max(bc_f, 1e-9))

  area_m2 = _get_area_m2(projectile)
  mass_kg = _get_mass_kg(projectile)

  v = v2 ** 0.5
  inv_v = 1.0 / v

  # Fd = 0.5 * rho * Cd * A * v^2
  # a = F/m, направлена против скорости: -a_mag * v_hat
  a_mag = 0.5 * float(rho_kg_m3) * cd * area_m2 * v2 / max(mass_kg, 1e-9)

  ax = -a_mag * vx * inv_v
  ay = -a_mag * vy * inv_v
  az = -a_mag * vz * inv_v
  return (ax, ay, az)
