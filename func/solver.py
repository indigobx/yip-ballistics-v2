# func/solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from func.classes import Ammo, Medium, ProjectileState, Weapon
from func.drag import DragConfig, drag_accel
from func.scenario import Scenario


Vec3 = np.ndarray


@dataclass(slots=True, frozen=True)
class SolverConfig:
  dt_s: float = 1.0 / 240.0
  t_max_s: float = 5.0
  integrator: str = 'rk4'
  drag_enabled: bool = True


@dataclass(slots=True, frozen=True)
class SolverMeta:
  scenario_id: str
  weapon_id: str
  ammo_id: str
  environment_medium_id: str
  dt_s: float
  t_max_s: float
  integrator: str
  stop_reason: str
  n_steps: int


@dataclass(slots=True, frozen=True)
class SolverResult:
  trajectory: pd.DataFrame
  events: pd.DataFrame
  meta: SolverMeta


@dataclass(slots=True, frozen=True)
class BallisticContext:
  ammo: Ammo
  weapon: Weapon
  medium: Medium
  gravity_mps2: float
  wind_vel_mps: Vec3
  drag_cfg: DragConfig


def _unit_vec_from_elevation_deg(elev_deg: float) -> Vec3:
  # Оси:
  # +X: вперёд (дальность)
  # +Y: вправо (дрейф)
  # +Z: вверх (высота)
  a = np.deg2rad(float(elev_deg))
  return np.array([np.cos(a), 0.0, np.sin(a)], dtype=np.float64)


def _derivatives_external(state: ProjectileState, ctx: BallisticContext) -> tuple[Vec3, Vec3]:
  # dpos/dt = vel
  dpos = state.vel_mps

  # Внешние силы:
  # 1) гравитация
  a_grav = np.array([0.0, 0.0, -ctx.gravity_mps2], dtype=np.float64)

  # 2) сопротивление воздуха (против скорости относительно воздуха)
  v_rel = state.vel_mps - ctx.wind_vel_mps
  a_drag_xyz = drag_accel(
    v_xyz_mps=(float(v_rel[0]), float(v_rel[1]), float(v_rel[2])),
    projectile=ctx.ammo.projectile,
    rho_kg_m3=ctx.medium.density_kg_m3,
    cfg=ctx.drag_cfg,
  )
  a_drag = np.array(a_drag_xyz, dtype=np.float64)

  dvel = a_grav + a_drag
  return dpos, dvel


def _rk4_step(
  state: ProjectileState,
  ctx: BallisticContext,
  dt: float,
  deriv_fn: Callable[[ProjectileState, BallisticContext], tuple[Vec3, Vec3]],
) -> ProjectileState:
  t0 = state.t_s
  p0 = state.pos_m
  v0 = state.vel_mps

  dp1, dv1 = deriv_fn(state, ctx)

  s2 = ProjectileState(
    t_s=t0 + 0.5 * dt,
    pos_m=p0 + 0.5 * dt * dp1,
    vel_mps=v0 + 0.5 * dt * dv1,
  )
  dp2, dv2 = deriv_fn(s2, ctx)

  s3 = ProjectileState(
    t_s=t0 + 0.5 * dt,
    pos_m=p0 + 0.5 * dt * dp2,
    vel_mps=v0 + 0.5 * dt * dv2,
  )
  dp3, dv3 = deriv_fn(s3, ctx)

  s4 = ProjectileState(
    t_s=t0 + dt,
    pos_m=p0 + dt * dp3,
    vel_mps=v0 + dt * dv3,
  )
  dp4, dv4 = deriv_fn(s4, ctx)

  p1 = p0 + (dt / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
  v1 = v0 + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)

  return ProjectileState(t_s=t0 + dt, pos_m=p1, vel_mps=v1)


def _stop_reason(state: ProjectileState, scenario: Scenario, ctx: BallisticContext) -> Optional[str]:
  x = float(state.pos_m[0])
  y = float(state.pos_m[1])
  z = float(state.pos_m[2])

  if state.t_s >= 0.999999 * 1e12:
    return 'invalid_time'

  if x >= scenario.limits.max_distance_m:
    return 'max_distance'

  drop_m = -z  # z0 = 0, вниз = положительный drop
  if drop_m >= scenario.limits.max_drop_m:
    return 'max_drop'

  if z >= scenario.limits.max_height_m:
    return 'max_height'

  if abs(y) >= scenario.limits.max_drift_m:
    return 'max_drift'

  v = state.speed_mps
  e = 0.5 * ctx.ammo.projectile.mass_kg * v * v
  if e < scenario.limits.min_energy_j:
    return 'min_energy'

  if state.t_s >= scenario.limits.max_distance_m / max(ctx.ammo.v0_mps, 1e-9) + 10.0:
    return 'time_guard'

  return None


def solve_scenario_external_ballistics_only(
  scenario: Scenario,
  weapon: Weapon,
  ammo: Ammo,
  medium: Medium,
  *,
  gravity_mps2: float,
  config: SolverConfig,
) -> SolverResult:
  dt = float(config.dt_s)
  if dt <= 0.0:
    raise ValueError(f"dt_s must be > 0, got: {dt}")

  wind = scenario.environment.wind
  # направление: 0 deg = вдоль +x (вперёд), 90 deg = +y (вправо)
  wd = np.deg2rad(float(wind.direction_deg))
  wind_vel = np.array(
    [np.cos(wd) * wind.speed_mps, np.sin(wd) * wind.speed_mps, 0.0],
    dtype=np.float64,
  )

  drag_cfg = DragConfig(enabled=bool(config.drag_enabled))

  ctx = BallisticContext(
    ammo=ammo,
    weapon=weapon,
    medium=medium,
    gravity_mps2=float(gravity_mps2),
    wind_vel_mps=wind_vel,
    drag_cfg=drag_cfg,
  )

  dir_vec = _unit_vec_from_elevation_deg(scenario.shot.elevation_angle_deg)
  v0 = ammo.v0_mps * dir_vec

  state = ProjectileState(
    t_s=0.0,
    pos_m=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    vel_mps=v0,
  )

  rows: list[dict[str, Any]] = []
  events_rows: list[dict[str, Any]] = []

  stop = None
  n = 0
  deriv_fn = _derivatives_external

  while state.t_s <= config.t_max_s:
    v_rel = state.vel_mps - ctx.wind_vel_mps
    rel_speed = float(np.linalg.norm(v_rel))
    mach = rel_speed / max(ctx.medium.speed_of_sound_m_s, 1e-9)
    z = float(state.pos_m[2])

    rows.append({
      't_s': state.t_s,
      'x_m': float(state.pos_m[0]),
      'y_m': float(state.pos_m[1]),
      'z_m': z,
      'drop_m': -z,
      'vx_mps': float(state.vel_mps[0]),
      'vy_mps': float(state.vel_mps[1]),
      'vz_mps': float(state.vel_mps[2]),
      'speed_mps': state.speed_mps,
      'rel_speed_mps': rel_speed,
      'mach': mach,
      'energy_j': 0.5 * ammo.projectile.mass_kg * state.speed_mps * state.speed_mps,
    })

    stop = _stop_reason(state, scenario, ctx)
    if stop is not None:
      break

    state = _rk4_step(state, ctx, dt, deriv_fn)
    n += 1

  if stop is None:
    stop = 't_max'

  traj = pd.DataFrame(rows)
  ev = pd.DataFrame(events_rows)

  meta = SolverMeta(
    scenario_id=scenario.id,
    weapon_id=weapon.id,
    ammo_id=ammo.id,
    environment_medium_id=medium.id,
    dt_s=dt,
    t_max_s=float(config.t_max_s),
    integrator=config.integrator,
    stop_reason=stop,
    n_steps=n,
  )

  return SolverResult(trajectory=traj, events=ev, meta=meta)
