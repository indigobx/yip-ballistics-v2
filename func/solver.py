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
  lift_enabled: bool = True
  lift_cl_alpha: float = 2.0
  magnus_enabled: bool = True
  magnus_cl_scale: float = 0.5


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
  lift_enabled: bool
  lift_cl_alpha: float
  magnus_enabled: bool
  magnus_cl_scale: float


def _unit_vec_from_elevation_deg(elev_deg: float) -> Vec3:
  # Оси:
  # +X: вперёд (дальность)
  # +Y: вправо (дрейф)
  # +Z: вверх (высота)
  a = np.deg2rad(float(elev_deg))
  return np.array([np.cos(a), 0.0, np.sin(a)], dtype=np.float64)


def _derivatives_external(
  state: ProjectileState,
  ctx: BallisticContext,
) -> tuple[Vec3, Vec3, Vec3, float, Vec3]:
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
    speed_of_sound_m_s=ctx.medium.speed_of_sound_m_s,
  )
  a_drag = np.array(a_drag_xyz, dtype=np.float64)

  a_lift = np.zeros(3, dtype=np.float64)
  a_magnus = np.zeros(3, dtype=np.float64)

  rel_speed = float(np.linalg.norm(v_rel))
  if rel_speed > 0.0:
    v_rel_hat = v_rel / rel_speed
    axis = state.axis_unit
    axis_dot = float(np.dot(axis, v_rel_hat))
    axis_perp = axis - axis_dot * v_rel_hat
    axis_perp_norm = float(np.linalg.norm(axis_perp))

    area_m2 = ctx.ammo.projectile.area_m2
    mass_kg = ctx.ammo.projectile.mass_kg

    if ctx.lift_enabled and axis_perp_norm > 0.0:
      aoa = float(np.arctan2(axis_perp_norm, max(abs(axis_dot), 1e-9)))
      cl = ctx.lift_cl_alpha * aoa
      lift_dir = axis_perp / axis_perp_norm
      a_lift = 0.5 * ctx.medium.density_kg_m3 * cl * area_m2 * (rel_speed ** 2) / max(mass_kg, 1e-9) * lift_dir

    if ctx.magnus_enabled:
      omega = state.angvel_radps
      omega_mag = float(np.linalg.norm(omega))
      if omega_mag > 0.0:
        magnus_dir = np.cross(omega, v_rel)
        magnus_norm = float(np.linalg.norm(magnus_dir))
        if magnus_norm > 0.0:
          magnus_dir = magnus_dir / magnus_norm
          radius = 0.5 * ctx.ammo.projectile.diameter_m
          spin_ratio = omega_mag * radius / max(rel_speed, 1e-9)
          cl_magnus = ctx.magnus_cl_scale * spin_ratio
          a_magnus = 0.5 * ctx.medium.density_kg_m3 * cl_magnus * area_m2 * (rel_speed ** 2) / max(mass_kg, 1e-9) * magnus_dir

  dvel = a_grav + a_drag + a_lift + a_magnus

  axis = state.axis_unit
  omega = state.angvel_radps
  daxis = np.cross(omega, axis)
  droll = float(np.dot(omega, axis))
  dangvel = np.zeros(3, dtype=np.float64)

  return dpos, dvel, daxis, droll, dangvel


def _rk4_step(
  state: ProjectileState,
  ctx: BallisticContext,
  dt: float,
  deriv_fn: Callable[[ProjectileState, BallisticContext], tuple[Vec3, Vec3, Vec3, float, Vec3]],
) -> ProjectileState:
  t0 = state.t_s
  p0 = state.pos_m
  v0 = state.vel_mps
  a0 = state.axis_unit
  r0 = state.roll_rad
  w0 = state.angvel_radps

  dp1, dv1, da1, dr1, dw1 = deriv_fn(state, ctx)

  s2 = ProjectileState(
    t_s=t0 + 0.5 * dt,
    pos_m=p0 + 0.5 * dt * dp1,
    vel_mps=v0 + 0.5 * dt * dv1,
    axis_unit=a0 + 0.5 * dt * da1,
    roll_rad=r0 + 0.5 * dt * dr1,
    angvel_radps=w0 + 0.5 * dt * dw1,
  )
  dp2, dv2, da2, dr2, dw2 = deriv_fn(s2, ctx)

  s3 = ProjectileState(
    t_s=t0 + 0.5 * dt,
    pos_m=p0 + 0.5 * dt * dp2,
    vel_mps=v0 + 0.5 * dt * dv2,
    axis_unit=a0 + 0.5 * dt * da2,
    roll_rad=r0 + 0.5 * dt * dr2,
    angvel_radps=w0 + 0.5 * dt * dw2,
  )
  dp3, dv3, da3, dr3, dw3 = deriv_fn(s3, ctx)

  s4 = ProjectileState(
    t_s=t0 + dt,
    pos_m=p0 + dt * dp3,
    vel_mps=v0 + dt * dv3,
    axis_unit=a0 + dt * da3,
    roll_rad=r0 + dt * dr3,
    angvel_radps=w0 + dt * dw3,
  )
  dp4, dv4, da4, dr4, dw4 = deriv_fn(s4, ctx)

  p1 = p0 + (dt / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
  v1 = v0 + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
  a1 = a0 + (dt / 6.0) * (da1 + 2.0 * da2 + 2.0 * da3 + da4)
  r1 = r0 + (dt / 6.0) * (dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4)
  w1 = w0 + (dt / 6.0) * (dw1 + 2.0 * dw2 + 2.0 * dw3 + dw4)

  a_norm = float(np.linalg.norm(a1))
  if a_norm > 0.0:
    a1 = a1 / a_norm
  else:
    a1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)

  return ProjectileState(
    t_s=t0 + dt,
    pos_m=p1,
    vel_mps=v1,
    axis_unit=a1,
    roll_rad=r1,
    angvel_radps=w1,
  )


def _angles_from_axis(axis: Vec3, roll_rad: float) -> tuple[float, float, float]:
  xy = float(np.hypot(axis[0], axis[1]))
  yaw = float(np.arctan2(axis[1], axis[0]))
  pitch = float(np.arctan2(axis[2], max(xy, 1e-9)))
  roll = float(roll_rad)
  return roll, pitch, yaw


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
    lift_enabled=bool(config.lift_enabled),
    lift_cl_alpha=float(config.lift_cl_alpha),
    magnus_enabled=bool(config.magnus_enabled),
    magnus_cl_scale=float(config.magnus_cl_scale),
  )

  dir_vec = _unit_vec_from_elevation_deg(scenario.shot.elevation_angle_deg)
  v0 = ammo.v0_mps * dir_vec

  twist_m_per_turn = weapon.barrel.twist_m_per_turn
  spin_rps = ammo.v0_mps / max(twist_m_per_turn, 1e-9)
  spin_radps = 2.0 * np.pi * spin_rps
  spin_sign = -1.0 if weapon.barrel.twist_clockwise else 1.0
  angvel = dir_vec * (spin_radps * spin_sign)

  state = ProjectileState(
    t_s=0.0,
    pos_m=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    vel_mps=v0,
    axis_unit=dir_vec,
    roll_rad=0.0,
    angvel_radps=angvel,
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
    if rel_speed > 0.0:
      v_rel_hat = v_rel / rel_speed
      aoa = float(np.arccos(np.clip(float(np.dot(state.axis_unit, v_rel_hat)), -1.0, 1.0)))
    else:
      aoa = 0.0

    angle_x, angle_y, angle_z = _angles_from_axis(state.axis_unit, state.roll_rad)

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
      'angle_x': angle_x,
      'angle_y': angle_y,
      'angle_z': angle_z,
      'angvel_x': float(state.angvel_radps[0]),
      'angvel_y': float(state.angvel_radps[1]),
      'angvel_z': float(state.angvel_radps[2]),
      'aoa': aoa,
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
