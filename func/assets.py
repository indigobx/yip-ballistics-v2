# func/assets.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from func.classes import Ammo, Medium, Weapon
from func.scenario import Scenario


@dataclass(slots=True, frozen=True)
class AssetPaths:
  root: Path

  @property
  def ammo_dir(self) -> Path:
    return self.root / 'ammo'

  @property
  def weapons_dir(self) -> Path:
    return self.root / 'weapons'

  @property
  def medium_dir(self) -> Path:
    return self.root / 'medium'

  @property
  def scenarios_dir(self) -> Path:
    return self.root / 'scenarios'


def load_yaml(path: Path) -> dict[str, Any]:
  with path.open('r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
  if not isinstance(data, dict):
    raise TypeError(f"YAML root must be a mapping, got: {type(data).__name__} in {path}")
  return data


def load_ammo(paths: AssetPaths, ammo_id: str) -> Ammo:
  p = paths.ammo_dir / f"{ammo_id}.yaml"
  d = load_yaml(p)
  return Ammo.from_dict(d)


def load_weapon(paths: AssetPaths, weapon_id: str) -> Weapon:
  p = paths.weapons_dir / f"{weapon_id}.yaml"
  d = load_yaml(p)
  return Weapon.from_dict(d)


def load_medium(paths: AssetPaths, medium_id: str) -> Medium:
  p = paths.medium_dir / f"{medium_id}.yaml"
  d = load_yaml(p)
  return Medium.from_dict(d)


def load_scenario(paths: AssetPaths, filename: str) -> Scenario:
  p = paths.scenarios_dir / filename
  d = load_yaml(p)
  return Scenario.from_dict(d)
