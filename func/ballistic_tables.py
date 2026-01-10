# func/ballistic_tables.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import yaml


@dataclass(slots=True, frozen=True)
class DragTable:
  mach: np.ndarray
  cd: np.ndarray


def _as_float(x: Any, *, name: str) -> float:
  try:
    v = float(x)
  except Exception as e:
    raise TypeError(f"Field {name} must be a number") from e
  if not np.isfinite(v):
    raise ValueError(f"Field {name} must be finite, got: {v}")
  return float(v)


def _parse_table_rows(rows: Any, *, model_id: str) -> Tuple[np.ndarray, np.ndarray]:
  if not isinstance(rows, list) or len(rows) < 2:
    raise ValueError(f"Model {model_id}: table must be a list with at least 2 points")

  mach_list: list[float] = []
  cd_list: list[float] = []

  for i, item in enumerate(rows):
    if not isinstance(item, dict) or len(item) != 1:
      raise ValueError(
        f"Model {model_id}: each table row must be a single-pair mapping, e.g. - 0.95: 0.30"
      )

    (k, v) = next(iter(item.items()))
    m = _as_float(k, name=f"models[{model_id}].table[{i}].mach")
    cd = _as_float(v, name=f"models[{model_id}].table[{i}].cd")
    mach_list.append(m)
    cd_list.append(cd)

  mach = np.array(mach_list, dtype=np.float64)
  cd = np.array(cd_list, dtype=np.float64)

  # Если пользователь случайно перемешал строки — сортируем и предупреждаем логикой:
  order = np.argsort(mach)
  mach = mach[order]
  cd = cd[order]

  if not np.all(np.diff(mach) > 0):
    raise ValueError(f"Model {model_id}: mach values must be strictly increasing")

  return mach, cd


def load_ballistic_tables(path: str) -> Dict[str, DragTable]:
  with open(path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

  if not isinstance(data, dict):
    raise ValueError('Invalid ballistic table file: root must be a mapping')
  models = data.get('models')
  if not isinstance(models, list) or not models:
    raise ValueError('Invalid ballistic table file: missing "models" list')

  out: Dict[str, DragTable] = {}
  for m in models:
    if not isinstance(m, dict):
      raise ValueError('Invalid model entry: must be a mapping')

    mid = str(m.get('id', '')).strip()
    if not mid:
      raise ValueError('Model id must be non-empty')

    mach, cd = _parse_table_rows(m.get('table'), model_id=mid)
    out[mid] = DragTable(mach=mach, cd=cd)

  return out
