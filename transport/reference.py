"""Immutable, offline XCOM and ESTAR tables. Energies in these tables are MeV."""
import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np


DATA = Path(__file__).resolve().parents[1] / "data"
XCOM_URL = "https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html"
ESTAR_URL = "https://physics.nist.gov/PhysRefData/Star/Text/ESTAR-t.html"
CUSTOM_ESTAR_URL = "https://physics.nist.gov/PhysRefData/Star/Text/ESTAR-ut.html"


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    description: str
    density: float
    reference_density: float
    excitation_eV: float
    composition: tuple
    composition_url: str
    estar_id: str | None


with (DATA / "materials.json").open(encoding="utf-8") as f:
    _meta = json.load(f)
RETRIEVED = _meta["retrieved"]
MATERIALS = MappingProxyType({
    row["key"]: Material(**{**row, "composition": tuple(row["composition"].items())})
    for row in _meta["materials"]
})


def bounded(name, value, low, high):
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} は {low:g}〜{high:g} の有限値で指定してください。")


def _rows(file, material):
    if material not in MATERIALS:
        raise ValueError("物質を参照表から選択してください。")
    with (DATA / file).open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r["material"] == material]


def _columns(rows, names):
    cols = []
    for col in names:
        array = np.array([float(row[col]) for row in rows])
        array.setflags(write=False)
        cols.append(array)
    return tuple(cols)


def log_value(x, xs, ys):
    """Log-linear interpolation inside a positive, strictly increasing grid."""
    bounded("表の参照範囲", x, xs[0], xs[-1])
    right = int(np.searchsorted(xs, x))
    if xs[right] == x:
        return float(ys[right])
    left = right - 1
    f = np.log(x / xs[left]) / np.log(xs[right] / xs[left])
    return float(ys[left] * np.exp(f * np.log(ys[right] / ys[left])))


@lru_cache(maxsize=7)
def _photon_segments(material):
    segments, current = [], []
    for row in _rows("xcom.csv", material):
        if row["edge_side"] == "above":
            segments.append(_columns(current, ("energy_MeV", "mass_mu_cm2_g")))
            current = []
        current.append(row)
    segments.append(_columns(current, ("energy_MeV", "mass_mu_cm2_g")))
    return tuple(segments)


def mass_attenuation(material, energy_kev):
    bounded("光子エネルギー (keV)", energy_kev, 30, 2000)
    energy = energy_kev / 1000
    for grid, values in reversed(_photon_segments(material)):
        if energy >= grid[0]:
            return log_value(energy, grid, values)
    raise ValueError("XCOM の参照範囲外です。")


@dataclass(frozen=True)
class ElectronTable:
    energy: np.ndarray
    collision: np.ndarray
    radiative: np.ndarray
    total: np.ndarray
    range_mass: np.ndarray


@lru_cache(maxsize=7)
def electron_table(material):
    return ElectronTable(*_columns(_rows("estar.csv", material), (
        "energy_MeV", "collision_MeV_cm2_g", "radiative_MeV_cm2_g",
        "total_MeV_cm2_g", "csda_g_cm2")))
