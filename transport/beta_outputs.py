"""Summaries and downloads from the same immutable beta sample as the plots."""
import csv
import io

import numpy as np

from .beta_spectrum import BETA_MODEL_VERSION, EQUILIBRIUM, MIRD, NUCLIDES, spectrum
from .beta_transport import RESIDUAL_LOW, STATUS
from .reference import CUSTOM_ESTAR_URL, ESTAR_URL


def metadata(run):
    slab, src = run.slab, run.emissions
    return {
        "model_version": BETA_MODEL_VERSION, "equilibrium": EQUILIBRIUM,
        "seed_PCG64": src.seed, "total_emissions": src.count,
        "material": slab.material.name, "material_key": slab.material.key,
        "density_g_cm3": slab.density, "density_factor": slab.density_scale,
        "thickness_cm": slab.thickness, "mass_thickness_g_cm2": slab.thickness*slab.density,
        "estar_source": ESTAR_URL if slab.material.estar_id else CUSTOM_ESTAR_URL,
        "estar_material": slab.material.estar_id or "custom: C57H104O6",
        "spectrum_reference": "ICRP 107 / MIRDspecs 20250101",
    }


def summary_rows(run):
    rows = []
    src = run.emissions
    for j, name in enumerate(NUCLIDES):
        spec = spectrum(name)
        mask = src.isotope == j
        lengths = run.path_cm[mask & np.isfinite(run.path_cm)]
        energies = src.energy[mask]
        losses = run.loss[mask & np.isfinite(run.loss)]
        rows.append({
            **metadata(run), "nuclide": name, "spectrum_source": MIRD.format(name),
            "emission_weight": .5, "n_emitted": len(energies),
            "sample_mean_initial_MeV": float(energies.mean()) if len(energies) else None,
            "spectrum_mean_MeV": spec.mean, "beta_branch_reference_mean_MeV": spec.reference_mean,
            "endpoint_MeV": float(spec.energy[-1]),
            "n_range_known": len(lengths), "n_initial_below_table": len(energies)-len(lengths),
            "n_path_le_thickness": int(np.count_nonzero(lengths <= run.slab.thickness)),
            "n_path_gt_thickness": int(np.count_nonzero(lengths > run.slab.thickness)),
            "median_path_cm": float(np.median(lengths)) if len(lengths) else None,
            "n_residual_below_table": int(np.count_nonzero(mask & (run.status == RESIDUAL_LOW))),
            "n_exact_loss": len(losses), "mean_loss_MeV_known_only": float(losses.mean()) if len(losses) else None,
        })
    return rows


def _value(v):
    return float(v) if np.isfinite(v) else None


def particle_rows(run, selection):
    src = run.emissions
    common = metadata(run)
    for i in range(*selection.indices(src.count)):
        name = NUCLIDES[src.isotope[i]]
        yield {
            "particle_id": i + 1, "nuclide": name, "initial_energy_MeV": float(src.energy[i]),
            "collision_MeV_cm2_g": _value(run.collision[i]), "radiative_MeV_cm2_g": _value(run.radiative[i]),
            "total_MeV_cm2_g": _value(run.total[i]), "csda_range_g_cm2": _value(run.mass_range[i]),
            "csda_path_cm": _value(run.path_cm[i]), "path_end_energy_MeV": _value(run.end_energy[i]),
            "electron_energy_loss_MeV": _value(run.loss[i]), "transport_status": STATUS[run.status[i]],
            **common, "spectrum_source": MIRD.format(name),
        }


def particle_csv(run, selection):
    """Stream rows into bytes on click, without a second particle dataframe."""
    with io.BytesIO() as buf, io.TextIOWrapper(buf, encoding="utf-8-sig", newline="") as target:
        rows = iter(particle_rows(run, selection))
        first = next(rows)
        writer = csv.DictWriter(target, fieldnames=first.keys())
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)
        target.flush()
        return buf.getvalue()
