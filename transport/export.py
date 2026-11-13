"""Rows used by the app's in-memory CSV downloads (empty = not applicable)."""
import csv
import io

from . import MODEL_VERSION
from .reference import CUSTOM_ESTAR_URL, ESTAR_URL, RETRIEVED, XCOM_URL


def common(slab, kind, energy, source):
    mat = slab.material
    return {
        "model_version": MODEL_VERSION, "radiation": kind, "energy_keV": energy,
        "material": mat.name, "material_definition": mat.description,
        "composition_mass_fractions": " | ".join(f"{s} {w:g}" for s,w in mat.composition),
        "reference_table_density_g_cm3": mat.reference_density,
        "nominal_density_g_cm3": mat.density, "density_factor": slab.density_scale,
        "density_g_cm3": slab.density, "thickness_cm": slab.thickness,
        "mass_thickness_g_cm2": slab.density * slab.thickness,
        "reference": source, "reference_retrieved": RETRIEVED,
        "composition_source": mat.composition_url,
    }


def photon_row(slab, beam, sample):
    return {
        **common(slab, "gamma", beam.energy_kev, XCOM_URL),
        "mass_attenuation_with_coherent_cm2_g": beam.mass_mu,
        "linear_attenuation_cm_inv": beam.mu, "hvl_cm": beam.hvl,
        "transmission_theory_fraction": beam.transmission, "ln_theory_transmission": -beam.mu*slab.thickness,
        "incident_photons": sample.count, "rng_PCG64_seed": sample.seed,
        "removed_from_primary": sample.count - sample.transmitted,
        "transmitted_primaries": sample.transmitted,
        "transmission_mc_fraction": sample.fraction,
        "mc_standard_error_fraction": sample.standard_error,
    }


def electron_row(slab, result):
    source = ESTAR_URL if slab.material.estar_id else CUSTOM_ESTAR_URL
    return {
        **common(slab, "beta_monoenergetic_electron", result.energy_kev, source),
        "estar_material": slab.material.estar_id or "custom: C57H104O6",
        "mean_excitation_eV": slab.material.excitation_eV,
        "collision_MeV_cm2_g": result.collision, "radiative_MeV_cm2_g": result.radiative,
        "total_MeV_cm2_g": result.total, "csda_range_g_cm2": result.range_mass,
        "csda_path_cm": result.range_cm, "slab_to_csda_ratio": slab.thickness/result.range_cm,
        "final_energy_keV": result.end_kev, "energy_lost_keV": result.loss_kev,
        "csda_ends_in_slab": result.stopped,
    }


def csv_bytes(rows):
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def tracks_csv(run, selected):
    """Build the requested CSV only on download. Photon IDs are global and one-based."""
    sample = run.sample
    with io.BytesIO() as buf, io.TextIOWrapper(buf, encoding="utf-8-sig", newline="") as text:
        writer = csv.writer(text)
        writer.writerow(("photon_id", "energy_keV", "first_interaction_depth_cm",
                         "sampled_free_path_cm", "track_end_cm", "passed_hvl_in_sample",
                         "transmitted_exit", "material", "thickness_cm", "density_g_cm3",
                         "hvl_cm", "rng_PCG64_seed", "model_version"))
        for i in range(*selected.indices(sample.count)):
            distance = float(sample.distances[i])
            interacted = bool(sample.interacted[i])
            passed = "" if run.hvl_passed is None else int(run.hvl_passed[i])
            writer.writerow((i + 1, run.beam.energy_kev, distance if interacted else "",
                             distance, min(distance, run.slab.thickness), passed, int(not interacted),
                             run.slab.material.name, run.slab.thickness, run.slab.density,
                             run.beam.hvl, sample.seed, MODEL_VERSION))
        text.flush()
        return buf.getvalue()
