"""
generate_artifacts.py — Produce the real result artifacts used by
NGP_Mejora_Presentacion.ipynb from the cached Gaia DR3 disk-star catalog.

Run once (no network — reads data/gaia_disk_stars.csv):
    .venv/bin/python generate_artifacts.py

Writes:
    results/param_sweep_results.csv   — via param_sweep.run_all_sweeps
    results/bootstrap_results.json    — bootstrap of the flagship
                                         great-circle SVD estimator
                                         (n_samples=2000, seed=42)
"""

import json
import time

from gaia_fetcher import fetch_gaia_stars
from param_sweep import run_all_sweeps
from bootstrap import bootstrap_great_circle_pole, save_bootstrap_results
from ngp_3d import great_circle_pole, ngp_3d_pipeline


def main():
    print("Loading cached Gaia DR3 disk stars ...")
    data = fetch_gaia_stars()
    print(f"  {len(data)} stars loaded.")

    print("\n== Flagship: great_circle_pole (SVD on unit direction vectors) ==")
    gc = great_circle_pole(data)
    print(f"  alpha_NGP = {gc['alpha_NGP']:.4f} h   delta_NGP = {gc['delta_NGP']:.4f} deg")

    print("\n== Contrast: ngp_3d_pipeline (3D RANSAC with 1/parallax distance) ==")
    import numpy as np
    ransac = ngp_3d_pipeline(data, rng=np.random.default_rng(42))
    print(f"  alpha_NGP = {ransac['alpha_NGP']:.4f} h   delta_NGP = {ransac['delta_NGP']:.4f} deg"
          f"   n_inliers={ransac['n_inliers']}")

    print("\n== param_sweep.run_all_sweeps ==")
    t0 = time.time()
    sweeps = run_all_sweeps(data)
    print(f"  wrote results/param_sweep_results.csv in {time.time()-t0:.1f}s "
          f"({sum(len(v) for v in sweeps.values())} total rows)")

    print("\n== bootstrap_great_circle_pole (n_samples=2000, seed=42) ==")
    t0 = time.time()
    boot = bootstrap_great_circle_pole(data, n_samples=2000, seed=42)
    save_bootstrap_results(boot, path="results/bootstrap_results.json")
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  alpha_NGP: mean={boot['alpha_mean']:.4f}h median={boot['alpha_median']:.4f}h "
          f"CI95={tuple(round(x, 4) for x in boot['alpha_ci95'])}")
    print(f"  delta_NGP: mean={boot['delta_mean']:.4f}deg median={boot['delta_median']:.4f}deg "
          f"CI95={tuple(round(x, 4) for x in boot['delta_ci95'])}")

    dlo, dhi = boot["delta_ci95"]
    iau_delta = 27.13
    contains_iau = dlo <= iau_delta <= dhi
    print(f"\n  CI95 for delta {'CONTAINS' if contains_iau else 'DOES NOT CONTAIN'} "
          f"the IAU reference delta={iau_delta} deg.")

    print("\nAll artifacts written to results/.")


if __name__ == "__main__":
    main()
