"""Autonomous Optical Celestial Navigation & Star Tracker Demo.

Executes an end-to-end Lost-in-Space attitude determination pipeline:
1. Synthetic frame generation with PSF and sensor noise.
2. OpenCV-based star spot detection and subpixel centroiding.
3. Lost-in-Space pattern matching using K-vector invariants.
4. Wahba's problem solution via Kabsch / SVD.
5. Error analysis in arcseconds and visual telemetry output.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from src.catalog.catalog_builder import build_default_catalog
from src.catalog.k_vector import StarPairDatabase
from src.pipeline.star_tracker import StarTracker
from src.simulation.camera_simulator import CameraParameters, CameraSimulator


def run_single_demo(
    target_constellation: str = "orion",
    save_output: bool = True,
    output_dir: str = "output",
) -> None:
    print("=" * 70)
    print("      AOCS / GNC AUTONOMOUS OPTICAL STAR TRACKER DEMO")
    print("=" * 70)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Catalog & K-Vector Database
    catalog_path = Path("data/catalogs/nav_stars_catalog.npz")
    print(f"[1/5] Loading Navigation Star Catalog from '{catalog_path}'...")
    catalog = build_default_catalog(catalog_path, total_stars=1600, max_magnitude=5.5)
    print(f"      Total stars in catalog: {len(catalog)} (limiting mag: +5.5)")

    # 2. Camera Sensor Configuration (1920x1080, 20 deg FOV)
    cam_params = CameraParameters(
        width=1920,
        height=1080,
        fov_x_deg=20.0,
        psf_sigma=1.25,
        noise_sigma=2.5,
        background_level=8.0,
    )
    print(f"[2/5] Camera Sensor Configuration:")
    print(f"      Resolution : {cam_params.width} x {cam_params.height} px")
    print(f"      Focal Length: {cam_params.focal_length:.2f} px")
    print(f"      H-FOV / V-FOV: {cam_params.fov_x_deg:.2f}° / {cam_params.fov_y_deg:.2f}°")
    print(f"      Diagonal FOV: {math.degrees(cam_params.diagonal_fov_rad):.2f}°")
    print(f"      Optics PSF  : Gaussian blur (sigma = {cam_params.psf_sigma} px)")

    # 3. Initialize Star Tracker & Invariant Database
    print(f"[3/5] Building Mortari K-Vector and Star Pair Database...")
    t_db0 = time.perf_counter()
    tracker = StarTracker(
        catalog=catalog,
        camera_params=cam_params,
        angular_tolerance_deg=0.15,
        min_inliers=4,
    )
    dt_db = (time.perf_counter() - t_db0) * 1000.0
    print(f"      Precomputed {tracker.pair_db.num_pairs} star pairs in {dt_db:.1f} ms")

    # 4. Generate Ground Truth Attitude & Synthetic Sensor Image
    sim = CameraSimulator(cam_params)

    # Select pointing direction
    if target_constellation == "orion":
        # Point camera toward Orion (Betelgeuse: HIP 27989)
        ref_star = catalog.get_star_by_id(27989)
        R_true = sim.attitude_from_boresight(ref_star.vector, roll_angle_rad=0.35)
        scene_name = "Orion Constellation (Betelgeuse boresight)"
    elif target_constellation == "ursa_major":
        # Point toward Big Dipper (Dubhe: HIP 54061)
        ref_star = catalog.get_star_by_id(54061)
        R_true = sim.attitude_from_boresight(ref_star.vector, roll_angle_rad=-0.4)
        scene_name = "Ursa Major (Dubhe boresight)"
    elif target_constellation == "cassiopeia":
        # Point toward Cassiopeia (Schedar: HIP 4427)
        ref_star = catalog.get_star_by_id(4427)
        R_true = sim.attitude_from_boresight(ref_star.vector, roll_angle_rad=0.7)
        scene_name = "Cassiopeia (Schedar boresight)"
    else:
        # Uniform random attitude
        R_true = sim.generate_random_attitude(seed=42)
        scene_name = "Random Attitude (SO(3) Uniform)"

    print(f"[4/5] Synthesizing Spaceborne Camera Frame...")
    print(f"      Scene Target : {scene_name}")

    proj_stars = sim.project_catalog(catalog, R_true)
    print(f"      Stars in FOV : {len(proj_stars)} stars")

    raw_frame = sim.render_image(proj_stars, add_noise=True, seed=101)
    clean_frame = sim.render_image(proj_stars, add_noise=False)

    # 5. Execute Star Tracker Pipeline
    print(f"[5/5] Executing Autonomous Attitude Determination Pipeline...")
    result = tracker.process_frame(raw_frame, R_true=R_true)

    # Print Results Telemetry
    print("\n" + "=" * 70)
    print("                      TELEMETRY REPORT")
    print("=" * 70)
    print(f"Tracker Status      : {result.status_message}")
    print(f"Processing Latency  : {result.execution_time_ms:.2f} ms ({1000.0/result.execution_time_ms:.1f} Hz)")
    print(f"Detected Star Spots : {len(result.detected_stars)}")
    print(f"Identified Stars    : {len(result.matched_stars)}")

    if result.success and result.solution is not None:
        sol = result.solution
        rot_true = Rotation.from_matrix(R_true)
        q_true = rot_true.as_quat()  # [x, y, z, w]
        q_true_wxyz = [q_true[3], q_true[0], q_true[1], q_true[2]]
        euler_true = rot_true.as_euler("xyz", degrees=True)

        print("\n--- Attitude Comparison ---")
        print(f"True Quaternion   [w,x,y,z]: [{q_true_wxyz[0]:.5f}, {q_true_wxyz[1]:.5f}, {q_true_wxyz[2]:.5f}, {q_true_wxyz[3]:.5f}]")
        print(f"Calc Quaternion   [w,x,y,z]: [{sol.quaternion[0]:.5f}, {sol.quaternion[1]:.5f}, {sol.quaternion[2]:.5f}, {sol.quaternion[3]:.5f}]")
        print(f"True Euler Angles [R,P,Y]  : [{euler_true[0]:.3f}°, {euler_true[1]:.3f}°, {euler_true[2]:.3f}°]")
        print(f"Calc Euler Angles [R,P,Y]  : [{sol.euler_angles_deg[0]:.3f}°, {sol.euler_angles_deg[1]:.3f}°, {sol.euler_angles_deg[2]:.3f}°]")

        print("\n--- Precision Metrics ---")
        print(f"TOTAL ATTITUDE ERROR       : {sol.error_angle_arcsec:.2f} arcsec  ({sol.error_angle_arcsec / 3600.0:.6f}°)")
        print(f"  - Boresight Pointing Error: {sol.boresight_error_arcsec:.2f} arcsec")
        print(f"  - Roll Axis (Clocking) Err: {sol.roll_error_arcsec:.2f} arcsec")
        print(f"Residual LOS Vector RMS    : {math.degrees(sol.residual_rms_rad)*3600.0:.2f} arcsec")

        print("\n--- Matched Stars Table ---")
        print(f"{'Det #':<6}{'Cat ID':<10}{'Name':<16}{'Vmag':<8}{'Subpix (u, v)':<22}{'Residual (\")':<12}")
        print("-" * 74)
        for m in result.matched_stars:
            det = result.detected_stars[m.detected_idx]
            name = m.catalog_star.name if m.catalog_star.name else f"CAT-{m.catalog_star_id}"
            res_arcsec = math.degrees(m.residual_angle_rad) * 3600.0
            print(
                f"{m.detected_idx:<6}{m.catalog_star_id:<10}{name:<16}{m.catalog_star.vmag:<8.2f}"
                f"({det.u:7.2f}, {det.v:7.2f})    {res_arcsec:<12.2f}"
            )
        print("-" * 74)

    # 6. Generate Visual Annotations and Matplotlib Dashboard
    annotated_bgr = tracker.annotate_image(raw_frame, result, proj_stars)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    if save_output:
        # Save full-resolution OpenCV annotated frame
        cv2.imwrite(str(out_path / "star_tracker_annotated.jpg"), annotated_bgr)

        # Create multi-panel Matplotlib figure
        fig = plt.figure(figsize=(18, 10), facecolor="#0e1117")
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0])

        # Panel 1: Full Annotated Star Field
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(annotated_rgb)
        ax1.set_title(
            f"Star Tracker Sensor Field ({scene_name})\n1920x1080 | FOV: 20° | {len(result.matched_stars)} Identifications",
            color="white",
            fontsize=12,
            pad=10,
        )
        ax1.axis("off")

        # Panel 2: Zoomed PSF Star Spot Centroid Analysis
        ax2 = fig.add_subplot(gs[0, 1])
        if result.detected_stars:
            brightest = result.detected_stars[0]
            ub, vb = int(round(brightest.u)), int(round(brightest.v))
            zw = 24
            y0, y1 = max(0, vb - zw), min(cam_params.height, vb + zw + 1)
            x0, x1 = max(0, ub - zw), min(cam_params.width, ub + zw + 1)
            crop = raw_frame[y0:y1, x0:x1]

            im2 = ax2.imshow(crop, cmap="inferno", extent=[x0, x1, y1, y0])
            ax2.plot(brightest.u, brightest.v, "c+", markersize=14, markeredgewidth=2, label=f"Centroid ({brightest.u:.2f}, {brightest.v:.2f})")
            ax2.set_title("Subpixel Star Spot PSF (Center of Gravity Moments)", color="white", fontsize=11)
            ax2.tick_params(colors="gray")
            ax2.legend(loc="upper right", facecolor="#1e222d", edgecolor="none", labelcolor="white")
            cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors="gray")
            cbar.set_label("Sensor DN", color="gray")

        # Panel 3: Reprojection Residuals & Error Statistics
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_facecolor("#151922")
        if result.matched_stars:
            star_labels = [
                m.catalog_star.name if m.catalog_star.name else str(m.catalog_star_id)
                for m in result.matched_stars
            ]
            residuals_arcsec = [math.degrees(m.residual_angle_rad) * 3600.0 for m in result.matched_stars]
            y_pos = np.arange(len(star_labels))

            bars = ax3.barh(y_pos, residuals_arcsec, color="#00d26a", edgecolor="#00a854", alpha=0.85)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(star_labels, color="white", fontsize=9)
            ax3.invert_yaxis()
            ax3.set_xlabel("Line of Sight Residual (arcsec)", color="white", fontsize=10)
            ax3.tick_params(colors="gray")
            ax3.grid(axis="x", color="#2a3040", linestyle="--")

            if result.solution and result.solution.error_angle_arcsec is not None:
                title_str = (
                    f"Attitude Error: {result.solution.error_angle_arcsec:.2f}\" | "
                    f"Boresight: {result.solution.boresight_error_arcsec:.2f}\" | "
                    f"Latency: {result.execution_time_ms:.1f} ms"
                )
                ax3.set_title(title_str, color="#00ffff", fontsize=11, weight="bold")

        plt.tight_layout()
        plot_file = out_path / "star_tracker_dashboard.png"
        fig.savefig(plot_file, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"\n[OK] Dashboard figure saved to: {plot_file}")
        print(f"[OK] Full-res annotated image saved to: {out_path / 'star_tracker_annotated.jpg'}")


def run_monte_carlo(num_trials: int = 15) -> None:
    """Run Monte-Carlo Lost-in-Space simulation across random attitudes."""
    print("=" * 70)
    print(f"   MONTE-CARLO SIMULATION: {num_trials} RANDOM ATTITUDES ACROSS SKY")
    print("=" * 70)

    catalog = build_default_catalog("data/catalogs/nav_stars_catalog.npz", total_stars=1600, max_magnitude=5.5)
    cam_params = CameraParameters(width=1920, height=1080, fov_x_deg=20.0)
    sim = CameraSimulator(cam_params)
    tracker = StarTracker(catalog=catalog, camera_params=cam_params)

    errors = []
    latencies = []
    matched_counts = []
    success_count = 0

    print(f"{'Trial':<8}{'Stars Vis':<12}{'Matched':<10}{'Error (\")':<14}{'Latency (ms)':<14}{'Status':<10}")
    print("-" * 68)

    for i in range(num_trials):
        # Pick random attitude
        R_true = sim.generate_random_attitude(seed=1000 + i)
        proj = sim.project_catalog(catalog, R_true)
        if len(proj) < 4:
            continue

        raw = sim.render_image(proj, add_noise=True, seed=2000 + i)
        res = tracker.process_frame(raw, R_true=R_true)

        if res.success and res.solution and res.solution.error_angle_arcsec is not None:
            success_count += 1
            err = res.solution.error_angle_arcsec
            errors.append(err)
            latencies.append(res.execution_time_ms)
            matched_counts.append(len(res.matched_stars))
            status = "LOCKED"
            err_str = f"{err:.2f}\""
        else:
            status = "FAILED"
            err_str = "N/A"

        print(f"{i+1:<8}{len(proj):<12}{len(res.matched_stars):<10}{err_str:<14}{res.execution_time_ms:<14.1f}{status:<10}")

    print("-" * 68)
    if errors:
        print(f"Success Rate       : {success_count}/{num_trials} ({100.0 * success_count / num_trials:.1f}%)")
        print(f"Mean Attitude Error: {np.mean(errors):.2f}\" (Median: {np.median(errors):.2f}\", Min: {np.min(errors):.2f}\", Max: {np.max(errors):.2f}\")")
        print(f"Mean Latency       : {np.mean(latencies):.1f} ms ({1000.0 / np.mean(latencies):.1f} Hz)")
        print(f"Mean Matched Stars : {np.mean(matched_counts):.1f} stars")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Optical Star Tracker Demo")
    parser.add_argument("--scene", type=str, default="orion", choices=["orion", "ursa_major", "cassiopeia", "random"], help="Sky scene to observe")
    parser.add_argument("--monte-carlo", type=int, default=0, help="Run N Monte-Carlo trials across random attitudes")
    args = parser.parse_args()

    if args.monte_carlo > 0:
        run_monte_carlo(args.monte_carlo)
    else:
        run_single_demo(target_constellation=args.scene)
