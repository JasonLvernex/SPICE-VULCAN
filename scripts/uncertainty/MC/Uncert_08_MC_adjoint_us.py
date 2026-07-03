#!/usr/bin/env python3
"""
Uncert_08_MC_adjoint_us.py — MC uncertainty from undersampled adjoint recons.

Collects the adjoint recon outputs produced by recon_04_adjoint_us.py and
computes two types of uncertainty:

  1. Prefitting uncertainty (image-domain)
        std of the xcorr-aligned adjoint recon across all MC subjects,
        per voxel per frequency point.  No fitting required.

  2. Concentration uncertainty  (optional, --fit-basis-dir)
        fsl_mrsi is run on each MC sample's adj_recon_aligned.nii.gz;
        std of the fitted concentrations across subjects.

Reads (from <adj-us-dir>/<mc_name>/adjoint_test/):
    adj_recon_aligned.nii.gz   xcorr-aligned adjoint recon (recon_04 output)

Writes (to <adj-us-dir>/uncertainty/):
    prefitting_std.nii.gz      std image (FID domain, NIfTI-MRS)
    prefitting_mean.nii.gz     mean image
    prefitting_std_map.npy     mean(|std spectrum|) map  (Ny, Nx)
    fig_prefitting_std.png     spatial std magnitude map
    fig_prefitting_spectrum.png mean ± std spectrum at a brain voxel
    [with --fit-basis-dir]:
    conc_std.npy               (Ny, Nx, n_metab)
    conc_mean.npy              (Ny, Nx, n_metab)
    metab_names.npy
    fig_conc_std_<metab>.png

Usage:
    python scripts/uncertainty/MC/Uncert_08_MC_adjoint_us.py \
        --adj-us-dir output/invivo_260623_01_us25/adjoint_recon_us \
        --data-dir   data/processed/invivo_260623_01_us25 \
        --dim 64 64 --n-seq-points 300 \
        --brain-threshold 0.16 \
        --plot-voxel 32 32

    # with concentration uncertainty
    python scripts/uncertainty/MC/Uncert_08_MC_adjoint_us.py \
        --adj-us-dir    output/invivo_260623_01_us25/adjoint_recon_us \
        --data-dir      data/processed/invivo_260623_01_us25 \
        --fit-basis-dir ./basis/ \
        --ppmlim 0.0 7.5 \
        --combine NAA NAAG --combine PCh GPC --combine Cr PCr \
        --plot-metabs NAA Cr Ins Glu PCh
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from warnings import filterwarnings
filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from tqdm import tqdm

from fsl_mrs.utils.misc import FIDToSpec, SpecToFID
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs, NIFTI_MRS

_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, str(_root))
from utils.scan_params import load_scan_params

D_TYPE = np.complex64


# ── fsl_mrsi helpers (reused from Uncert_05) ─────────────────────────────────

def _run_fsl_mrsi(data_file, basis_path, mask_file, ppmlim, out_file,
                   baseline, combine_groups,
                   conj_basis=True, no_conj_fid=True):
    cmd = [
        "fsl_mrsi",
        "--data",     str(data_file),
        "--basis",    str(basis_path),
        "--mask",     str(mask_file),
        "--baseline", baseline,
        "--ppmlim",   str(ppmlim[0]), str(ppmlim[1]),
        "--overwrite",
        "--output",   str(out_file),
        "--no_rescale",
    ]
    if conj_basis:
        cmd.append("--conj_basis")
    if no_conj_fid:
        cmd.append("--no_conj_fid")
    for group in combine_groups:
        cmd += ["--combine"] + list(group)
    result = subprocess.run(cmd, env=os.environ.copy(),
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        err = subprocess.CalledProcessError(result.returncode, cmd)
        err.stderr = result.stderr
        raise err


def _load_conc_maps(fit_dir, metab_names_ref=None):
    raw_dir = Path(fit_dir) / "concs" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Concentration folder not found: {raw_dir}")
    conc_maps = {}
    names = []
    for f in sorted(raw_dir.glob("*.nii*")):
        name = f.name
        for ext in (".nii.gz", ".nii"):
            if name.endswith(ext):
                name = name[:-len(ext)]
                break
        data = np.squeeze(nib.load(str(f)).get_fdata())
        if data.ndim == 3 and data.shape[-1] == 1:
            data = data[:, :, 0]
        conc_maps[name] = data.T   # (Nx, Ny) → (Ny, Nx)
        names.append(name)
    if metab_names_ref is not None:
        names = metab_names_ref
    return conc_maps, names


# ── plotting ──────────────────────────────────────────────────────────────────

def _overlay_map(data_2d, wref_norm, brain_mask, title, cmap, label, out_path, vmax=None):
    masked = np.where(brain_mask, data_2d, np.nan)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(wref_norm, origin="lower", cmap="gray", alpha=0.6, zorder=0)
    im = ax.imshow(masked, origin="lower", vmin=0, vmax=vmax, cmap=cmap, alpha=0.9, zorder=1)
    ax.contour(brain_mask, levels=[0.5], colors="white", linewidths=0.7, zorder=2)
    ax.set_title(title, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2e}"))
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="black")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MC uncertainty from undersampled adjoint recons",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adj-us-dir",    required=True,
                   help="adjoint_recon_us/ folder produced by recon_04_adjoint_us.py")
    p.add_argument("--data-dir",      required=True,
                   help="Processed-data dir for scan params and wref "
                        "(e.g. data/processed/invivo_260623_01_us25)")
    p.add_argument("--mc-dirs",       nargs="+", default=None,
                   help="Override which MC subdirs to use "
                        "(default: all non-'uncertainty' subdirs in --adj-us-dir)")

    # acquisition
    p.add_argument("--dim",           type=int, nargs=2, default=[64, 64])
    p.add_argument("--n-seq-points",  type=int, default=300)
    p.add_argument("--ppm-center",    type=float, default=3.027)
    p.add_argument("--dwelltime",     type=float, default=None)
    p.add_argument("--k-points",      type=int,   default=None)
    p.add_argument("--center-freq",   type=float, default=None)
    p.add_argument("--brain-threshold", type=float, default=0.16)
    p.add_argument("--brain-erosion",   type=int,   default=3)

    # plot
    p.add_argument("--plot-voxel",    type=int, nargs=2, default=[32, 32],
                   metavar=("NY", "NX"))

    # concentration uncertainty (optional)
    p.add_argument("--fit-basis-dir", default=None,
                   help="Fitting basis for fsl_mrsi; omit to skip concentration uncertainty")
    p.add_argument("--ppmlim",        type=float, nargs=2, default=[0.0, 7.5])
    p.add_argument("--baseline",      default="poly, 0")
    p.add_argument("--combine",       nargs="+", action="append", default=[],
                   metavar="METAB",
                   help="Metabolite combine groups e.g. --combine NAA NAAG")
    p.add_argument("--plot-metabs",   nargs="+",
                   default=["NAA", "Cr", "Ins", "Glu", "PCh"])
    p.add_argument("--cleanup",       action="store_true",
                   help="Delete temp per-sample fit dirs after collecting")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    adj_us_dir = Path(args.adj_us_dir)
    data_dir   = args.data_dir.rstrip("/") + "/"
    out_dir    = adj_us_dir / "uncertainty"
    out_dir.mkdir(exist_ok=True)

    load_scan_params(args, data_dir, k_key="k_mrsi")
    Ny, Nx  = args.dim
    N_SEQ   = args.n_seq_points
    TS      = (args.k_points / N_SEQ) * args.dwelltime
    sweepwidth = 1.0 / TS
    FREQ_AXIS  = np.linspace(-sweepwidth / 2, sweepwidth / 2, N_SEQ)
    PPM_AXIS   = FREQ_AXIS / args.center_freq + args.ppm_center

    # ── Discover MC subdirs ───────────────────────────────────────────────────
    if args.mc_dirs:
        mc_dirs = [Path(d) for d in args.mc_dirs]
    else:
        mc_dirs = sorted(
            d for d in adj_us_dir.iterdir()
            if d.is_dir() and d.name != "uncertainty"
        )
    print(f"[uncert-08] adj_us_dir:  {adj_us_dir}")
    print(f"[uncert-08] MC subjects: {[d.name for d in mc_dirs]}")
    print(f"[uncert-08] Out:         {out_dir}")

    # ── Brain mask ────────────────────────────────────────────────────────────
    wref_img  = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    wref_2d   = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask       = wref_norm > args.brain_threshold
    brain_mask_inner = binary_erosion(brain_mask, iterations=args.brain_erosion)

    # ── Reference affine ─────────────────────────────────────────────────────
    affine_path = data_dir + "affine.npy"
    affine = np.load(affine_path) if os.path.exists(affine_path) else np.eye(4)

    # ── Load all aligned adjoint recons ──────────────────────────────────────
    print("\n[uncert-08] Loading adjoint recon outputs …")
    mc_fids  = []
    mc_names = []
    for mc in mc_dirs:
        nii_path = mc / "adjoint_test" / "adj_recon_aligned.nii.gz"
        if not nii_path.exists():
            print(f"  [skip] {mc.name}: adj_recon_aligned.nii.gz not found")
            continue
        nii      = NIFTI_MRS(str(nii_path))
        fid_data = np.array(nii.image[:, :, 0, :]).astype(D_TYPE)  # (Nx, Ny, N_seq)
        mc_fids.append(fid_data)
        mc_names.append(mc.name)
        print(f"  loaded {mc.name}  shape={fid_data.shape}")

    if len(mc_fids) < 2:
        raise RuntimeError(f"Need at least 2 MC samples; found {len(mc_fids)}")

    N_mc     = len(mc_fids)
    mc_stack = np.stack(mc_fids, axis=0)   # (N_mc, Nx, Ny, N_seq)  FID

    # ── 1. Prefitting uncertainty ─────────────────────────────────────────────
    print(f"\n[uncert-08] Computing prefitting uncertainty  (N_mc={N_mc}) …")
    fid_mean = mc_stack.mean(axis=0)                              # (Nx, Ny, N_seq)
    fid_std  = mc_stack.std(axis=0)                               # (Nx, Ny, N_seq)

    # Spatial maps: mean |spectrum| magnitude
    spec_mean = FIDToSpec(fid_mean.reshape(-1, N_SEQ), axis=-1).reshape(Nx, Ny, N_SEQ)
    spec_std  = FIDToSpec(fid_std.reshape(-1, N_SEQ),  axis=-1).reshape(Nx, Ny, N_SEQ)
    std_map   = np.mean(np.abs(spec_std), axis=-1).T   # (Ny, Nx) — transpose to match display

    # Mean: save as NIfTI-MRS (complex FID, (Nx, Ny, 1, N_seq))
    gen_nifti_mrs(
        np.ascontiguousarray(fid_mean[:, :, np.newaxis, :]),
        dwelltime=TS, spec_freq=297.219, affine=affine,
    ).save(str(out_dir / "prefitting_mean.nii.gz"))
    print(f"[uncert-08] Saved prefitting_mean.nii.gz")

    # Std: real-valued → save 2D spatial map as regular float NIfTI
    np.save(str(out_dir / "prefitting_std_map.npy"), std_map)
    std_nii_data = np.ascontiguousarray(std_map.T[::-1, :]).astype(np.float32)
    nib.save(nib.Nifti1Image(std_nii_data[:, :, np.newaxis], affine),
             str(out_dir / "prefitting_std_map.nii.gz"))
    print(f"[uncert-08] Saved prefitting_std_map.npy / prefitting_std_map.nii.gz  shape={std_map.shape}")

    # ── Plot: spatial std map ─────────────────────────────────────────────────
    vmax_std = float(np.nanpercentile(std_map[brain_mask], 95))
    _overlay_map(
        std_map, wref_norm, brain_mask,
        title=f"Prefitting std (N_mc={N_mc})",
        cmap="Reds", label="Std |spectrum| (arb.)",
        out_path=str(out_dir / "fig_prefitting_std.png"),
        vmax=vmax_std,
    )
    print(f"[uncert-08] Saved fig_prefitting_std.png")

    # ── Plot: mean spectrum ± std at a brain voxel ────────────────────────────
    vy, vx = args.plot_voxel
    spec_mean_yx = FIDToSpec(fid_mean[vx, vy], axis=-1)  # note: fid stored as (Nx,Ny,T)
    spec_std_yx  = FIDToSpec(fid_std[vx, vy],  axis=-1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(PPM_AXIS, np.real(spec_mean_yx), color="steelblue", label="Mean")
    ax.fill_between(PPM_AXIS,
                    np.real(spec_mean_yx) - np.real(spec_std_yx),
                    np.real(spec_mean_yx) + np.real(spec_std_yx),
                    alpha=0.3, color="steelblue", label="±1σ")
    ax.set_xlabel("ppm")
    ax.set_ylabel("Re(spectrum)")
    ax.set_title(f"Prefitting uncertainty — voxel [{vy},{vx}]  (N_mc={N_mc})")
    ax.invert_xaxis()
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(out_dir / "fig_prefitting_spectrum.png"), dpi=150)
    plt.close(fig)
    print(f"[uncert-08] Saved fig_prefitting_spectrum.png")

    # ── 2. Concentration uncertainty (optional) ───────────────────────────────
    if not args.fit_basis_dir:
        print("\n[uncert-08] --fit-basis-dir not provided; skipping concentration uncertainty.")
        print(f"[uncert-08] Done. Results in {out_dir}")
        return

    print(f"\n[uncert-08] Running concentration uncertainty (fsl_mrsi on {N_mc} samples) …")
    fit_basis_dir  = args.fit_basis_dir
    combine_groups = args.combine if args.combine else []

    # Brain mask NIfTI for fsl_mrsi
    mask_nii = str(out_dir / "brain_mask.nii.gz")
    from fsl.data.image import Image
    Image((wref_2d * brain_mask).astype(np.float32).T).save(mask_nii)

    output_concs    = []
    metab_names_ref = None
    failed          = []

    for i, (mc, fid_i) in enumerate(tqdm(zip(mc_dirs, mc_fids), total=N_mc,
                                          desc="MC fitting")):
        nii_path = mc / "adjoint_test" / "adj_recon_aligned.nii.gz"
        if not nii_path.exists():
            failed.append(mc.name)
            continue

        tmp_fit = str(out_dir / f"_tmp_fit_{mc.name}.nii.gz")
        try:
            _run_fsl_mrsi(
                data_file      = str(nii_path),
                basis_path     = fit_basis_dir,
                mask_file      = mask_nii,
                ppmlim         = args.ppmlim,
                out_file       = tmp_fit,
                baseline       = args.baseline,
                combine_groups = combine_groups,
            )
            conc_maps, metab_names = _load_conc_maps(tmp_fit, metab_names_ref)
            if metab_names_ref is None:
                metab_names_ref = metab_names
                print(f"[uncert-08] Metabolites: {metab_names_ref}")

            nan_map  = np.full((Ny, Nx), np.nan, dtype=float)
            one_iter = np.stack(
                [conc_maps.get(m, nan_map) for m in metab_names_ref], axis=-1
            )   # (Ny, Nx, n_metab)
            output_concs.append(one_iter)

        except subprocess.CalledProcessError as e:
            tqdm.write(f"[MC {mc.name}] fsl_mrsi failed (exit {e.returncode})")
            if getattr(e, "stderr", None):
                tqdm.write(e.stderr[-3000:])
            failed.append(mc.name)
        except Exception as e:
            tqdm.write(f"[MC {mc.name}] failed: {repr(e)}")
            failed.append(mc.name)
        finally:
            if args.cleanup and os.path.isdir(tmp_fit):
                shutil.rmtree(tmp_fit)

    if not output_concs:
        print("[uncert-08] ERROR: all concentration fits failed.")
        return

    print(f"[uncert-08] Fitting done: {len(output_concs)}/{N_mc} succeeded"
          + (f"  failed: {failed}" if failed else ""))

    concs_arr = np.stack(output_concs, axis=0)          # (N_ok, Ny, Nx, n_metab)
    conc_std  = np.nanstd(concs_arr,  axis=0)           # (Ny, Nx, n_metab)
    conc_mean = np.nanmean(concs_arr, axis=0)

    np.save(str(out_dir / "conc_std.npy"),    conc_std)
    np.save(str(out_dir / "conc_mean.npy"),   conc_mean)
    np.save(str(out_dir / "metab_names.npy"), np.array(metab_names_ref))
    print(f"[uncert-08] Saved conc_std/conc_mean  shape={conc_std.shape}")

    # ── Plot concentration std/mean maps ──────────────────────────────────────
    combined_names = ["+".join(g) for g in combine_groups]
    plot_metabs = list(args.plot_metabs) + [n for n in combined_names
                                            if n not in args.plot_metabs]
    for meta in plot_metabs:
        if meta not in metab_names_ref:
            print(f"[warn] '{meta}' not in results, skipping plot.")
            continue
        idx = metab_names_ref.index(meta)
        std_sl  = conc_std[:, :, idx]
        mean_sl = conc_mean[:, :, idx]
        vmax_s  = float(np.nanpercentile(std_sl[brain_mask],  90)) if brain_mask.any() else None
        vmax_m  = float(np.nanpercentile(mean_sl[brain_mask], 90)) if brain_mask.any() else None
        _overlay_map(std_sl,  wref_norm, brain_mask, f"Conc std: {meta}",
                     "Reds",   "Std (arb.)",           str(out_dir / f"fig_conc_std_{meta}.png"),  vmax=vmax_s)
        _overlay_map(mean_sl, wref_norm, brain_mask, f"Conc mean: {meta}",
                     "inferno", "Mean (arb.)",          str(out_dir / f"fig_conc_mean_{meta}.png"), vmax=vmax_m)
        print(f"[uncert-08] Saved fig_conc_std_{meta}.png / fig_conc_mean_{meta}.png")

    print(f"\n[uncert-08] Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
