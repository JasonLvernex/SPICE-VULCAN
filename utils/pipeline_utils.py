"""Shared pipeline utilities."""

import os
import numpy as np
from scipy.ndimage import binary_erosion


def make_brain_mask(wref_img: np.ndarray, threshold: float, erosion: int = 3):
    """Brain mask from normalised absolute wref_o.

    Returns (wref_norm, brain_mask, brain_mask_inner).
    """
    wref_2d = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask = wref_norm > threshold
    brain_mask_inner = binary_erosion(brain_mask, iterations=erosion)
    return wref_norm, brain_mask, brain_mask_inner


def try_symlink_shared_output(data_dir, subject_out_dir, subdir):
    """If wref_o.npy in data_dir is a symlink, the wref is shared from a primary
    subject whose coilmap/b0map/... results are identical for all subjects.
    In that case, create subject_out_dir/<subdir> as a relative symlink pointing
    to the primary subject's equivalent directory instead of recomputing.

    Returns True (caller should return early) if symlinked successfully.
    Returns False if wref is not a symlink, or primary output not found yet.
    """
    wref_o_path = os.path.join(data_dir, "wref_o.npy")
    if not os.path.islink(wref_o_path):
        return False

    primary_data_dir = os.path.dirname(os.path.realpath(wref_o_path))
    primary_subj_id  = os.path.basename(primary_data_dir)

    out_root       = os.path.dirname(os.path.abspath(subject_out_dir))
    primary_subdir = os.path.join(out_root, primary_subj_id, subdir)

    if not os.path.isdir(primary_subdir):
        print(f"[pipeline] wref files are symlinked from primary subject {primary_subj_id!r}.")
        print(f"[pipeline] Primary {subdir}/ not found at: {primary_subdir}")
        print(f"[pipeline] Run this script on the primary subject first, then re-run here.")
        return False

    os.makedirs(subject_out_dir, exist_ok=True)
    link_path = os.path.join(subject_out_dir, subdir)

    if os.path.lexists(link_path):
        if os.path.islink(link_path):
            os.remove(link_path)
        else:
            print(f"[pipeline] {link_path} exists and is not a symlink — skipping auto-symlink.")
            return False

    rel = os.path.relpath(primary_subdir, subject_out_dir)
    os.symlink(rel, link_path)
    print(f"[pipeline] wref files are symlinked from primary subject {primary_subj_id!r}.")
    print(f"[pipeline] Symlinked {subdir}/  →  {rel}  (skipping recompute).")
    return True
