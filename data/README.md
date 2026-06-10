# Data Directory

This directory contains the raw input data required to run the pipeline.
The data files are not included in version control due to file size.

## Expected Files

| File | Description | Required |
|------|-------------|----------|
| `mrsi_data.npy` | Raw MRSI k-space data `(1, K, N_seq, N_coils)` complex64 | Yes |
| `mrsi_ksp.npy` | MRSI k-space trajectory `(3, K, N_seq)` — kx, ky, kz | Yes |
| `wref_data.npy` | Water-reference k-space data `(1, K, N_seq, N_coils)` complex64 | Yes |
| `wref_ksp.npy` | Water-reference trajectory `(3, K, N_seq)` | Yes |
| `wref_o.npy` | Water-reference magnitude image `(Ny, Nx, N_seq)` — used for brain mask | Yes |
| `sigma_noise.npy` | Noise standard deviation scalar (float) | Yes (steps 09–12) |
| `ecalib.npy` | Pre-computed ESPIRiT coil map `(Ny, Nx, N_coils)` — only needed for `--method rni` | Optional |
| `ref_vox_spec.npy` | Reference voxel FID spectrum `(N_seq,)` — for cross-correlation frequency alignment | Optional |
| `meas_*.nii.gz` | Structural reference NIfTI for spatial co-registration | Optional |

## Usage

Point the pipeline scripts to this directory via `--data-dir ./data/`:

```bash
python scripts/01_coil_correction.py --data-dir ./data/ --out-dir ./output/
```
