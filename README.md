# PCDiff: Physics-Conditioned Diffusion for Low-Dose CT Reconstruction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official implementation of:

> **PCDiff: A Physics-Conditioned Diffusion Model for Low-Dose CT Reconstruction with Sinogram-Domain Consistency**
> Md Imam Ahasan, Chengliang Wang, A F M Abdun Noor, Md Rakibul Islam, Kah Ong Michael Goh, S M Hasan Mahmud
> Submitted to *Computer Modeling in Engineering & Sciences (CMES)*, 2026.

---

## Overview

Low-dose CT (LDCT) reconstruction is a severely ill-posed inverse problem: reducing dose amplifies sinogram
noise, and most learned reconstruction methods never verify that their output, if re-projected, would
reproduce the sinogram it was actually derived from. PCDiff is a diffusion model that (1) conditions every
reverse-diffusion step on the adjoint backprojection of the measured sinogram, and (2) adds an explicit
sinogram-domain consistency loss during training, so that measurement consistency is enforced throughout
training rather than corrected for post-hoc at inference time.

On LoDoPaB-CT and a Mayo-AAPM 2016 sparse-view (180-view) benchmark, PCDiff attains the highest PSNR/SSIM and
lowest sinogram-domain residual among seven compared methods, with statistically significant gains over the
strongest diffusion baseline. See the manuscript for full results, ablations, and the DPS head-to-head
comparison.

---

## Key Contributions

- Physics-conditioned diffusion: persistent per-step conditioning on the adjoint backprojection `c = A*(y)`,
  from the initial noise sample through the final reconstruction (not just at initialization).
- An explicit sinogram-consistency loss `L_phys = ||A(x_hat0_phys) - y||^2`, jointly optimized with the
  standard diffusion denoising loss and a VGG-16 perceptual loss (`src/losses/`).
- End-to-end differentiable CT forward/adjoint operators, with a CUDA (`torch-radon`) production path and a
  pure-PyTorch CPU fallback verified against the same adjoint dot-product test (`src/ct/`).
- A deterministic DPM-Solver accelerated sampler that reuses the same trained checkpoint with no retraining
  (`PhysicsConditionedDDPM.dpm_solver_sample`), and pixel-wise uncertainty estimation from repeated
  conditional samples (`src/metrics/uncertainty.py`).
- A full experimental harness: seven baselines reimplemented in one shared codebase (FBP, RED-CNN, DU-GAN,
  Hformer, ASCON, CoreDiff, DPS), loss-term and capacity-matched ablations, paired significance testing, and
  dose/projection-count robustness sweeps (`scripts/`).

---

## Method Summary

### Conditioning mechanism

At every diffusion timestep `t`, the denoiser is conditioned on:
- the noisy latent `x_t`,
- the physics prior `c = A*(y)` (unfiltered adjoint backprojection of the measured sinogram), concatenated
  channel-wise with `x_t`,
- a sinusoidal timestep embedding for `t`.

Conditioning is persistent across the entire reverse trajectory, not applied only at initialization.

### Training objective

```
L = ||eps - eps_theta(x_t, c, t)||^2            (diffusion denoising loss)
  + lambda * ||A(x_hat0_phys) - y||^2           (physics-consistency loss, lambda = 1.0)
  + mu * sum_l ||phi_l(x_hat0) - phi_l(x0)||_1   (VGG-16 perceptual loss, mu = 0.1)
```

The network operates on images normalized to `[-1,1]`; `x_hat0` is affinely rescaled to the physical
attenuation units the sinogram lives in (`x_min`/`x_max`, per-dataset — see `src/ct/units.py`) before `A` is
applied in `L_phys`, so the residual is computed in commensurate units with `y`.

### Accelerated sampling

`PhysicsConditionedDDPM.sample` runs the full `T=250`-step ancestral DDPM reverse process.
`PhysicsConditionedDDPM.dpm_solver_sample` reuses the *same trained checkpoint* with a deterministic,
first-order DPM-Solver update over a strided `S`-step subsequence (`S=20` in the paper), with no retraining,
reducing inference from ~1.60 s/image to ~0.18 s/image.

### Uncertainty estimation

Drawing `K` independent reverse trajectories from the same conditioning `c` yields `K` reconstruction
samples; the empirical mean and standard deviation across samples are the point-estimate reconstruction and
pixel-wise uncertainty map, respectively (`src/metrics/uncertainty.py`: `mean_and_std`, `coverage_and_ece`,
`error_gap`, `pearson_r`).

---

## Repository Structure

```
configs/                    Base configs (LoDoPaB-CT, Mayo-AAPM 2016) and their
                             capacity-matched PCDiff-Small variants.
scripts/
  train.py                  Shared training entry point: proposed method + all baselines
                             (--model proposed|redcnn|hformer|ascon|dugan|corediff|dps_prior).
  evaluate.py                Shared evaluation entry point, incl. FBP (closed-form) and DPS
                             (--model fbp|dps|... ; --sampler ddpm|dpm_solver for `proposed`).
  sample.py                  Draws reconstruction samples (+ mean/std) for qualitative inspection.
  prepare_lodopab.sh         Download instructions for LoDoPaB-CT.
  prepare_mayo_aapm.py       Builds Mayo-AAPM 2016 sinograms (forward-projects in physical units,
                             then injects Poisson low-dose noise) and train/val/test splits.
  run_ablation.py            Loss-term ablation (full / w/o physics / w/o perceptual / w/o denoising).
  run_dose_sweep.py          Zero-shot dose- and projection-count robustness sweeps.
  run_significance_test.py   Paired Wilcoxon signed-rank + bootstrap CIs between two evaluate.py runs.
  count_params.py            Prints trainable parameter counts by instantiating each model directly.
src/
  ct/
    operators.py             Production CUDA (torch-radon) forward/adjoint operator.
    operators_torch.py       Pure-PyTorch CPU fallback (rotate-and-sum via grid_sample).
    fbp.py                   Ram-Lak (Hann-windowed) filtered backprojection -- the FBP baseline.
    poisson.py                Poisson low-dose noise injection in the sinogram domain.
    units.py                  [-1,1] <-> physical-attenuation-unit rescale (x_min/x_max).
  diffusion/
    ddpm.py                   PhysicsConditionedDDPM: q_sample, ancestral sampling, DPM-Solver sampling.
    dps.py                    UnconditionalDDPM + dps_sample: the DPS baseline's prior and inference-time
                               gradient correction.
    schedule.py                Beta/alpha_bar/posterior-variance schedule construction.
  losses/
    physics.py                 Sinogram-consistency loss (with the physical-unit rescale above).
    perceptual.py               VGG-16 perceptual loss.
  metrics/
    image.py, sinogram.py, uncertainty.py, stats.py   PSNR/SSIM, sinogram residual (incl. the paper's
    relative residual), uncertainty calibration (incl. Error Gap), Wilcoxon + bootstrap.
  models/
    unet.py                    Residual, self-attention U-Net denoiser (Ho et al. / Nichol & Dhariwal).
    baselines/                  RED-CNN, DU-GAN, Hformer, ASCON, CoreDiff reimplementations.
  data/                         LoDoPaB-CT and Mayo-AAPM 2016 Dataset classes, augmentation.
tests/                          Unit tests for every module above.
```

---

## Usage

```bash
pip install -r requirements.txt   # add torch-radon separately for the CUDA operator, or use
                                   # src/ct/operators_torch.py's CPU fallback for development/testing

# Train the proposed method
python scripts/train.py --config configs/lodopab.yaml --model proposed

# Train a baseline (same conditioning/data pipeline, for an apples-to-apples comparison)
python scripts/train.py --config configs/lodopab.yaml --model corediff

# Train the DPS prior (unconditional; physics reintroduced only at eval time)
python scripts/train.py --config configs/lodopab.yaml --model dps_prior

# Evaluate: proposed method, full T=250 ancestral sampling
python scripts/evaluate.py --config configs/lodopab.yaml --model proposed \
    --ckpt runs/lodopab/lodopab_pcdiff_proposed/checkpoints/best.pt --out_json results/proposed.json

# Evaluate: same checkpoint, DPM-Solver-accelerated (Algorithm 2, S=20)
python scripts/evaluate.py --config configs/lodopab.yaml --model proposed \
    --ckpt runs/lodopab/lodopab_pcdiff_proposed/checkpoints/best.pt \
    --sampler dpm_solver --dpm_solver_steps 20 --out_json results/proposed_dpm_solver.json

# Evaluate: FBP baseline (closed-form, no checkpoint)
python scripts/evaluate.py --config configs/lodopab.yaml --model fbp --out_json results/fbp.json

# Evaluate: DPS baseline (inference-time gradient correction; --eta grid-searched per dataset)
python scripts/evaluate.py --config configs/lodopab.yaml --model dps \
    --ckpt runs/lodopab/lodopab_pcdiff_dps_prior/checkpoints/best.pt --eta 0.1 --out_json results/dps.json

# Paired significance test against a baseline result file
python scripts/run_significance_test.py --proposed results/proposed.json --baseline results/corediff.json \
    --metric psnr --out_json results/significance_psnr.json
```

See `configs/lodopab.yaml` / `configs/mayo_aapm.yaml` for the full training configuration, and
`configs/lodopab_small.yaml` / `configs/mayo_aapm_small.yaml` for the capacity-matched PCDiff-Small variant
(`base_channels: 48`; add `loss.physics_weight: 0.0` for the paired "Small w/o L_phys" ablation row).

---

## Citation

```bibtex
@article{ahasan2026pcdiff,
  title   = {PCDiff: A Physics-Conditioned Diffusion Model for Low-Dose CT Reconstruction with Sinogram-Domain Consistency},
  author  = {Ahasan, Md Imam and Wang, Chengliang and Noor, A F M Abdun and Islam, Md Rakibul and Goh, Kah Ong Michael and Mahmud, S M Hasan},
  journal = {Computer Modeling in Engineering \& Sciences},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`).
