#!/usr/bin/env python3
"""
Fit binary flux ratio f2 per spectral channel from CHARA OIFITS data.

The primary is resolved by CHARA (diam1 ≈ 0.94 mas, baselines up to 310 m),
so the correct V² model for a binary with one resolved component is:

    V²(u,v,λ) = [f₁²·V_UD(B,λ,d)² + f₂² + 2·f₁·f₂·V_UD·cos φ] / (f₁+f₂)²

where V_UD = 2J₁(x)/x, x = π·d_rad·B/λ  (uniform disk, diameter d in mas),
      φ = 2π/λ · (u·Δα + v·Δδ),  and f₁ = 1 − f₂.

Both f₂ and diam1 are fitted jointly per spectral channel, fixing the binary
geometry (sep, PA) from the Genet et al. table_HD158837_Genet.txt broad-band
results.

T₂ is then estimated from f₂(λ) using the Planck function ratio: H-only (no
cross-instrument calibration) and H+K jointly.
"""

import numpy as np
from astropy.io import fits
from scipy.optimize import minimize
from scipy.special import j1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Known binary parameters (from Genet table, MIRC-X H-band) ─────────────────
SEP_MAS = 8.1800          # mas, from MIRC-X fit
PA_DEG  = 324.586         # deg, from MIRC-X fit (CHARA-selected node)
MAS2RAD = np.pi / (180 * 3600 * 1000)

# Separation vector in radians (OIFITS convention: u=East, v=North)
dRA  = SEP_MAS * MAS2RAD * np.sin(np.deg2rad(PA_DEG))   # East
dDec = SEP_MAS * MAS2RAD * np.cos(np.deg2rad(PA_DEG))   # North

# ── OIFITS file paths ──────────────────────────────────────────────────────────
MIRCX_FILE = (
    '/data/chara/02jun2025/2025Jun02_mircx_bbias_250602_Genet/'
    'oifits_bbias_ncoh5_snr3_flux5_itime150_select/calibrated/'
    'oiprep_2025Jun02_HD_158837_mircx_bbias_ncoh5_snr3_flux5.fits'
)
MYSTIC_FILE = (
    '/data/chara/02jun2025/2025Jun02_mystic_bbias_sky_250602_Genet/'
    'oifits_bbias_ncoh5_snr3_flux5_itime150/calibrated/'
    'oiprep_2025Jun02_HD_158837_mystic_bbias_ncoh5_snr3_flux5.fits'
)

# ── Uniform disk visibility ────────────────────────────────────────────────────
def v_ud(u, v, lam, diam_mas):
    """Uniform disk visibility amplitude. diam_mas in mas, lam in metres."""
    diam_rad = diam_mas * MAS2RAD
    B = np.sqrt(u**2 + v**2)
    x = np.pi * diam_rad * B / lam
    # Avoid division by zero at x=0
    result = np.where(x < 1e-6, 1.0, 2 * j1(x) / x)
    return result

# ── Binary V² model (resolved primary, unresolved secondary) ──────────────────
def v2_binary(u, v, lam, f2, diam1_mas):
    """
    V² for a binary with a resolved primary (uniform disk, diameter diam1_mas)
    and an unresolved secondary (point source).
    u, v in metres; lam in metres; f2 and diam1_mas are scalars.
    """
    phi = 2 * np.pi / lam * (u * dRA + v * dDec)
    f1  = 1.0 - f2
    V1  = v_ud(u, v, lam, diam1_mas)
    num = f1**2 * V1**2 + f2**2 + 2*f1*f2*V1*np.cos(phi)
    return num  # denominator (f1+f2)^2 = 1 since f1+f2=1

# ── Fit f2 and diam1 per channel from one OIFITS file ─────────────────────────
def fit_f2_per_channel(filepath, diam1_init_mas):
    """
    Fit f2 and diam1 jointly per spectral channel.
    diam1_init_mas: initial guess for primary diameter (from Genet broad-band fit).
    """
    f = fits.open(filepath)
    wav_m   = [h.data['EFF_WAVE'] for h in f if h.name == 'OI_WAVELENGTH'][0]
    n_chan  = len(wav_m)

    # Collect all V2 rows across HDUs
    all_u, all_v, all_v2, all_v2err, all_flag = [], [], [], [], []
    for hdu in f:
        if hdu.name != 'OI_VIS2':
            continue
        all_u.append(hdu.data['UCOORD'])
        all_v.append(hdu.data['VCOORD'])
        all_v2.append(hdu.data['VIS2DATA'])
        all_v2err.append(hdu.data['VIS2ERR'])
        all_flag.append(hdu.data['FLAG'])
    f.close()

    u    = np.concatenate(all_u)
    v    = np.concatenate(all_v)
    v2   = np.vstack(all_v2)      # shape (n_baselines, n_chan)
    verr = np.vstack(all_v2err)
    flag = np.vstack(all_flag)

    results = []
    for i in range(n_chan):
        lam  = wav_m[i]
        mask = ~flag[:, i] & ~np.isnan(v2[:, i]) & ~np.isnan(verr[:, i]) & (verr[:, i] > 0)
        if mask.sum() < 5:
            results.append((lam*1e6, np.nan, np.nan, np.nan, np.nan))
            continue

        ui, vi  = u[mask], v[mask]
        v2i     = v2[mask, i]
        v2erri  = verr[mask, i]

        def chi2(params):
            f2_val, d1_val = params
            if f2_val < 0 or f2_val > 0.49 or d1_val < 0.3 or d1_val > 2.0:
                return 1e10
            model = v2_binary(ui, vi, lam, f2_val, d1_val)
            return np.sum(((v2i - model) / v2erri)**2)

        res = minimize(chi2, x0=[0.04, diam1_init_mas],
                       method='Nelder-Mead',
                       options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 10000})
        f2_fit, d1_fit = res.x
        chi2_min = res.fun

        # Uncertainty on f2 from Δχ² = 1 (profile over diam1)
        f2_grid = np.linspace(max(0, f2_fit - 0.05), min(0.49, f2_fit + 0.05), 2000)
        f2_sigma = np.nan
        profile = []
        for f2_try in f2_grid:
            # Minimize over diam1 for each f2
            res1 = minimize(lambda d: chi2([f2_try, d[0]]), x0=[d1_fit],
                            method='Nelder-Mead',
                            options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 500})
            profile.append(res1.fun - chi2_min)
        profile = np.array(profile)
        try:
            lo_idx = np.where((f2_grid < f2_fit) & (profile > 1))[0]
            lo = f2_grid[lo_idx[-1]] if len(lo_idx) else f2_fit
        except Exception:
            lo = f2_fit
        try:
            hi_idx = np.where((f2_grid > f2_fit) & (profile > 1))[0]
            hi = f2_grid[hi_idx[0]] if len(hi_idx) else f2_fit
        except Exception:
            hi = f2_fit
        f2_sigma = (hi - lo) / 2

        results.append((lam*1e6, f2_fit, f2_sigma, d1_fit, chi2_min))

    return results

# ── Genet broad-band diam1 for initial guesses ────────────────────────────────
DIAM1_H_MAS = 0.9437   # mas, from Genet MIRC-X H-band fit
DIAM1_K_MAS = 0.9614   # mas, from Genet MYSTIC K-band fit

# ── Run fits ───────────────────────────────────────────────────────────────────
print("Fitting f2 and diam1 per spectral channel (this takes ~1–2 min)...")
print()
results_H = fit_f2_per_channel(MIRCX_FILE, DIAM1_H_MAS)
results_K = fit_f2_per_channel(MYSTIC_FILE, DIAM1_K_MAS)

print(f"{'lambda(um)':>12s}  {'f2':>8s}  {'sig_f2':>8s}  {'diam1(mas)':>11s}  band")
print("─" * 60)
all_lam, all_f2, all_f2err, all_band = [], [], [], []
for lam, f2, sig, d1, c2 in results_H:
    if not np.isnan(f2):
        print(f"  {lam:10.4f}  {f2:8.5f}  {sig:8.5f}  {d1:11.4f}  H")
        all_lam.append(lam); all_f2.append(f2); all_f2err.append(sig); all_band.append('H')
print()
for lam, f2, sig, d1, c2 in results_K:
    if not np.isnan(f2):
        print(f"  {lam:10.4f}  {f2:8.5f}  {sig:8.5f}  {d1:11.4f}  K")
        all_lam.append(lam); all_f2.append(f2); all_f2err.append(sig); all_band.append('K')

all_lam   = np.array(all_lam)
all_f2    = np.array(all_f2)
all_f2err = np.array(all_f2err)
all_band  = np.array(all_band)

# ── Estimate T2 from H/K median flux ratio ────────────────────────────────────
# Within H-band (0.15 μm span), the Planck slope is only 2–5% between
# T2=4000 K and T2=6000 K, while per-channel σ(f2)/f2 ≈ 8%.  T2 cannot be
# usefully constrained from within-H variation alone.
# The K-band channels at 2.27–2.35 μm show a drop in f2 (possible CO 2-0
# band-head absorption at 2.293 μm); those channels are excluded below.

def planck(lam_um, T):
    h, c, k = 6.62607e-34, 2.99792e8, 1.38065e-23
    lam = lam_um * 1e-6
    return (2*h*c**2 / lam**5) / (np.exp(h*c/(lam*k*T)) - 1.0)

T1 = 5000.0  # G3III primary

mask_H = all_band == 'H'
lam_H, f2_H, err_H = all_lam[mask_H], all_f2[mask_H], all_f2err[mask_H]

# K-band: exclude λ > 2.25 μm (CO band-head region)
mask_K = (all_band == 'K') & (all_lam < 2.25)
lam_K, f2_K, err_K = all_lam[mask_K], all_f2[mask_K], all_f2err[mask_K]

# Weighted means
w_H   = 1.0 / err_H**2;   f2_H_mean = np.sum(w_H*f2_H) / np.sum(w_H)
w_K   = 1.0 / err_K**2;   f2_K_mean = np.sum(w_K*f2_K) / np.sum(w_K)
# Error on weighted mean
f2_H_err = 1.0 / np.sqrt(np.sum(w_H))
f2_K_err = 1.0 / np.sqrt(np.sum(w_K))

# Also record all K channels separately for the plot
mask_K_all = all_band == 'K'
lam_K_all  = all_lam[mask_K_all]
f2_K_all   = all_f2[mask_K_all]
err_K_all  = all_f2err[mask_K_all]

# Solve for T2 from f2_H_mean / f2_K_mean = [B(lam_H_eff, T2)/B(lam_K_eff, T2)]
#                                          / [B(lam_H_eff, T1)/B(lam_K_eff, T1)]
# (angular diameter ratio cancels in the ratio)
from scipy.optimize import brentq

lam_H_eff = np.average(lam_H, weights=w_H)   # effective H wavelength
lam_K_eff = np.average(lam_K, weights=w_K)   # effective K wavelength (CO-free)
ratio_obs  = f2_H_mean / f2_K_mean

def ratio_model(T2):
    return ((planck(lam_H_eff, T2) / planck(lam_K_eff, T2)) /
            (planck(lam_H_eff, T1) / planck(lam_K_eff, T1)))

try:
    T2_HK = brentq(lambda T2: ratio_model(T2) - ratio_obs, 2000, 30000)
    # Propagate: delta T2 from delta(ratio)
    dT  = 10.0
    dratio_dT = (ratio_model(T2_HK + dT) - ratio_model(T2_HK - dT)) / (2*dT)
    ratio_err  = ratio_obs * np.sqrt((f2_H_err/f2_H_mean)**2 + (f2_K_err/f2_K_mean)**2)
    T2_HK_err  = abs(ratio_err / dratio_dT)
except Exception as e:
    T2_HK, T2_HK_err = np.nan, np.nan
    print(f"H/K T2 solve failed: {e}")

# Implied (R2/R1)^2 from H-band mean
scale_HK = f2_H_mean / (planck(lam_H_eff, T2_HK) / planck(lam_H_eff, T1))

print()
print("─" * 65)
print("Temperature and mass from H/K weighted-mean flux ratio:")
print("─" * 65)
print(f"  Effective λ_H = {lam_H_eff:.4f} μm,  λ_K = {lam_K_eff:.4f} μm")
print(f"  Weighted mean f2_H = {f2_H_mean:.5f} ± {f2_H_err:.5f}")
print(f"  Weighted mean f2_K = {f2_K_mean:.5f} ± {f2_K_err:.5f}  (λ < 2.25 μm)")
print(f"  f2_H/f2_K = {ratio_obs:.4f}  →  T2 = {T2_HK:.0f} ± {T2_HK_err:.0f} K")
print(f"  (R2/R1)² = {scale_HK:.5f}")
print()

# Implied f2_G and beta
LAM_G  = 0.590
f2_G   = scale_HK * planck(LAM_G, T2_HK) / planck(LAM_G, T1)
a0_G   = 3.9653   # mas
a_rel  = 13.779   # mas
M_tot  = 3.83     # Msun
beta   = f2_G + a0_G / a_rel
print(f"  Implied f2_G = {f2_G:.5f}")
print(f"  β = f2_G + a0/a_rel = {beta:.4f}")
print(f"  m2 = {beta*M_tot:.2f} Msun,  m1 = {(1-beta)*M_tot:.2f} Msun")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

# H-band points
ax.errorbar(lam_H, f2_H, yerr=err_H, fmt='o', color='steelblue',
            label='MIRC-X (H)', capsize=3, markersize=6)
# K-band: CO-free channels (filled), CO-affected channels (open)
ax.errorbar(lam_K, f2_K, yerr=err_K,
            fmt='s', color='firebrick', label='MYSTIC K (CO-free)', capsize=3, markersize=6)
mask_K_co = (all_band == 'K') & (all_lam >= 2.25)
if mask_K_co.sum():
    ax.errorbar(all_lam[mask_K_co], all_f2[mask_K_co], yerr=all_f2err[mask_K_co],
                fmt='s', color='firebrick', fillstyle='none',
                label='MYSTIC K (CO?)', capsize=3, markersize=6)

# Model curve for the best-fit T2
lam_grid = np.linspace(1.48, 2.42, 500)
if not np.isnan(T2_HK):
    model_curve = scale_HK * np.array([planck(l, T2_HK)/planck(l, T1) for l in lam_grid])
    ax.plot(lam_grid, model_curve, 'k-',
            label=f'H/K Planck: T₂={T2_HK:.0f}±{T2_HK_err:.0f} K', lw=1.5)

# Weighted means
ax.axhline(f2_H_mean, color='steelblue', ls='--', lw=1, alpha=0.7,
           label=f'H mean={f2_H_mean:.4f}')
ax.axhline(f2_K_mean, color='firebrick', ls='--', lw=1, alpha=0.7,
           label=f'K mean={f2_K_mean:.4f}')

ax.axvspan(1.50, 1.72, alpha=0.08, color='blue', label='H band')
ax.axvspan(2.00, 2.40, alpha=0.08, color='red',  label='K band')
ax.axvline(2.25, color='gray', ls=':', lw=0.8, alpha=0.7, label='CO cut')

ax.set_xlabel('Wavelength (μm)', fontsize=12)
ax.set_ylabel('f₂  (secondary flux fraction)', fontsize=12)
ax.set_title('HD 158837 — per-channel f₂ from CHARA OIFITS\n'
             f'sep={SEP_MAS:.3f} mas, PA={PA_DEG:.1f}° (fixed), resolved primary (diam1 free)',
             fontsize=11)
ax.legend(fontsize=8.5, ncol=2)
ax.set_xlim(1.45, 2.45)
ax.set_ylim(0, 0.12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
outfile = 'chara_f2_per_channel.png'
plt.savefig(outfile, dpi=150)
print(f"\nPlot saved to {outfile}")
