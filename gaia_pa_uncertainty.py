#!/usr/bin/env python3
"""
Monte Carlo uncertainty on the Gaia DR3 predicted photocentre PA at the CHARA epoch.

The corr_vec field in the Gaia NSS Orbital solution is the upper triangle of the
parameter correlation matrix, stored column-major.  For nss_solution_type='Orbital'
with bit_index=8191, the 12 parameters in order are:

  0  ra
  1  dec
  2  parallax
  3  pmra
  4  pmdec
  5  a_thiele_innes  (A)
  6  b_thiele_innes  (B)
  7  f_thiele_innes  (F)
  8  g_thiele_innes  (G)
  9  eccentricity    (e)
  10 period          (P)
  11 t_periastron    (T0)

Source: Gaia DR3 data model documentation,
  https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/
  sec_dm_non--single_stars_tables/ssec_dm_nss_two_body_orbit.html
"""

import numpy as np
from astropy.table import Table

# ── Load data ─────────────────────────────────────────────────────────────────
gaia_nss = Table.read('gaia_nss_hd158837.ecsv', format='ascii.ecsv')[0]

# CHARA MIRC-X epoch and observation
with open('table_HD158837_Genet.txt') as f:
    lines = f.readlines()
tok      = lines[2].split()
mjd      = float(tok[1])
pa_chara = float(tok[5])
t_jd     = mjd + 2400000.5

# ── Unpack corr_vec into full 12x12 correlation matrix ───────────────────────
# Upper triangle, column-major: (0,1),(0,2),(1,2),(0,3),(1,3),(2,3),(0,4),...
n       = 12
corr_vec = np.array(gaia_nss['corr_vec'])
C = np.eye(n)
k = 0
for col in range(1, n):
    for row in range(col):
        C[row, col] = corr_vec[k]
        C[col, row] = corr_vec[k]
        k += 1

# Parameter names (for reference)
PARAM_NAMES = ['ra', 'dec', 'parallax', 'pmra', 'pmdec',
               'A', 'B', 'F', 'G', 'e', 'P', 't_peri']

# Central values and errors for the 12 parameters
mu    = np.array([
    float(gaia_nss['ra']),
    float(gaia_nss['dec']),
    float(gaia_nss['parallax']),
    float(gaia_nss['pmra']),
    float(gaia_nss['pmdec']),
    float(gaia_nss['a_thiele_innes']),
    float(gaia_nss['b_thiele_innes']),
    float(gaia_nss['f_thiele_innes']),
    float(gaia_nss['g_thiele_innes']),
    float(gaia_nss['eccentricity']),
    float(gaia_nss['period']),
    float(gaia_nss['t_periastron']),
])
sigma = np.array([
    float(gaia_nss['ra_error']),
    float(gaia_nss['dec_error']),
    float(gaia_nss['parallax_error']),
    float(gaia_nss['pmra_error']),
    float(gaia_nss['pmdec_error']),
    float(gaia_nss['a_thiele_innes_error']),
    float(gaia_nss['b_thiele_innes_error']),
    float(gaia_nss['f_thiele_innes_error']),
    float(gaia_nss['g_thiele_innes_error']),
    float(gaia_nss['eccentricity_error']),
    float(gaia_nss['period_error']),
    float(gaia_nss['t_periastron_error']),
])

# Full covariance matrix
cov = np.outer(sigma, sigma) * C

# ── PA calculation for a single set of parameters ────────────────────────────
def eccentric_anomaly(M, e, tol=1e-12):
    E = M.copy() if hasattr(M, 'copy') else float(M)
    for _ in range(100):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if np.all(np.abs(dE) < tol):
            break
    return E

def photocentre_pa(A, B, F, G, e, P, t_peri_days, t_jd):
    """Return photocentre PA (degrees, 0-360) at epoch t_jd."""
    T0   = 2457388.5 + t_peri_days          # convert from days since J2016.0
    M    = 2.0 * np.pi * ((t_jd - T0) / P % 1.0)
    E    = eccentric_anomaly(M, e)
    x    = np.cos(E) - e
    y    = np.sqrt(1.0 - e**2) * np.sin(E)
    X_ph = A * x + F * y
    Y_ph = B * x + G * y
    return np.degrees(np.arctan2(X_ph, Y_ph)) % 360.0

# ── Monte Carlo ───────────────────────────────────────────────────────────────
N_MC = 500_000
rng  = np.random.default_rng(42)

# Draw samples from the multivariate Gaussian
samples = rng.multivariate_normal(mu, cov, size=N_MC)

# Indices of the orbital parameters we need
iA, iB, iF, iG = 5, 6, 7, 8
ie, iP, iT      = 9, 10, 11

pa_samples = photocentre_pa(
    samples[:, iA], samples[:, iB],
    samples[:, iF], samples[:, iG],
    samples[:, ie], samples[:, iP],
    samples[:, iT], t_jd,
)

# The 'Omega+180' solution has the secondary at the same PA as the photocentre.
# Compute circular mean and std of PA distribution.
pa_rad    = np.deg2rad(pa_samples)
mean_sin  = np.mean(np.sin(pa_rad))
mean_cos  = np.mean(np.cos(pa_rad))
pa_mean   = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0

# Circular std: angular deviations from the mean, wrapped to (-180, 180)
dpa       = ((pa_samples - pa_mean + 180) % 360) - 180
pa_std    = np.std(dpa)

pa_central = photocentre_pa(
    mu[iA], mu[iB], mu[iF], mu[iG],
    mu[ie], mu[iP], mu[iT], t_jd,
)

# ── Report ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Gaia DR3 photocentre PA uncertainty at CHARA epoch")
print("=" * 60)
print(f"\nCHARA epoch : JD {t_jd:.2f}  (MJD {mjd:.3f})")
print(f"CHARA PA    : {pa_chara:.3f} deg  (MIRC-X H-band)")
print()
print(f"Central PA (nominal elements) : {pa_central:.2f} deg")
print(f"MC mean PA  ({N_MC//1000}k samples)   : {pa_mean:.2f} deg")
print(f"MC sigma_PA                   : {pa_std:.2f} deg")
print()
residual = ((pa_mean - pa_chara + 180) % 360) - 180
print(f"Residual (predicted - CHARA)  : {residual:+.1f} deg")
print(f"Discrepancy in sigma          : {abs(residual)/pa_std:.1f} sigma")
print()

# Per-parameter sensitivity (vary one at a time, holding others fixed)
print("─" * 60)
print("Sensitivity: sigma_PA contribution per parameter")
print("─" * 60)
orbital_params = [
    (iA, 'A (Thiele-Innes)'),
    (iB, 'B (Thiele-Innes)'),
    (iF, 'F (Thiele-Innes)'),
    (iG, 'G (Thiele-Innes)'),
    (ie, 'eccentricity'),
    (iP, 'period'),
    (iT, 't_periastron'),
]
contributions = []
for idx, name in orbital_params:
    p_hi = mu.copy(); p_hi[idx] += sigma[idx]
    p_lo = mu.copy(); p_lo[idx] -= sigma[idx]
    pa_hi = photocentre_pa(p_hi[iA],p_hi[iB],p_hi[iF],p_hi[iG],p_hi[ie],p_hi[iP],p_hi[iT],t_jd)
    pa_lo = photocentre_pa(p_lo[iA],p_lo[iB],p_lo[iF],p_lo[iG],p_lo[ie],p_lo[iP],p_lo[iT],t_jd)
    dpa_hi = ((pa_hi - pa_central + 180) % 360) - 180
    dpa_lo = ((pa_lo - pa_central + 180) % 360) - 180
    contrib = abs(dpa_hi - dpa_lo) / 2
    contributions.append(contrib)
    print(f"  {name:25s}  sigma={sigma[idx]:.4f}  =>  dPA = {contrib:.2f} deg")

print(f"\n  Quadrature sum (orbital only) : {np.sqrt(sum(c**2 for c in contributions)):.2f} deg")
print(f"  Full MC sigma_PA              : {pa_std:.2f} deg")
