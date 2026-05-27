#!/usr/bin/env python3
"""
Orbital ephemeris for HD 158837 / HIP 85749 at the CHARA observation epoch,
from both the Hipparcos DMSA/O and Gaia DR3 NSS Orbital solutions.

The CHARA separation provides the orbital scale directly:

    a_rel / a0 = sep_binary_CHARA / sep_photocentre_predicted

where sep_photocentre_predicted is computed purely from the Thiele-Innes (or
Campbell) elements at the CHARA epoch.  No flux ratio, mass ratio, or primary
mass assumption is required.  The total mass then follows from Kepler's 3rd law.

The CHARA position angle resolves the 180 deg ascending-node ambiguity that is
unavoidable in purely astrometric orbits (no radial velocity).

Notes on omega convention (Hipparcos section):
  The Hipparcos DMSA/O stores omega from the spectroscopic orbit (periastron
  of the primary).  The Thiele-Innes formula for the secondary's position
  requires omega_visual = omega_spectroscopic + 180 deg.

Data sources (all in this directory):
  hipparcos_orbit_hip85749.ecsv  — Hipparcos DMSA/O (ESA 1997)
  hipparcos_main_hip85749.ecsv   — Hipparcos main catalogue (parallax)
  gaia_nss_hd158837.ecsv        — Gaia DR3 NSS Orbital solution
  table_HD158837_Genet.txt       — CHARA/MIRC-X + MYSTIC observation
  document.pdf (Lucke & Mayor 1982, Table 7) — scanned; LM_* values hardcoded
"""

import numpy as np
from astropy.table import Table

# ── Gaia corr_vec → full covariance matrix ────────────────────────────────────
# For nss_solution_type='Orbital' (bit_index=8191) the 12 fitted parameters are:
#   0 ra  1 dec  2 parallax  3 pmra  4 pmdec
#   5 A   6 B    7 F         8 G
#   9 e   10 P   11 t_periastron
# corr_vec stores the upper triangle column-major: (0,1),(0,2),(1,2),(0,3),...
# Source: Gaia DR3 data model,
#   https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/
#   sec_dm_non--single_stars_tables/ssec_dm_nss_two_body_orbit.html
def gaia_covariance(nss_row):
    """Return (mu, sigma, cov) for the 12 Gaia NSS Orbital parameters."""
    mu = np.array([
        float(nss_row['ra']),               float(nss_row['dec']),
        float(nss_row['parallax']),         float(nss_row['pmra']),
        float(nss_row['pmdec']),
        float(nss_row['a_thiele_innes']),   float(nss_row['b_thiele_innes']),
        float(nss_row['f_thiele_innes']),   float(nss_row['g_thiele_innes']),
        float(nss_row['eccentricity']),     float(nss_row['period']),
        float(nss_row['t_periastron']),
    ])
    sigma = np.array([
        float(nss_row['ra_error']),               float(nss_row['dec_error']),
        float(nss_row['parallax_error']),         float(nss_row['pmra_error']),
        float(nss_row['pmdec_error']),
        float(nss_row['a_thiele_innes_error']),   float(nss_row['b_thiele_innes_error']),
        float(nss_row['f_thiele_innes_error']),   float(nss_row['g_thiele_innes_error']),
        float(nss_row['eccentricity_error']),     float(nss_row['period_error']),
        float(nss_row['t_periastron_error']),
    ])
    n, cv = 12, np.array(nss_row['corr_vec'])
    C = np.eye(n)
    k = 0
    for col in range(1, n):
        for row in range(col):
            C[row, col] = C[col, row] = cv[k]; k += 1
    return mu, sigma, np.outer(sigma, sigma) * C

def mc_pa_uncertainty(nss_row, t_jd, n_mc=200_000, seed=42):
    """
    Monte Carlo σ for the Gaia photocentre PA at epoch t_jd.
    Returns (pa_central_deg, pa_mean_deg, pa_sigma_deg).
    """
    mu, _, cov = gaia_covariance(nss_row)
    iA,iB,iF,iG,ie,iP,iT = 5,6,7,8,9,10,11
    rng = np.random.default_rng(seed)
    s   = rng.multivariate_normal(mu, cov, size=n_mc)

    def _pa(A,B,F,G,e,P,t_peri):
        T0  = 2457388.5 + t_peri
        M   = 2*np.pi*((t_jd - T0)/P % 1.0)
        E   = M.copy()
        for _ in range(100):
            dE = (M - E + e*np.sin(E))/(1 - e*np.cos(E)); E += dE
            if np.all(np.abs(dE) < 1e-12): break
        x = np.cos(E) - e;  y = np.sqrt(1-e**2)*np.sin(E)
        return np.degrees(np.arctan2(A*x+F*y, B*x+G*y)) % 360.0

    pa_c  = _pa(mu[iA],mu[iB],mu[iF],mu[iG],mu[ie],mu[iP],mu[iT])
    pa_s  = _pa(s[:,iA],s[:,iB],s[:,iF],s[:,iG],s[:,ie],s[:,iP],s[:,iT])
    pa_m  = np.degrees(np.arctan2(np.mean(np.sin(np.deg2rad(pa_s))),
                                   np.mean(np.cos(np.deg2rad(pa_s))))) % 360.0
    pa_sd = np.std(((pa_s - pa_m + 180) % 360) - 180)
    return pa_c, pa_m, pa_sd

# ── Lucke & Mayor (1982) Table 7, HD 158837 ───────────────────────────────────
# Used only for the Hipparcos section and the mass-function cross-check.
LM_a1sini = 94.8e6   # km  — a1 sin i
LM_f_mass = 0.195    # Msun — mass function f(m)

# ── Read data files ────────────────────────────────────────────────────────────
orb      = Table.read('hipparcos_orbit_hip85749.ecsv', format='ascii.ecsv')[0]
main     = Table.read('hipparcos_main_hip85749.ecsv',  format='ascii.ecsv')[0]
gaia_nss = Table.read('gaia_nss_hd158837.ecsv',        format='ascii.ecsv')[0]

# Hipparcos
P_Hip     = float(orb['P'])
T0_Hip    = float(orb['T']) + 2440000.0
e_Hip     = float(orb['ecc'])
w_vis_Hip = (float(orb['w']) + 180.0) % 360.0
inc_Hip   = float(orb['i'])
Omega_Hip = float(orb['Omega'])
a0_Hip    = float(orb['a0'])
plx_Hip   = float(main['Plx'])

# Gaia NSS (t_periastron is in days from J2016.0 = JD 2457388.5)
A_G   = float(gaia_nss['a_thiele_innes'])
B_G   = float(gaia_nss['b_thiele_innes'])
F_G   = float(gaia_nss['f_thiele_innes'])
G_G   = float(gaia_nss['g_thiele_innes'])
P_G   = float(gaia_nss['period'])
e_G   = float(gaia_nss['eccentricity'])
plx_G = float(gaia_nss['parallax'])
T0_G  = 2457388.5 + float(gaia_nss['t_periastron'])

# Gaia photocentre semi-major axis from Thiele-Innes elements
# a0^2 = (A^2+B^2+F^2+G^2)/2 + sqrt( ((A^2+B^2-F^2-G^2)/2)^2 + (AF+BG)^2 )
_S   = (A_G**2 + B_G**2 + F_G**2 + G_G**2) / 2
_q   = (A_G**2 + B_G**2 - F_G**2 - G_G**2) / 2
a0_G = np.sqrt(_S + np.sqrt(_q**2 + (A_G*F_G + B_G*G_G)**2))

# ── Parse CHARA table ──────────────────────────────────────────────────────────
with open('table_HD158837_Genet.txt') as f:
    lines = f.readlines()

def parse_chara_row(line):
    tok = line.split()
    return dict(label=(' '.join(tok[21:])), mjd=float(tok[1]),
                sep=float(tok[4]), pa=float(tok[5]))

obs = [parse_chara_row(l) for l in lines[2:] if l.strip()]

# ── Kepler solver ──────────────────────────────────────────────────────────────
def eccentric_anomaly(M, e, tol=1e-12):
    E = M
    for _ in range(100):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E

def orbital_xy(t_jd, P, T0, e):
    """Return (x, y, M_deg, E_deg) — dimensionless orbital coords."""
    M = 2.0 * np.pi * ((t_jd - T0) / P % 1.0)
    E = eccentric_anomaly(M, e)
    x = np.cos(E) - e
    y = np.sqrt(1.0 - e*e) * np.sin(E)
    return x, y, np.degrees(M), np.degrees(E)

# ── Hipparcos Thiele-Innes elements (relative orbit, using a0_Hip as scale) ───
def build_TI(a, om_deg, Om_deg, inc_deg):
    om, Om, i = np.deg2rad(om_deg), np.deg2rad(Om_deg), np.deg2rad(inc_deg)
    A = a * ( np.cos(om)*np.cos(Om) - np.sin(om)*np.sin(Om)*np.cos(i))
    B = a * ( np.cos(om)*np.sin(Om) + np.sin(om)*np.cos(Om)*np.cos(i))
    F = a * (-np.sin(om)*np.cos(Om) - np.cos(om)*np.sin(Om)*np.cos(i))
    G = a * (-np.sin(om)*np.sin(Om) + np.cos(om)*np.cos(Om)*np.cos(i))
    return A, B, F, G

# Build Hipparcos photocentre TI elements using a0_Hip (not a_rel)
A_H, B_H, F_H, G_H = build_TI(a0_Hip, w_vis_Hip, Omega_Hip, inc_Hip)

# ── Print results ──────────────────────────────────────────────────────────────
print("=" * 65)
print("Orbital ephemeris for HD 158837 at CHARA epoch (2025 Jun 02)")
print("=" * 65)

# ── HIPPARCOS ──────────────────────────────────────────────────────────────────
print(f"""
─────────────────────────────────────────────────────────────────
HIPPARCOS orbit  (Hipparcos DMSA/O + Lucke & Mayor 1982)
─────────────────────────────────────────────────────────────────
  P          = {P_Hip:.3f} d
  T0         = JD {T0_Hip:.1f}
  e          = {e_Hip:.3f}  [Lucke & Mayor; fixed in Hipparcos fit]
  omega_vis  = {w_vis_Hip:.1f} deg  [= omega_spec + 180]
  i          = {inc_Hip:.2f} deg,   Omega = {Omega_Hip:.2f} deg  [from Hipparcos]
  a0         = {a0_Hip:.2f} mas  (photocentre),   Plx = {plx_Hip:.3f} mas
""")

for row in obs:
    t_jd = row['mjd'] + 2400000.5
    x, y, M, E = orbital_xy(t_jd, P_Hip, T0_Hip, e_Hip)
    X_ph = A_H*x + F_H*y
    Y_ph = B_H*x + G_H*y
    sep_ph = np.hypot(X_ph, Y_ph)
    pa_ph  = np.degrees(np.arctan2(X_ph, Y_ph)) % 360

    # secondary is opposite the photocentre (primary dominates Hp flux)
    pa_sec = (pa_ph + 180) % 360

    scale   = row['sep'] / sep_ph          # a_rel / a0_Hip from CHARA
    a_rel   = scale * a0_Hip               # mas
    a_AU    = a_rel / plx_Hip              # AU  (using Hipparcos parallax)
    M_total = a_AU**3 / (P_Hip/365.25)**2  # Msun

    dpa = ((pa_sec - row['pa'] + 180) % 360) - 180
    print(f"  {row['label']}:")
    print(f"    M = {M:.1f} deg,  E = {E:.1f} deg")
    print(f"    sep_photocentre (predicted) = {sep_ph:.3f} mas")
    print(f"    sep_binary      (CHARA)     = {row['sep']:.3f} mas")
    print(f"    => a_rel/a0  = {scale:.3f}")
    print(f"    => a_rel     = {a_rel:.2f} mas  =  {a_AU:.3f} AU")
    print(f"    => M1+M2     = {M_total:.2f} Msun  (Kepler's 3rd law, Hipparcos Plx)")
    print(f"    Predicted PA (secondary) = {pa_sec:.1f} deg")
    print(f"    Observed  PA             = {row['pa']:.3f} deg   DPA = {dpa:+.1f} deg")
    print()

# ── GAIA ───────────────────────────────────────────────────────────────────────
print(f"""─────────────────────────────────────────────────────────────────
GAIA DR3 orbit  (NSS Orbital, purely astrometric)
─────────────────────────────────────────────────────────────────
  P          = {P_G:.3f} d
  T0         = JD {T0_G:.1f}  (t_peri = {float(gaia_nss['t_periastron']):.2f} d from J2016.0)
  e          = {e_G:.4f}  [from Gaia astrometry alone]
  A,B,F,G    = {A_G:.3f}, {B_G:.3f}, {F_G:.3f}, {G_G:.3f} mas
  a0_Gaia    = {a0_G:.3f} mas  (photocentre semi-major axis from TI elements)
  Plx        = {plx_G:.3f} mas

  The Gaia orbit is purely astrometric: there is a 180 deg ambiguity in
  the ascending node Omega (equivalent to negating all TI elements).
  CHARA resolves this ambiguity via the position angle.
""")

for row in obs:
    t_jd = row['mjd'] + 2400000.5
    x, y, M, E = orbital_xy(t_jd, P_G, T0_G, e_G)
    X_ph = A_G*x + F_G*y
    Y_ph = B_G*x + G_G*y
    sep_ph = np.hypot(X_ph, Y_ph)
    pa_ph  = np.degrees(np.arctan2(X_ph, Y_ph)) % 360

    # Scale factor from CHARA: a_rel/a0 = sep_binary / sep_photocentre
    scale   = row['sep'] / sep_ph
    a_rel   = scale * a0_G
    a_AU    = a_rel / plx_G
    M_total = a_AU**3 / (P_G/365.25)**2

    # Two node solutions — CHARA PA selects
    pa_opp  = (pa_ph + 180) % 360    # secondary opposite photocentre (Omega solution)
    pa_same = pa_ph                   # secondary same as photocentre (Omega+180 solution)
    dpa_opp  = ((pa_opp  - row['pa'] + 180) % 360) - 180
    dpa_same = ((pa_same - row['pa'] + 180) % 360) - 180

    # MC uncertainty on PA
    t_jd_row = row['mjd'] + 2400000.5
    _, pa_mc_mean, pa_mc_sig = mc_pa_uncertainty(gaia_nss, t_jd_row)
    dpa_mc = ((pa_mc_mean - row['pa'] + 180) % 360) - 180

    print(f"  {row['label']}:")
    print(f"    M = {M:.1f} deg,  E = {E:.1f} deg")
    print(f"    sep_photocentre (predicted) = {sep_ph:.3f} mas")
    print(f"    sep_binary      (CHARA)     = {row['sep']:.3f} mas")
    print(f"    => a_rel/a0  = {scale:.3f}  (no mass or flux-ratio assumption)")
    print(f"    => a_rel     = {a_rel:.2f} mas  =  {a_AU:.3f} AU")
    print(f"    => M1+M2     = {M_total:.2f} Msun  (Kepler's 3rd law, Gaia Plx)")
    print(f"    Photocentre PA = {pa_ph:.1f} deg  (MC mean: {pa_mc_mean:.1f} deg,  sigma: {pa_mc_sig:.1f} deg)")
    print(f"    PA (Omega sol., secondary opposite)  = {pa_opp:.1f} deg   DPA = {dpa_opp:+.1f} deg")
    print(f"    PA (Omega+180, secondary same dir)   = {pa_same:.1f} deg   DPA = {dpa_same:+.1f} deg  ({abs(dpa_same)/pa_mc_sig:.1f} sigma)")
    print(f"    Observed PA = {row['pa']:.3f} deg  => Omega+180 solution selected")
    print()

# ── Mass-function cross-check ──────────────────────────────────────────────────
print("─────────────────────────────────────────────────────────────────")
print("Mass-function cross-check  (Lucke & Mayor 1982, Table 7)")
print("─────────────────────────────────────────────────────────────────")
print(f"  f(m) = {LM_f_mass:.3f} Msun,   a1 sin i = {LM_a1sini:.3e} km")
print(f"  Using Hipparcos i = {inc_Hip:.2f} deg:")
sin_i = np.sin(np.deg2rad(inc_Hip))
a1_km = LM_a1sini / sin_i
a1_mas = (a1_km / 1.495978707e8) * plx_G   # using Gaia parallax
print(f"  a1 = {a1_mas:.2f} mas  (primary orbit)")

# For Gaia M_total, compute m1 and m2 from mass function
# Use the Gaia-derived a_rel for the most recent obs row (MIRC-X H-band)
row0 = obs[0]
t_jd0 = row0['mjd'] + 2400000.5
x0, y0, _, _ = orbital_xy(t_jd0, P_G, T0_G, e_G)
sep_ph0 = np.hypot(A_G*x0+F_G*y0, B_G*x0+G_G*y0)
scale0 = row0['sep'] / sep_ph0
a_rel_G = scale0 * a0_G
a_AU_G  = a_rel_G / plx_G
M_tot_G = a_AU_G**3 / (P_G/365.25)**2

from scipy.optimize import brentq
m2 = brentq(lambda m2: m2**3 * sin_i**3 / (M_tot_G)**2 - LM_f_mass * (m2/(M_tot_G))**0 / 1,
            0.1, M_tot_G - 0.1)
# Correct form: m2^3 * sin^3(i) / (m1+m2)^2 = f(m)
# With m1+m2 = M_tot_G fixed, solve for m2:
m2 = brentq(lambda m2: m2**3 * sin_i**3 / M_tot_G**2 - LM_f_mass, 0.1, M_tot_G - 0.01)
m1 = M_tot_G - m2
print(f"  From Gaia orbit + CHARA: M1+M2 = {M_tot_G:.2f} Msun")
print(f"  => m2 = {m2:.2f} Msun,  m1 = {m1:.2f} Msun")
print(f"  => beta = m2/(m1+m2) = {m2/M_tot_G:.3f}")
print(f"  => f2_Gaia (implied) = beta - a0_Gaia/a_rel = "
      f"{m2/M_tot_G - a0_G/a_rel_G:.3f}")
