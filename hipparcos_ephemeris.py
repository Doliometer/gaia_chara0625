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

def el_badry_inflation(ruwe, plx_mas, sigma_eta_mas=0.5):
    """
    Parallax uncertainty inflation factor from El-Badry (2025, arXiv:2504.11528).
    Equations 3-4: alpha=2.77, f0=3.73, beta=0.065, gamma=-0.056.

    sigma_eta_mas: per-CCD along-scan uncertainty from Holl et al. (2023) /
      El-Badry et al. (2024).  For G~5.3 we use 0.5 mas; the dependence is
      weak (gamma=-0.056), so the exact value matters little.

    Note (Sect. 5.4): the correction is validated for five-parameter solutions.
    For NSS orbital parallaxes it is approximate — the true inflation factor
    likely has more complex parameter dependencies.
    """
    if ruwe <= 1.0:
        return 1.0
    alpha, f0, beta, gamma = 2.77, 3.73, 0.065, -0.056
    f_max = f0 * (plx_mas / 10.0)**beta * (sigma_eta_mas / 0.1)**gamma
    return 1.0 + (f_max - 1.0) * (1.0 - np.exp(-alpha * (ruwe - 1.0)))

def mc_gaia_uncertainty(nss_row, t_jd, sep_chara, n_mc=200_000, seed=42):
    """
    Monte Carlo uncertainties for Gaia photocentre PA, sep, scale, and mass
    at epoch t_jd.

    Returns (pa_central, pa_mean, pa_sigma,
             sep_central, sep_sigma, scale_sigma,
             M_central, M_orb_sigma, plx_nom).

    M_orb_sigma is the mass uncertainty from orbital elements alone
    (parallax held fixed at its nominal NSS value), so that the parallax
    contribution can be added separately with or without El-Badry inflation.
    """
    mu, _, cov = gaia_covariance(nss_row)
    iA,iB,iF,iG,ie,iP,iT = 5,6,7,8,9,10,11
    plx_nom = mu[2]   # index 2 = parallax in the 12-parameter NSS solution
    rng = np.random.default_rng(seed)
    s   = rng.multivariate_normal(mu, cov, size=n_mc)

    def _solve(A,B,F,G,e,P,t_peri):
        T0  = 2457388.5 + t_peri
        M_a = 2*np.pi*((t_jd - T0)/P % 1.0)
        E   = M_a.copy()
        for _ in range(100):
            dE = (M_a - E + e*np.sin(E))/(1 - e*np.cos(E)); E += dE
            if np.all(np.abs(dE) < 1e-12): break
        x = np.cos(E) - e;  y = np.sqrt(1-e**2)*np.sin(E)
        X, Y  = A*x+F*y, B*x+G*y
        sep   = np.hypot(X, Y)
        pa    = np.degrees(np.arctan2(X, Y)) % 360.0
        _S    = (A**2+B**2+F**2+G**2)/2
        _q    = (A**2+B**2-F**2-G**2)/2
        a0    = np.sqrt(np.clip(_S + np.sqrt(np.clip(_q**2+(A*F+B*G)**2, 0, None)), 0, None))
        return sep, pa, a0

    # Central values
    sep_c, pa_c, a0_c = _solve(mu[iA],mu[iB],mu[iF],mu[iG],mu[ie],mu[iP],mu[iT])
    a_rel_c = sep_chara * a0_c / sep_c
    M_c     = (a_rel_c / plx_nom)**3 / (mu[iP]/365.25)**2

    # MC samples
    sep_s, pa_s, a0_s = _solve(s[:,iA],s[:,iB],s[:,iF],s[:,iG],s[:,ie],s[:,iP],s[:,iT])

    # PA statistics
    pa_m  = np.degrees(np.arctan2(np.mean(np.sin(np.deg2rad(pa_s))),
                                   np.mean(np.cos(np.deg2rad(pa_s))))) % 360.0
    pa_sd = np.std(((pa_s - pa_m + 180) % 360) - 180)

    # Scale and sep statistics
    scale_s  = sep_chara / sep_s
    sep_sd   = np.std(sep_s)
    scale_sd = np.std(scale_s)

    # Mass: orbital contribution only (plx fixed at nominal to isolate it)
    a_rel_s  = sep_chara * a0_s / sep_s
    M_orb_s  = (a_rel_s / plx_nom)**3 / (s[:,iP]/365.25)**2
    M_orb_sd = np.std(M_orb_s)

    return pa_c, pa_m, pa_sd, sep_c, sep_sd, scale_sd, M_c, M_orb_sd, plx_nom

# ── Lucke & Mayor (1982) Table 7, HD 158837 ───────────────────────────────────
# Used only for the Hipparcos section and the mass-function cross-check.
LM_a1sini = 94.8e6   # km  — a1 sin i
LM_f_mass = 0.195    # Msun — mass function f(m)

# ── Read data files ────────────────────────────────────────────────────────────
orb      = Table.read('hipparcos_orbit_hip85749.ecsv', format='ascii.ecsv')[0]
main     = Table.read('hipparcos_main_hip85749.ecsv',  format='ascii.ecsv')[0]
gaia_nss = Table.read('gaia_nss_hd158837.ecsv',        format='ascii.ecsv')[0]
gaia_src = Table.read('gaia_source_hd158837.ecsv',     format='ascii.ecsv')[0]

# RUWE and G magnitude from five-parameter solution (used for El-Badry inflation)
RUWE_5par = float(gaia_src['ruwe'])
G_mag     = float(gaia_src['phot_g_mean_mag'])

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

# ── Gaia Campbell elements (Appendix A, Halbwachs et al. 2023) ────────────────
# Semi-major axis (same formula as above, via u and v)
_u_G = (A_G**2 + B_G**2 + F_G**2 + G_G**2) / 2
_v_G = A_G*G_G - B_G*F_G
a0_G_camp = np.sqrt(_u_G + np.sqrt((_u_G+_v_G)*(_u_G-_v_G)))   # = a0_G

# omega+Omega and omega-Omega from arctan2 (Eq A.3)
_wp_Om = np.arctan2(B_G - F_G, A_G + G_G)
_wm_Om = np.arctan2(B_G + F_G, G_G - A_G)
_omega0 = (_wp_Om + _wm_Om) / 2
_Omega0 = (_wp_Om - _wm_Om) / 2

# Resolve quadrant: sin(omega+Omega) same sign as (B-F); sin(omega-Omega) same sign as (B+F)
omega_G = Omega_G = None
for k1 in [0, 1]:
    for k2 in [0, 1]:
        _om = _omega0 + k1*np.pi
        _Om = _Omega0 + k2*np.pi
        if (np.sin(_om+_Om)*(B_G-F_G) >= 0 and np.sin(_om-_Om)*(B_G+F_G) >= 0):
            omega_G = np.degrees(_om) % 360
            Omega_G = np.degrees(_Om) % 360
            break
    if omega_G is not None:
        break

# Inclination (Eq A.5-A.6)
_om_r = np.deg2rad(omega_G); _Om_r = np.deg2rad(Omega_G)
_d1 = abs((A_G + G_G) * np.cos(_om_r - _Om_r))
_d2 = abs((B_G - F_G) * np.sin(_om_r - _Om_r))
if _d1 >= _d2:
    inc_G = 2*np.degrees(np.arctan(np.sqrt(abs((A_G-G_G)*np.cos(_om_r+_Om_r)) / _d1)))
else:
    inc_G = 2*np.degrees(np.arctan(np.sqrt(abs((B_G+F_G)*np.sin(_om_r+_Om_r)) / _d2)))

# CHARA resolved the node ambiguity to the Omega+180 solution
Omega_G_resolved = (Omega_G + 180) % 360

# ── Parse CHARA table ──────────────────────────────────────────────────────────
with open('table_HD158837_Genet.txt') as f:
    lines = f.readlines()

def parse_chara_row(line):
    tok = line.split()
    return dict(label=(' '.join(tok[21:])), mjd=float(tok[1]),
                sep=float(tok[4]), pa=float(tok[5]),
                f2=float(tok[11]),
                diam1=float(tok[15]), diam2=float(tok[17]))

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

    # MC uncertainties on PA, sep, scale, and mass
    t_jd_row = row['mjd'] + 2400000.5
    _, pa_mc_mean, pa_mc_sig, _, sep_mc_sig, scale_mc_sig, M_mc, M_orb_sd, plx_nom = \
        mc_gaia_uncertainty(gaia_nss, t_jd_row, row['sep'])
    dpa_mc = ((pa_mc_mean - row['pa'] + 180) % 360) - 180

    # Parallax uncertainty: nominal NSS and El-Badry inflated
    sigma_plx_nom = float(gaia_nss['parallax_error'])
    f_EB          = el_badry_inflation(RUWE_5par, plx_nom)
    sigma_plx_EB  = f_EB * sigma_plx_nom
    # Parallax enters M as M ∝ plx^{-3}; add in quadrature with orbital contribution
    sigma_M_nom = np.sqrt(M_orb_sd**2 + (M_mc * 3 * sigma_plx_nom / plx_nom)**2)
    sigma_M_EB  = np.sqrt(M_orb_sd**2 + (M_mc * 3 * sigma_plx_EB  / plx_nom)**2)

    print(f"  {row['label']}:")
    print(f"    M = {M:.1f} deg,  E = {E:.1f} deg")
    print(f"    sep_photocentre (predicted) = {sep_ph:.3f} +/- {sep_mc_sig:.3f} mas")
    print(f"    sep_binary      (CHARA)     = {row['sep']:.3f} mas")
    print(f"    => a_rel/a0  = {scale:.3f} +/- {scale_mc_sig:.3f}  (no mass or flux-ratio assumption)")
    print(f"    => a_rel     = {a_rel:.2f} mas  =  {a_AU:.3f} AU")
    print(f"    => M1+M2     = {M_total:.2f} Msun")
    print(f"         sigma (orbital elements, MC)       = {M_orb_sd:.2f} Msun  ({M_orb_sd/M_total*100:.0f}%)")
    print(f"         sigma (+ nominal NSS plx)          = {sigma_M_nom:.2f} Msun  ({sigma_M_nom/M_total*100:.0f}%)")
    print(f"         sigma (+ El-Badry plx, f={f_EB:.2f})   = {sigma_M_EB:.2f} Msun  ({sigma_M_EB/M_total*100:.0f}%)")
    print(f"           [RUWE={RUWE_5par:.2f}, sigma_plx: {sigma_plx_nom:.3f} -> {sigma_plx_EB:.3f} mas]")
    print(f"    Photocentre PA = {pa_ph:.1f} deg  (MC mean: {pa_mc_mean:.1f} deg,  sigma: {pa_mc_sig:.1f} deg)")
    print(f"    PA (Omega sol., secondary opposite)  = {pa_opp:.1f} deg   DPA = {dpa_opp:+.1f} deg")
    print(f"    PA (Omega+180, secondary same dir)   = {pa_same:.1f} deg   DPA = {dpa_same:+.1f} deg  ({abs(dpa_same)/pa_mc_sig:.1f} sigma)")
    print(f"    Observed PA = {row['pa']:.3f} deg  => Omega+180 solution selected")
    print()

# ── Gaia Campbell elements summary (printed before mass-function section) ──────
print("─────────────────────────────────────────────────────────────────")
print("Gaia Campbell elements  (converted from Thiele-Innes, Appendix A)")
print("─────────────────────────────────────────────────────────────────")
print(f"  a0    = {a0_G:.4f} mas")
print(f"  omega = {omega_G:.2f} deg")
print(f"  Omega = {Omega_G:.2f} deg  (raw TI solution)")
print(f"  Omega = {Omega_G_resolved:.2f} deg  (CHARA-resolved, Omega+180 selected)")
print(f"  i     = {inc_G:.2f} deg")
print()
print("  Hipparcos Campbell elements (directly from catalogue):")
print(f"    a0    = {a0_Hip:.2f} mas")
print(f"    omega = {float(orb['w']):.2f} deg  (spectroscopic convention)")
print(f"    Omega = {Omega_Hip:.2f} deg")
print(f"    i     = {inc_Hip:.2f} deg  (sigma = 31.9 deg)")
print()
print(f"  Delta_i     = {inc_G - inc_Hip:+.1f} deg")
print(f"  Delta_Omega = {Omega_G_resolved - Omega_Hip:+.1f} deg  (CHARA-resolved Gaia vs Hipparcos)")
print(f"  Note: i_Gaia > 90 deg implies retrograde motion; Hipparcos has direct motion.")
print(f"        Difference is {abs(inc_G - inc_Hip)/31.9:.1f} sigma in Hipparcos alone.")

# ── Mass-function cross-check ──────────────────────────────────────────────────
print()
print("─────────────────────────────────────────────────────────────────")
print("Mass-function cross-check  (Lucke & Mayor 1982, Table 7)")
print("─────────────────────────────────────────────────────────────────")
print(f"  f(m) = {LM_f_mass:.3f} Msun,   a1 sin i = {LM_a1sini:.3e} km")

# Gaia a_rel from MIRC-X H-band
row0 = obs[0]
t_jd0 = row0['mjd'] + 2400000.5
x0, y0, _, _ = orbital_xy(t_jd0, P_G, T0_G, e_G)
sep_ph0 = np.hypot(A_G*x0+F_G*y0, B_G*x0+G_G*y0)
scale0  = row0['sep'] / sep_ph0
a_rel_G = scale0 * a0_G
a_AU_G  = a_rel_G / plx_G
M_tot_G = a_AU_G**3 / (P_G/365.25)**2

from scipy.optimize import brentq, brentq as _brentq

# ── Planck function (SI) ───────────────────────────────────────────────────────
def planck(lam_um, T):
    """Spectral radiance B_lambda at wavelength lam_um (microns) and temp T (K)."""
    h, c, k = 6.62607e-34, 2.99792e8, 1.38065e-23
    lam = lam_um * 1e-6
    return (2*h*c**2 / lam**5) / (np.exp(h*c / (lam*k*T)) - 1.0)

LAM_G, LAM_H, LAM_K = 0.590, 1.650, 2.200   # microns

print(f"  From Gaia orbit + CHARA: M1+M2 = {M_tot_G:.2f} Msun")
print()

for i_label, i_val in [('Hipparcos', inc_Hip), ('Gaia', inc_G)]:
    sin_i = np.sin(np.deg2rad(i_val))
    a1_km  = LM_a1sini / sin_i
    a1_mas = (a1_km / 1.495978707e8) * plx_G
    m2 = brentq(lambda m2: m2**3 * sin_i**3 / M_tot_G**2 - LM_f_mass,
                0.1, M_tot_G - 0.01)
    m1 = M_tot_G - m2
    print(f"  Using {i_label} i = {i_val:.2f} deg  (sin i = {sin_i:.4f}):")
    print(f"    a1 = {a1_mas:.2f} mas  (primary orbit, Gaia parallax)")
    print(f"    => m2 = {m2:.2f} Msun,  m1 = {m1:.2f} Msun")
    print(f"    => beta = {m2/M_tot_G:.3f},  f2_G implied = {m2/M_tot_G - a0_G/a_rel_G:.3f}")
    print()

# Retain m2, m1, sin_i from Hipparcos for the colour section that follows
sin_i = np.sin(np.deg2rad(inc_Hip))
m2 = brentq(lambda m2: m2**3 * sin_i**3 / M_tot_G**2 - LM_f_mass,
            0.1, M_tot_G - 0.01)
m1 = M_tot_G - m2

# ── CHARA colour analysis: f2, H/K colour ratio, and beta ─────────────────────
print()
print("=" * 65)
print("CHARA colour analysis: flux ratios, H/K colour, and beta")
print("=" * 65)
print("""
The photocentre equation links the mass ratio beta = m2/(m1+m2) to the
flux ratio f2_lambda and the relative/photocentre orbit at the same
wavelength:

    a0_lambda = (beta - f2_lambda) * a_rel
    =>  beta  =  f2_lambda + a0_lambda / a_rel

Gaia's a0 is measured in G-band (~590 nm); CHARA measures f2 in H and K.
The secondary was not cleanly resolved (diam2 is below the formal resolution
limit and is treated as unreliable); diam1 for the primary is well-determined.
T2 is therefore estimated from the f2_H/f2_K colour ratio alone, in which
the unknown (R2/R1)^2 cancels:

    f2_H/f2_K = [B_H(T2)/B_K(T2)] / [B_H(T1)/B_K(T1)]

This is independent of angular diameters.  The predicted f2_G then gives
beta without relying on the spectroscopic mass function or inclination.
""")

# Primary effective temperature: G3III ~ 5000 K
T1 = 5000.0

# Measured flux ratios
f2_H = obs[0]['f2']
f2_K = obs[1]['f2']
diam1_H = obs[0]['diam1']
diam1_K = obs[1]['diam1']

print(f"  CHARA measured flux ratios:")
print(f"    f2_H (MIRC-X) = {f2_H:.5f}")
print(f"    f2_K (MYSTIC) = {f2_K:.5f}")
print(f"  Primary angular diameters (reliable):")
print(f"    diam1_H = {diam1_H:.4f} mas,  diam1_K = {diam1_K:.4f} mas")
print(f"  Secondary angular diameters (unreliable — below resolution limit):")
print(f"    diam2_H = {obs[0]['diam2']:.4f} mas,  diam2_K = {obs[1]['diam2']:.4f} mas  [not used]")
print()
print(f"  Assumed T1 (G3III primary) = {T1:.0f} K")
print()

# T2 from H/K flux ratio (R2/R1 cancels)
T2_grid   = np.linspace(2000, 10000, 100000)
ratio_obs = f2_H / f2_K
ratio_T1  = planck(LAM_H, T1) / planck(LAM_K, T1)
ratio_grid = planck(LAM_H, T2_grid) / planck(LAM_K, T2_grid)
T2 = T2_grid[np.argmin(np.abs(ratio_grid - ratio_obs * ratio_T1))]

# Predict f2 ratios relative to f2_H (R2/R1 still cancels in ratios)
f2_G_over_fH = (planck(LAM_G, T2) / planck(LAM_H, T2)) * \
               (planck(LAM_H, T1) / planck(LAM_G, T1))
f2_K_over_fH = (planck(LAM_K, T2) / planck(LAM_H, T2)) * \
               (planck(LAM_H, T1) / planck(LAM_K, T1))
f2_G_pred = f2_H * f2_G_over_fH
f2_K_pred = f2_H * f2_K_over_fH

print(f"  T2 from f2_H/f2_K colour ratio: {T2:.0f} K")
print(f"  Predicted f2_K / f2_H = {f2_K_over_fH:.4f}  (observed: {f2_K/f2_H:.4f})")
print(f"  Predicted f2_G        = {f2_G_pred:.5f}")
print()

# a_rel from H-band CHARA (use nominal sep_ph)
t_jd_H = obs[0]['mjd'] + 2400000.5
x0, y0, _, _ = orbital_xy(t_jd_H, P_G, T0_G, e_G)
sep_ph_H = np.hypot(A_G*x0+F_G*y0, B_G*x0+G_G*y0)
a_rel_H  = obs[0]['sep'] * a0_G / sep_ph_H

beta_col = f2_G_pred + a0_G / a_rel_H
print(f"  beta = f2_G + a0_G/a_rel = {f2_G_pred:.5f} + {a0_G/a_rel_H:.4f} = {beta_col:.4f}")
print(f"  => m2 = {beta_col*M_tot_G:.2f} Msun,  m1 = {(1-beta_col)*M_tot_G:.2f} Msun")
print()

# Sensitivity to ±5% on the f2_H/f2_K ratio
print(f"  Sensitivity (±5% on f2_H/f2_K):")
for delta in [-0.05, 0.0, +0.05]:
    r   = ratio_obs * (1 + delta)
    T2t = T2_grid[np.argmin(np.abs(ratio_grid - r * ratio_T1))]
    fG  = f2_H * (planck(LAM_G,T2t)/planck(LAM_H,T2t)) * (planck(LAM_H,T1)/planck(LAM_G,T1))
    b   = fG + a0_G / a_rel_H
    print(f"    ratio x{1+delta:.2f}: T2={T2t:.0f} K,  f2_G={fG:.5f},  beta={b:.4f},  "
          f"m2={b*M_tot_G:.2f} Msun")
print()

print("  Summary — beta = m2/(m1+m2):")
print(f"    Mass function + Hipparcos i             = {m2/M_tot_G:.4f}")
print(f"    CHARA f2_H, no colour correction        = {f2_H + a0_G/a_rel_H:.4f}")
print(f"    CHARA H/K colour ratio (T2={T2:.0f} K)  = {beta_col:.4f}")
