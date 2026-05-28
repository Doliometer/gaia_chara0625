# HD 158837 / HIP 85749 — Astrometric Orbit Analysis

Comparison of the Hipparcos and Gaia DR3 astrometric orbital solutions for
HD 158837 (G3III, single-lined spectroscopic binary, P ≈ 419 d) against a
CHARA interferometric observation obtained on 2025 June 02.

## Background

HD 158837 is a spectroscopic binary with an orbital period of ~418 d, first
characterised by Lucke & Mayor (1982, A&A 105, 318) from CORAVEL radial-velocity
observations.  An earlier claim of a wider visual companion (Aitken 1911;
WDS 17314+0243) is now considered erroneous: all speckle interferometry
observations since 1976 (Hartkopf, Tokovinin and others) return only upper
limits at separations < 0.04 arcsec, and many of the original visual measures
were already flagged as uncertain or erroneous by subsequent observers.  The Hipparcos mission (ESA 1997) added an astrometric orbital
solution (DMSA/O), but this fixed the period, eccentricity, and argument of
periastron directly from Lucke & Mayor and fitted only the inclination, ascending
node, and photocentre semi-major axis to the Hipparcos abscissae.  The Gaia DR3
Non-Single Star (NSS) catalogue provides a fully independent astrometric orbital
solution from 443 along-scan measurements, with no spectroscopic priors.

## Data files

| File | Contents |
|---|---|
| `gaia_nss_hd158837.ecsv` | Gaia DR3 NSS Orbital solution (Thiele-Innes elements, P, e, T₀, parallax) |
| `gaia_source_hd158837.ecsv` | Gaia DR3 main source catalogue entry |
| `hipparcos_orbit_hip85749.ecsv` | Hipparcos DMSA/O orbital solution (I/239/hip\_dm\_o) |
| `hipparcos_main_hip85749.ecsv` | Hipparcos main catalogue entry (I/239/hip\_main) |
| `table_HD158837_Genet.txt` | CHARA/MIRC-X (H-band) and MYSTIC (K-band) interferometric observation, 2025 Jun 02; columns: date, MJD, HJD, JY, sep (mas), PA (deg), σ\_maj, σ\_min, σ\_PA, f1, σ\_f1, f2, σ\_f2, f3, σ\_f3, diam1 (mas), σ\_diam1, diam2 (mas), σ\_diam2, χ²\_V2, χ²\_CP |

The Lucke & Mayor (1982) spectroscopic elements used in `hipparcos_ephemeris.py`
are hardcoded as constants (the paper is not redistributed here):

- a₁ sin i = 94.8 × 10⁶ km
- f(m) = 0.195 M☉

## Scripts

### `query_hipparcos.py`
Queries VizieR (via astroquery) for the Hipparcos main catalogue entry and
DMSA/O orbital solution and writes them as ECSV files.

### `hipparcos_ephemeris.py`
Computes the predicted sky position of the secondary at the CHARA epoch from
both the Hipparcos and Gaia orbits, and compares with the CHARA measurement.

### `chara_spectral_f2.py`
Fits f₂ and diam1 jointly per spectral channel from the MIRC-X (H-band) and
MYSTIC (K-band) OIFITS files, using the correct resolved-primary V² model.
Derives T₂ from the H/K weighted-mean flux ratio (excluding CO-contaminated
K-band channels at 2.27–2.35 μm).  Produces `chara_f2_per_channel.png`.

**Key result:** the CHARA binary separation at a known orbital phase directly
determines the scale of the Gaia astrometric orbit without any flux-ratio or
mass assumption:

```
a_rel / a₀  =  sep_binary (CHARA)  /  sep_photocentre (Gaia TI prediction)
```

From the Gaia orbit + CHARA (H-band):

| Quantity | Value |
|---|---|
| a_rel / a₀ | 3.48 |
| a_rel | 13.8 mas = 1.72 AU |
| M₁ + M₂ | 3.8 M☉  (Kepler's 3rd law, Gaia parallax) |

The CHARA position angle also resolves the 180° ascending-node ambiguity
inherent in purely astrometric orbits, selecting the Ω + 180° solution
(ΔPA ≈ −17°) over the Ω solution (ΔPA ≈ +163°).

**Caution — near-degeneracy in the Gaia orbital solution.**
The low eccentricity (e ≈ 0.10) means that periastron is geometrically
ill-defined, creating a near-degeneracy between the time of periastron T₀
and the Thiele-Innes orientation elements.  This is reflected in the Gaia
`corr_vec`: the correlations corr(T₀, A) = +0.998, corr(T₀, F) = −0.990,
and corr(T₀, G) = −0.992 are close to ±1, indicating that the fit can trade
off *when* periastron occurs against *how the orbit is oriented* with almost
no change in the astrometric residuals.  As a result, the marginal
uncertainty on T₀ alone (σ ≈ 21 d, 5% of the period) overstates the
true timing uncertainty when the TI elements are held fixed, and the
marginal uncertainty on individual TI elements overstates their geometric
uncertainty.  A Monte Carlo propagation through the full covariance matrix
gives σ_PA ≈ 17° at the CHARA epoch — much smaller than the ~35° obtained
by adding TI and T₀ contributions in quadrature — and places the observed
17° PA residual at < 1 σ.  Interpretations of the Gaia orbital elements
(e.g. implied inclination or ascending node) should account for this
covariance structure rather than treating the parameter errors independently.
The Gaia DR3 NSS documentation notes the same degeneracy for near-circular
orbits and applies a pseudo-circular fix below e < 0.0005; HD 158837
(e ≈ 0.10) is well above that threshold and receives no automatic mitigation,
so the problem is present but silent.

**Hipparcos vs Gaia orbit direction.**
Converting the Gaia Thiele-Innes elements to Campbell form (Appendix A of
Halbwachs et al. 2023) gives i_Gaia = 119.8°, while Hipparcos gives
i_Hip = 76.3° (σ = 31.9°).  Because i > 90° corresponds to retrograde
orbital motion, the two solutions disagree not only in the inclination value
(Δi = +43.5°) but in the sense of the orbit.  A Monte Carlo propagation
through the full Gaia covariance matrix yields σ_i(Gaia) ≈ 1.8°, so the
two solutions are ~24σ apart in Gaia's uncertainty — a genuine contradiction
rather than a marginal tension.  The Hipparcos DMSA/O fit for this star fixed
e and ω from Lucke & Mayor and fitted only three parameters (i, Ω, a₀) to
relatively few abscissae, so its formal σ_i = 31.9° is likely underestimated.
The Gaia solution, based on 443 along-scan measurements with no spectroscopic
priors, is almost certainly more reliable on inclination.

The Hipparcos orbit (PA residual ≈ 95°, implied M₁+M₂ ≈ 12 M☉) is poorly
constrained — its inclination and ascending node have formal errors of ±32°
and ±20° respectively — and is not suitable for orbital scale calibration.

**Conclusions on individual stellar temperatures and masses.**
The CHARA H- and K-band flux ratios constrain the secondary temperature via
the H/K colour ratio, in which the unknown angular diameter ratio (R₂/R₁)
cancels.  Assuming T₁ = 5000 K for the G3III primary, the best-fit secondary
temperature is T₂ ≈ 5000 ± 700 K (statistical), suggesting a late-G or early-K
subgiant or dwarf companion.  This estimate is based on a per-channel fit of
the binary V² model to the OIFITS data (`chara_spectral_f2.py`), which fits
f₂ and diam1 jointly per spectral channel using the correct resolved-primary
formula; the broad-band Genet values (f₂_H = 0.041, f₂_K = 0.040) give a
consistent but slightly higher estimate of ≈ 5550 K.  Note that the secondary
angular diameter (diam2) returned by the CHARA fit is below the formal
resolution limit and is treated as unreliable; diam1 for the primary (≈ 0.94
mas in H, ≈ 0.96 mas in K) is well-determined and consistent across channels.

The per-channel K-band fit reveals a drop in f₂ at 2.27–2.35 μm (f₂ falling
from ≈ 0.040 to < 0.010), coinciding with the CO 2-0 band head at 2.293 μm.
These channels are excluded from the H/K temperature comparison; only the
continuum region 2.06–2.23 μm is used for K-band.

Three estimates of the mass ratio β = m₂/(m₁+m₂) are derived:

| Method | β | m₂ (M☉) | m₁ (M☉) |
|---|---|---|---|
| Spectroscopic f(m) + Hipparcos i = 76.3° | 0.381 | 1.46 | 2.37 |
| Spectroscopic f(m) + Gaia i = 119.8° | 0.427 | 1.64 | 2.20 |
| CHARA H/K colour → f₂_G (no inclination needed) | 0.325 | 1.24 | 2.59 |

All three are consistent within the ~55% uncertainty on M_total from the Gaia
orbital solution.  The colour-based estimate is the most model-independent,
requiring neither the spectroscopic mass function nor an inclination, though
it does assume a primary temperature and blackbody spectra.  A ±5% systematic
on the f₂_H/f₂_K ratio shifts T₂ by ~1000 K and β by ~0.03, so the
cross-instrument flux calibration between MIRC-X and MYSTIC remains the
dominant uncertainty.  The per-channel analysis tightens the statistical error
on the band-averaged f₂ values (to ±0.001) but does not reduce this systematic,
which arises from calibrator-star and pipeline differences between the two
instruments.

## Dependencies

```
astropy
astroquery
numpy
scipy
```
