# HD 158837 / HIP 85749 — Astrometric Orbit Analysis

Comparison of the Hipparcos and Gaia DR3 astrometric orbital solutions for the
single-lined spectroscopic binary HD 158837 (G3III + companion, P ≈ 419 d)
against a CHARA interferometric observation obtained on 2025 June 02.

## Background

HD 158837 is a member of a triple system.  The spectroscopic binary component
has an orbital period of ~418 d and was first characterised by Lucke & Mayor
(1982, A&A 105, 318), who derived radial-velocity orbital elements from CORAVEL
observations.  The Hipparcos mission (ESA 1997) added an astrometric orbital
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

The Hipparcos orbit (PA residual ≈ 95°, implied M₁+M₂ ≈ 12 M☉) is poorly
constrained — its inclination and ascending node have formal errors of ±32°
and ±20° respectively — and is not suitable for orbital scale calibration.

## Dependencies

```
astropy
astroquery
numpy
scipy
```
