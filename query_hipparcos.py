from astroquery.vizier import Vizier
from astropy.table import Table

HIP = 85749

v = Vizier(columns=["**"], row_limit=10)

# Main Hipparcos catalogue entry
hip_main = v.query_constraints(catalog="I/239/hip_main", HIP=str(HIP))
if hip_main:
    t = hip_main[0]
    t.meta["QUERY"] = f"I/239/hip_main HIP={HIP}"
    t.write(f"hipparcos_main_hip{HIP}.ecsv", format="ascii.ecsv", overwrite=True)
    print(f"Written hipparcos_main_hip{HIP}.ecsv  ({len(t)} row)")
else:
    print(f"No result in I/239/hip_main for HIP {HIP}")

# DMSA/O — orbital solution
hip_orb = v.query_constraints(catalog="I/239/hip_dm_o", HIP=str(HIP))
if hip_orb:
    t = hip_orb[0]
    t.meta["QUERY"] = f"I/239/hip_dm_o HIP={HIP}"
    t.write(f"hipparcos_orbit_hip{HIP}.ecsv", format="ascii.ecsv", overwrite=True)
    print(f"Written hipparcos_orbit_hip{HIP}.ecsv  ({len(t)} row)")
else:
    print(f"No result in I/239/hip_dm_o for HIP {HIP}")
