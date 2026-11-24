from abnormal_shape_SA import squash_beam as solid_angle
import numpy as np
from tqdm import tqdm

"""
I am going to go through the solid angles for the foil packs and establish a value so that we can determine the relationship between the
neutron flux and the solid angle
"""

# Now we can approximate the solid angle from a disk to diak approximation which will be converted to a geometric flux conversion factor
beam_spot_csv = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/PythonPrograms/Neutron_Flux_Analysis/small_spot.csv'
data = np.genfromtxt(beam_spot_csv, delimiter=",", skip_header=1)

# Convert array to a list of tuples
beam_spots = [tuple(row) for row in data]

R_Foil = 0.5029
angle = [0, 7, 10, 15, 20, 25, 30, 45, 60]
distance = [12.8, 11.8, 10.8, 9.8, 7.29, 5.78, 4.76, 3.672, 3.06]
omegas = []
omega_errs = []

results = []

# print(distance[0])
# print(beam_spots)

for d, ang in tqdm(zip(distance,angle), total=len(angle)):
    omega,omega_err,_,_ = solid_angle(beam_spots, R_Foil, d, ang)
    omegas.append(omega)
    omega_errs.append(omega_err)
    results.append((ang,d,omega,omega_err))

# This will be for the current run with the PLA foil holder

f_distance = 1.5
f_angle = 0
f_omega,f_omega_err,_,_ = solid_angle(beam_spots,R_Foil,f_distance,f_angle)
results.append((f_angle,f_distance,f_omega,f_omega_err))


# Print aligned table
header = f"{'Angle (deg)':>12} | {'Distance (cm)':>14} | {'Omega (sr)':>14} | {'Error (sr)':>14}"
print("\n" + header)
print("-" * len(header))
for ang, dist, omega, err in results:
    print(f"{ang:>12.1f} | {dist:>14.3f} | {omega:>14.6e} | {err:>14.6e}")