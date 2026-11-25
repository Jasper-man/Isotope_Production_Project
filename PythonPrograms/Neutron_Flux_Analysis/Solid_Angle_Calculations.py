from abnormal_shape_SA import squash_beam as solid_angle
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
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
raw_beam_spots = [(float(row[0]), float(row[1])) for row in data]
poly = Polygon(raw_beam_spots)

# shift the centroid to (0,0) so the source aligns with the foil geometry
cx, cy = poly.centroid.x, poly.centroid.y
beam_spots = [(x - cx, y - cy) for x, y in raw_beam_spots]

R_Foil = 0.5029
angle = [0, 7, 10, 15, 20, 25, 30, 45, 60]
distance = [12.8, 11.8, 10.8, 9.8, 7.29, 5.78, 4.76, 3.672, 3.06]
omegas = []
omega_errs = []

results = []

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


"""
I am adding a plotting feature to see what the coordinates for the beam spot look like, I think that the issue is that the beam spot is off center and this is favoring
one side of the geometry. This could be based on how Danica plotted the pounts or it could indicate that the beam spot is off center, or both
"""

# beam_spot_csv = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/PythonPrograms/Neutron_Flux_Analysis/small_spot.csv'
# data = np.atleast_2d(np.genfromtxt(beam_spot_csv, delimiter=",", skip_header=1))
# beam_spots = [(float(row[0]), float(row[1])) for row in data]

# poly = Polygon(beam_spots)
# cx, cy = poly.centroid.x, poly.centroid.y
# centered_spots = [(x - cx, y - cy) for x, y in beam_spots]
# centered_poly = Polygon(centered_spots)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# # 1. Raw coordinates from CSV (showing vertex connection order)
# raw_x, raw_y = poly.exterior.xy
# ax1.fill(raw_x, raw_y, alpha=0.25, color='steelblue', label='Polygon Area')
# ax1.plot(raw_x, raw_y, '-o', color='steelblue', markersize=4, label='Perimeter Order')
# ax1.plot(raw_x[0], raw_y[0], 'go', markersize=8, label='Start Point (Row 0)')
# ax1.plot(cx, cy, 'rx', markersize=10, markeredgewidth=2, label=f'Centroid ({cx:.2f}, {cy:.2f})')
# ax1.set_title(f"Raw Beam Spot (Valid: {poly.is_valid}, Area: {poly.area:.3f} cm²)")
# ax1.set_xlabel("X (cm)")
# ax1.set_ylabel("Y (cm)")
# ax1.set_aspect('equal')
# ax1.grid(True, linestyle='--', alpha=0.5)
# ax1.legend(loc = 'upper left')

# # 2. Centered at (0, 0) for solid angle calculation
# cent_x, cent_y = centered_poly.exterior.xy
# ax2.fill(cent_x, cent_y, alpha=0.25, color='darkorange', label='Centered Spot')
# ax2.plot(cent_x, cent_y, '-o', color='darkorange', markersize=4)
# ax2.axhline(0, color='black', linewidth=0.8, linestyle=':')
# ax2.axvline(0, color='black', linewidth=0.8, linestyle=':')
# ax2.plot(0, 0, 'rx', markersize=10, markeredgewidth=2, label='Beam Axis (0, 0)')
# ax2.set_title("Centroid-Shifted Beam Spot")
# ax2.set_xlabel("X (cm)")
# ax2.set_ylabel("Y (cm)")
# ax2.set_aspect('equal')
# ax2.grid(True, linestyle='--', alpha=0.5)
# ax2.legend(loc = 'upper left')

# plt.tight_layout()
# plt.savefig("beam_spot_plot.png", dpi=200)
# plt.show()