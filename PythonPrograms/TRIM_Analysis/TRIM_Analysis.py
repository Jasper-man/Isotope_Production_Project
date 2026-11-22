import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARSE THE LEGACY OUTPUT FILE
# ==========================================
file_path = "/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/PythonPrograms/TRIM_Analysis/TRANSMIT_10000.txt"

# Add "Type" to the beginning of the list to handle the 'T' column
columns = ["Type", "Ion_Num", "Z", "Energy_eV", "Depth", "Y", "Z_coord", "CosX", "CosY", "CosZ"]

try:
    # Use sep='\s+' to fix the delimiter warning, and skip exactly 12 header lines
    df = pd.read_csv(file_path, sep='\s+', skiprows=12, names=columns, engine='python')
    df = df.dropna() # Drops any trailing footer text
    
    # Filter strictly for transmitted protons
    protons = df[(df["Z"] == 1.0) & (df["Type"] == 'T')].copy()
    print(f"Successfully loaded {len(protons)} transmitted protons.")
except Exception as e:
    print(f"Error reading file: {e}")

# ==========================================
# 2. PHYSICS CONVERSIONS
# ==========================================
# Convert final exit energy from eV to MeV
protons["Energy_MeV"] = protons["Energy_eV"] / 1e6

# Convert the forward directional cosine back into a physical polar angle
protons["Angle_deg"] = np.degrees(np.arccos(protons["CosX"]))

# ==========================================
# 3. BINNING AND SOLID-ANGLE NORMALIZATION
# ==========================================
# Set up a grid: 1 MeV energy bins, 2-degree angular bins
energy_bins = np.arange(0, 25, 1.0) 
angle_bins = np.arange(0, 45, 2.0)  

# Sort every proton into the 2D grid
raw_counts, xedges, yedges = np.histogram2d(
    protons["Energy_MeV"], 
    protons["Angle_deg"], 
    bins=[energy_bins, angle_bins]
)

# Solid Angle Correction (dOmega = 2 * pi * sin(theta) * dTheta)
# Particles at wider angles are spread over a larger conical area. We must divide 
# the raw counts by this physical area to get a true representation of intensity.
angle_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
dTheta_rad = np.radians(2.0) # Our 2-degree bin width in radians

solid_angles = 2 * np.pi * np.sin(np.radians(angle_centers)) * dTheta_rad

# Normalize the grid (divide each angular column by its respective solid angle)
normalized_yield = raw_counts / solid_angles

# ==========================================
# 4. GENERATE THE HEATMAP
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

# Transpose the yield matrix (.T) so Energy is on the X-axis and Angle is on the Y-axis
c = ax.pcolormesh(xedges, yedges, normalized_yield.T, cmap='turbo', shading='auto')

ax.set_title("Double-Differential Proton Spectrum Exiting 4mm Beryllium", pad=15)
ax.set_xlabel("Exiting Proton Energy (MeV)")
ax.set_ylabel("Exit Angle (Degrees)")
fig.colorbar(c, label="Yield Density (Protons / MeV / Steradian)")

plt.tight_layout()
plt.show()