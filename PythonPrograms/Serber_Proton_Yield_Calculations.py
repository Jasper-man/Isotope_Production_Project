import numpy as np
from numpy import trapezoid as trapz

def calculate_modified_serber_yield(E_d, E_p, theta_deg, amu=9.0, z=4.0):
    # 1. Incoming Deuteron Coulomb Barrier (from original script)
    E_c_in = (z * 197 / 137.0) / (1.57 + 1.25 * amu**0.333) - 0.9
    
    # If the deuteron energy is below the barrier, no breakup occurs
    if E_d <= E_c_in:
        return 0.0
        
    # 2. Outgoing Proton Coulomb Barrier (Approximate for Beryllium)
    # Protons must overcome this to escape the parent nucleus
    E_c_out = 1.5 
    if E_p <= E_c_out:
        return 0.0

    # 3. Kalbach Total Cross-Section (Using outer Beryllium defaults)
    def kalbach_xs(E_d, amu, A=69.9E-3, w=26.6):
        mu = 22.3
        return A * (amu**0.333 + 1.26) / (1.0 + np.exp((mu - E_d) / w))

    # 4. Serber Energy Probability (Symmetric direct breakup)
    def P_En(E_p, E_d, E_c):
        w = 1.0 * 0.385 * 2.225
        eps = E_d - E_c
        x = eps * w / ((E_p - 0.5 * eps)**2 + eps * w)**1.5
        sm = 1.0 + eps / (eps**2 + 4.0 * eps * w)**0.5
        return x / sm

    # 5. Serber Angular Probability
    def P_tht(tht, E_d, E_c):
        tht_0 = 1.0 * 0.812 * np.sqrt(2.225 / (E_d - E_c)) * (1.0 - ((E_d - E_c) / 8E3))
        x = tht_0 / (tht_0**2 + tht**2)**1.5
        sm = np.pi / (tht_0 * np.sqrt(tht_0**2 + np.pi**2))
        return x / sm

    # 6. Calculate discrete probabilities
    sigma_K = kalbach_xs(E_d, amu)
    prob_E = P_En(E_p, E_d, E_c_in)
    prob_tht = P_tht(theta_deg * np.pi / 180.0, E_d, E_c_in)
    
    # 7. Total unnormalized yield for this specific bin
    bin_yield = sigma_K * prob_E * prob_tht
    
    return bin_yield

def calculate_deuteron_depth_profile(E_inc=40.0, E_min=1.0, dE=1.0, rho=1.85, amu=9.0, z=4.0):
    # 1. Target number density in atoms/barn-cm
    n = rho * 6.022E-1 / amu
    
    # 2. Bethe-Bloch stopping power (MeV/cm) from original script
    def dEdx(E_d, n, z):
        I = 11.2 + 11.7 * z if z < 14 else 52.8 + 8.71 * z
        beta2 = 1.0 - (1.0 / (1.0 + (E_d / 1.8756E3))**2)
        F = np.log(1.02E6 * beta2 / (1.0 - beta2)) - beta2
        S = 0.508 * (z * n) * (F - np.log(I)) / beta2
        return np.where(S > 5, S, 5.0)

    # 3. Create energy grid stepping BACKWARDS from E_inc down to E_min
    # The deuteron enters at E_inc (depth 0) and slows down.
    energies = np.arange(E_inc, E_min - 0.0001, -dE)
    
    # 4. Calculate stopping power at every energy step
    stopping_power = dEdx(energies, n, z)
    
    # 5. Integrate to find depth
    # dx = dE * (1 / stopping_power)
    inverse_stopping = 1.0 / stopping_power
    depths = np.zeros_like(energies)
    
    # Cumulative trapezoidal integration
    for i in range(1, len(energies)):
        # Average the inverse stopping power between step i-1 and i
        avg_inv_S = 0.5 * (inverse_stopping[i-1] + inverse_stopping[i])
        
        # dE is implicitly positive here because we want absolute distance traveled
        dx = dE * avg_inv_S 
        depths[i] = depths[i-1] + dx
        
    return energies, depths, stopping_power

# Example Execution:
# E_inc = 33.0 # MeV incident deuteron
# E_min = 2.0  # MeV Coulomb cutoff estimate
# dE = 0.1     # Fine step size for accurate integration
# energies, depths, stopping = calculate_deuteron_depth_profile(E_inc, E_min, dE)

# Print the first few steps
# print(f"{'Energy (MeV)':<15} | {'Stopping Power (MeV/cm)':<25} | {'Depth (cm)':<15}")
# print("-" * 60)
# for i in range(0, len(energies), int(5.0/dE)): # Print every 5 MeV
#     print(f"{energies[i]:<15.2f} | {stopping[i]:<25.2f} | {depths[i]:<15.4f}")

def P_tht(tht, E_d, E_c, R_tht=1.0):
		tht_0 = R_tht*0.812*np.sqrt(2.225/(E_d-E_c))*(1.0-((E_d-E_c)/8E3))

		x = tht_0/(tht_0**2+tht**2)**1.5
		sm = np.pi/(tht_0*np.sqrt(tht_0**2+np.pi**2))
		return x/sm


def calculate_directional_cosines(theta_deg):
    """
    Converts a polar observation angle into 3D directional cosines for SRIM.
    Assuming the beam travels perfectly forward along the X-axis.
    """
    # 1. Convert the polar angle (theta) from degrees to radians
    theta_rad = np.radians(theta_deg)
    
    # 2. Breakup is cylindrically symmetric, so we randomly generate 
    # the azimuthal angle (phi) anywhere between 0 and 2*pi radians.
    phi_rad = np.random.uniform(0, 2 * np.pi)
    
    # 3. Calculate the directional cosines
    # X is the forward depth axis into the target
    cosX = np.cos(theta_rad)
    
    # Y and Z define the lateral spray of the particle
    cosY = np.sin(theta_rad) * np.cos(phi_rad)
    cosZ = np.sin(theta_rad) * np.sin(phi_rad)
    
    return cosX, cosY, cosZ

# Example Execution:
# theta_input = 15.0 # A proton emitted at a 15-degree angle
# cX, cY, cZ = calculate_directional_cosines(theta_input)

# print(f"Polar Angle: {theta_input} degrees")
# print(f"CosX (Forward): {cX:.5f}")
# print(f"CosY (Lateral): {cY:.5f}")
# print(f"CosZ (Lateral): {cZ:.5f}")
# # Verification: The sum of the squares must exactly equal 1.0
# print(f"Vector Magnitude: {cX**2 + cY**2 + cZ**2:.1f}")


# --- 1. Deuteron Energy & Depth Arrays ---
# Define the incident beam and how finely you want to track its stopping path.
E_inc = 33.0   # Incident deuteron energy in MeV
E_min = 2.0    # Stopping cutoff (roughly the entrance Coulomb barrier)
dE_d = 0.5     # Integration step size in MeV (smaller = higher spatial resolution)

# Call the depth function we built earlier to generate these matched arrays
deuteron_energies, depth_array, stopping_power = calculate_deuteron_depth_profile(E_inc, E_min, dE_d)

# --- 2. Proton Energy Grid ---
# Define the possible energies of the emitted protons.
E_p_min = 1.5    # Minimum energy to escape the Beryllium Coulomb barrier
E_p_max = 33.0   # Maximum theoretical proton energy (roughly matches E_inc)
dE_p = 0.5       # Bin size for proton energies in MeV

proton_energy_grid = np.arange(E_p_min, E_p_max + dE_p, dE_p)

# --- 3. Angular Grid ---
# Define the observation angles relative to the forward beam axis.
tht_min = 0.0    # Straight forward
tht_max = 90.0   # Serber breakup is forward-peaked; backward emission (>90) is negligible
d_tht = 2.0      # Bin size in degrees

angular_grid = np.arange(tht_min, tht_max + d_tht, d_tht)

# Run a pre-computation loop over all grids
total_calculated_yield = 0.0

for E_d in deuteron_energies:
    for E_p in proton_energy_grid:
        for theta_deg in angular_grid:
            
            # Add the raw physics probability of this specific bin to the grand total
            total_calculated_yield += calculate_modified_serber_yield(E_d, E_p, theta_deg)

print(f"Total Raw Calculated Yield: {total_calculated_yield}")

# Conceptual loop to generate TRIM.DAT
total_simulated_particles = 100000 
yield_scaling_factor = total_simulated_particles / total_calculated_yield 

with open("TRIM.DAT", "w") as file:
    # Write the SRIM header (SRIM requires a specific header format)
    file.write("==> SRIM-2013 TRIM.DAT File <==\n")
    file.write("Ion_E(eV) X(A) Y(A) Z(A) CosX CosY CosZ\n")
    
    # Loop over your calculated depth/deuteron energy array
    for depth_cm, E_d in zip(depth_array, deuteron_energies):
        
        # Loop over possible emitted proton energies
        for E_p in proton_energy_grid:
            
            # Loop over your angles
            for theta_deg in angular_grid:
                
                # 1. Calculate the physics yield for this specific bin
                # (This is where P_tht does its work inside the math)
                bin_yield = calculate_modified_serber_yield(E_d, E_p, theta_deg)
                
                # 2. Determine how many protons to simulate for this bin
                num_particles = int(bin_yield * yield_scaling_factor)
                
                # 3. Write individual particles to the file
                for _ in range(num_particles):
                    # Get 3D aiming vectors for this specific angle
                    cosX, cosY, cosZ = calculate_directional_cosines(theta_deg)
                    
                    # Convert units for SRIM (MeV to eV, cm to Angstroms)
                    E_p_eV = E_p * 1e6
                    depth_angstroms = depth_cm * 1e8
                    
                    # Write the row: Energy, X, Y, Z, CosX, CosY, CosZ
                    row = f"{E_p_eV:.1f} {depth_angstroms:.1f} 0.0 0.0 {cosX:.5f} {cosY:.5f} {cosZ:.5f}\n"
                    file.write(row)



