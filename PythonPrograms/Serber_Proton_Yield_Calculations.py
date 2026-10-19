import numpy as np
from numpy import trapezoid as trapz

def calculate_modified_serber_yield(E_d, E_p, theta_deg, amu=9.0, z=4.0):
    # Deuteron binding energy in MeV
    E_B = 2.225

    # 1. Incoming Deuteron Coulomb Barrier
    E_c_in = (z * 197 / 137.0) / (1.57 + 1.25 * amu**0.333) - 0.9
    
    # THRESHOLD FIX: Deuteron must overcome BOTH the Coulomb barrier and binding energy
    if E_d <= (E_c_in + E_B):
        return 0.0
        
    # 2. Outgoing Proton Energy Limits (Coulomb minimum & Conservation of Energy maximum)
    E_c_out = 1.5 
    if E_p <= E_c_out or E_p >= (E_d - E_B):
        return 0.0

    # 3. Kalbach Total Cross-Section (Using outer Beryllium defaults)
    def kalbach_xs(E_d, amu, A=69.9E-3, w=26.6):
        mu = 22.3
        return A * (amu**0.333 + 1.26) / (1.0 + np.exp((mu - E_d) / w))

    # 4. Serber Energy Probability (Symmetric direct breakup)
    def P_En(E_p, E_d, E_c):
        w = 1.0 * 0.385 * E_B
        # Subtract binding energy from available kinetic energy
        eps = E_d - E_c - E_B
        x = eps * w / ((E_p - 0.5 * eps)**2 + eps * w)**1.5
        sm = 1.0 + eps / (eps**2 + 4.0 * eps * w)**0.5
        return x / sm

    # 5. Serber Angular Probability
    def P_tht(tht, E_d, E_c):
        eps = E_d - E_c - E_B
        tht_0 = 1.0 * 0.812 * np.sqrt(E_B / eps) * (1.0 - (eps / 8E3))
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

# Define the physical limit of the target
target_thickness_cm = 0.4  # 4mm Beryllium sheet

# Note: We added depth_cm to the zip here so we can track when it exits the block
for E_d, depth_cm in zip(deuteron_energies, depth_array):
    # Stop calculating if the deuteron passes through the sheet
    if depth_cm > target_thickness_cm:
        break
        
    for E_p in proton_energy_grid:
        for theta_deg in angular_grid:
            # Add the raw physics probability of this specific bin to the grand total
            total_calculated_yield += calculate_modified_serber_yield(E_d, E_p, theta_deg)

print(f"Total Raw Calculated Yield: {total_calculated_yield}")

# Conceptual loop to generate TRIM.DAT
total_simulated_particles = 100000 
yield_scaling_factor = total_simulated_particles / total_calculated_yield 

# ==========================================
# 4. GENERATE TRIM.DAT FILE
# ==========================================

print("Writing TRIM.DAT file...")
particles_written = 0

with open("TRIM.DAT", "w", newline='\r\n') as file:
    # TRIM requires exactly 10 lines of header comments
    file.write("1- Custom Proton Simulation\n")
    file.write("2- Generated via Python\n")
    file.write("3- \n")
    file.write("4- \n")
    file.write("5- \n")
    file.write("6- \n")
    file.write("7- \n")
    file.write("8- Protons into Beryllium\n")
    file.write("9- Data Table line consist of: EventName(5 char)+8 numbers\n")
    file.write("10-Name  Z  Energy(eV)  X(A)  Y(A)  Z(A)  CosX  CosY  CosZ\n")
    
    target_thickness_A = target_thickness_cm * 1e8

    for depth_cm, E_d in zip(depth_array, deuteron_energies):
        # Stop generating protons if the deuteron is outside the sheet
        if depth_cm > target_thickness_cm:
            break
            
        depth_angstroms = depth_cm * 1e8  
        
        # FIX 1: Shift the start position 1.0 Angstrom deep to avoid the X=0.0 boundary bug
        safe_depth = depth_angstroms + 1.0 
        
        # FIX 3: Back-wall bug prevention. Do not spawn exactly on the back boundary.
        if safe_depth >= target_thickness_A:
            safe_depth = target_thickness_A - 1.0
            
        for E_p in proton_energy_grid:
            E_p_eV = E_p * 1e6  
            
            for theta_deg in angular_grid:
                bin_yield = calculate_modified_serber_yield(E_d, E_p, theta_deg)
                num_particles = int(bin_yield * yield_scaling_factor)
                
                for _ in range(num_particles):
                    cosX, cosY, cosZ = calculate_directional_cosines(theta_deg)
                    
                    # FIX 2: Strip out "negative zeros" (-0.00000) that crash legacy VB6
                    cosX = 0.0 if abs(cosX) < 1e-6 else cosX
                    cosY = 0.0 if abs(cosY) < 1e-6 else cosY
                    cosZ = 0.0 if abs(cosZ) < 1e-6 else cosZ
                    
                    row = f"H     1 {E_p_eV:.1f} {safe_depth:.1f} 0.0 0.0 {cosX:.5f} {cosY:.5f} {cosZ:.5f}\n"
                    file.write(row)
                    particles_written += 1

print(f"Success! {particles_written} protons written to TRIM.DAT.")