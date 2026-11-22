import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy import trapezoid as trapz


def get_experimental_cross_section(csv_filepath, energy_col='Energy', xs_col='CrossSection'):
    """
    Reads experimental cross-section data from a CSV, plots the linear interpolation,
    and returns a callable function to evaluate the cross-section at any energy.
    """
    # 1. Load and sort the data
    df = pd.read_csv(csv_filepath)
    df = df.sort_values(by=energy_col)
    
    energies = df[energy_col].values
    cross_sections = df[xs_col].values
    
    # 2. Create the linear interpolation function
    # bounds_error=False and fill_value="extrapolate" prevent crashes if 
    # the simulation requests an energy slightly outside your dataset bounds.
    interpolated_function = interp1d(
        energies, 
        cross_sections, 
        kind='linear', 
        bounds_error=False, 
        fill_value=(cross_sections[0], cross_sections[-1]) # Flat extrapolation beyond data
    )
    
    # 3. Generate a dense grid for smooth plotting
    plot_energies = np.linspace(energies.min(), energies.max(), 500)
    plot_cs = interpolated_function(plot_energies)
    
    # 4. Plot the results
    plt.figure(figsize=(8, 5))
    plt.scatter(energies, cross_sections, color='red', label='Experimental Data', zorder=5)
    plt.plot(plot_energies, plot_cs, color='blue', linestyle='--', label='Linear Interpolation')
    
    plt.title('Experimental Cross Section vs. Energy')
    plt.xlabel('Deuteron Energy (MeV)')
    plt.ylabel('Cross Section (barns/mb)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return interpolated_function


def calculate_deuteron_depth_profile(E_inc=33.0, E_min=1.0, dE=1.0, rho=1.848, amu=9.0, z=4.0):
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
    energies = np.arange(E_inc, E_min - 0.0001, -dE)
    
    # 4. Calculate stopping power at every energy step
    stopping_power = dEdx(energies, n, z)
    
    # 5. Integrate to find depth
    inverse_stopping = 1.0 / stopping_power
    depths = np.zeros_like(energies)
    
    # Cumulative trapezoidal integration
    for i in range(1, len(energies)):
        avg_inv_S = 0.5 * (inverse_stopping[i-1] + inverse_stopping[i])
        dx = dE * avg_inv_S 
        depths[i] = depths[i-1] + dx
        
    return energies, depths, stopping_power

# --- 1. Deuteron Energy & Depth Arrays ---
E_inc = 33.0   
E_min = 2.0    
dE_d = 0.5     

deuteron_energies, depth_array, stopping_power = calculate_deuteron_depth_profile(E_inc, E_min, dE_d)

# --- 2. Calculate Probabilities Along the Path ---
target_thickness_cm = 0.4  # Assume all breakup happend in 4mm beryllium sheet
valid_energies = []
valid_depths = []
probabilities = []
sigma_function = get_experimental_cross_section('/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/PythonPrograms/TRIM_Analysis/d_Be_xs.csv', 'EN-MEAN (EV) 1.1', 'DATA (B) 0.1')

# Collect valid points and calculate weights[cite: 3]
for i in range(len(deuteron_energies)):
    E_d = deuteron_energies[i]
    depth_cm = depth_array[i]
    
    # Stop calculating if the deuteron passes through the sheet
    if depth_cm > target_thickness_cm:
        break
        
    # Calculate distance traveled in this step (dx)
    if i == 0:
        dx = depth_array[1] - depth_array[0]
    else:
        dx = depth_array[i] - depth_array[i-1]
        
    # Probability is proportional to cross-section * distance traveled
    sigma = sigma_function(E_d)
    weight = sigma * dx
    
    valid_energies.append(E_d)
    valid_depths.append(depth_cm)
    probabilities.append(weight)

# Normalize probabilities so they sum to 1.0
probabilities = np.array(probabilities)
probabilities /= probabilities.sum()

# --- 3. Sample Collision Locations ---
total_simulated_particles = 10000

# Randomly sample the indices based on the calculated probability distribution
sampled_indices = np.random.choice(
    len(valid_depths), 
    size=total_simulated_particles, 
    p=probabilities
)

# Create the final arrays of collision locations and their corresponding energies
sampled_depths = np.array(valid_depths)[sampled_indices]
sampled_energies = np.array(valid_energies)[sampled_indices]

print(f"Generated arrays for {total_simulated_particles} collision events.")
print(f"Average collision depth: {sampled_depths.mean():.4f} cm")
print(f"Average deuteron energy at collision: {sampled_energies.mean():.2f} MeV")

# ==========================================
# 4. GENERATE TRIM.DAT FILE
# ==========================================
print("Writing TRIM.DAT file...")
particles_written = 0

with open("TRIM.DAT", "w", newline='\r\n') as file:
    # TRIM requires exactly 10 lines of header comments[cite: 3]
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
    
    target_thickness_A = target_thickness_cm * 1e8 # convert to angstroms

    # Iterate directly through our probabilistically sampled arrays
    for depth_cm, E_d in zip(sampled_depths, sampled_energies):
        depth_angstroms = depth_cm * 1e8  
        
        # FIX 1: Shift the start position 1.0 Angstrom deep to avoid the X=0.0 boundary bug
        safe_depth = depth_angstroms + 1.0 
        
        # FIX 3: Back-wall bug prevention. Do not spawn exactly on the back boundary.
        if safe_depth >= target_thickness_A:
            safe_depth = target_thickness_A - 1.0
            
        E_p_eV = (E_d + 4.587) * 1e6 # all of the reaction energy is given to the proton
        cosX, cosY, cosZ = 0.0, 0.0, 0.0 # protons are completly forward focused
        
        row = f"H     1 {E_p_eV:.1f} {safe_depth:.1f} 0.0 0.0 {cosX:.5f} {cosY:.5f} {cosZ:.5f}\n"
        file.write(row)
        particles_written += 1   

print(f"Success! {particles_written} protons written to TRIM.DAT.")





"""
This section was testing code for my own understanding of the process involved and working out the best way to represent the probabilities of the breakup.
I am leaving it here for future reference but it is not needed for the program to run
"""

# I need to make an array that looks at all of the depths and tabulates the probabilities of the deuteron breaking up there, normalizing it to 1

# dx = 0.01 # units of mm
# x = np.arange(0,4+dx,dx)
# x_probability = np.zeros(len(x))
# x_energy = np.zeros(len(x))
# dE_loss = np.zeros(len(x))

# # print(len(x))
# # print(len(x_probability))

# # The total macroscopic cross section
# sig_t = 0.000148 # this is in terms of mm
# dE = 0.0246684 # this is in terms of MeVcm^2/mg
# dE_cm = dE * 1848 # I convert to energy loss per cm
# dE_mm = dE_cm / 10 # finally get the energy loss in terms of mm

# # Function to calculate dynamic stopping power (MeV/mm) based on current energy
# def get_dE_mm(E_current):
#     # Material constants for Be
#     z = 4.0
#     amu = 9.0
#     rho = 1.848
#     n = rho * 6.022E-1 / amu
#     I = 11.2 + 11.7 * z if z < 14 else 52.8 + 8.71 * z
    
#     # Bethe-Bloch (returns MeV/cm)
#     beta2 = 1.0 - (1.0 / (1.0 + (E_current / 1.8756E3))**2)
#     F = np.log(1.02E6 * beta2 / (1.0 - beta2)) - beta2
#     S_cm = 0.508 * (z * n) * (F - np.log(I)) / beta2
    
#     # Convert MeV/cm to MeV/mm
#     return S_cm / 10.0


# def probability(E0 = 33, dE = dE_mm):
    
#     # loop through the distance probability and fill the array with the probability at each point
#     # We need to start the loop at the initial energy and distance and track the energy loss as we move through the material
#     current_energy = E0
#     for d in range(len(x_probability)):
#         x_probability[d] = sig_t * dx
#         # Tabulate the Deuteron energy at each point in the Be
#         x_energy[d] = current_energy

#         if current_energy != 0:
#             # change the current energy to reflect the energy loss in dx
#             current_dE = get_dE_mm(current_energy) * dx
#             dE_loss[d]= current_dE
#             # print(current_dE)
#             current_energy -= current_dE

#         # make a catch look if E < 0
#         if current_energy < 0:
#             current_energy = 0

#     # Find the total probability and normalize it to the length of the Be plate
#     prob_tot = np.sum(x_probability)

#     x_prob_normal = x_probability / prob_tot

#     return x_prob_normal

# x_prob_normal = probability()

# print(dE_loss)
# print(x)

