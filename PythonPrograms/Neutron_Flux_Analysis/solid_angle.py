import numpy as np
from scipy.integrate import nquad

# NOTE: These default radii for the target and the source are a combination of the measurement from our target fois and from the Be targer from ther Breede paper
def tilted_solid_angle(d, alpha_degrees, Rs = 1, Rt = 0.5):
    """
    Calculates the exact average solid angle of a tilted circular target disk 
    as seen from a coaxial circular source disk, with ZERO far-distance approximations.
    
    Parameters:
    - d : float : Distance between the centers of the disks (cm)
    - alpha_degrees : float : Tilt angle of the target disk (degrees)
    - Rs : float : Radius of the emitting source disk (cm)
    - Rt : float : Radius of the target foil (cm)
    
    Returns:
    - Omega_avg : float : Average solid angle in steradians (sr)
    """
    # Convert tilt angle to radians
    alpha = np.radians(alpha_degrees)
    
    # 1. Define the 4D Integrand based on our derivation
    # Variables: phi (target angle), r (target radius), theta_s (source angle), rho (source radius)
    def integrand(phi, r, theta_s, rho):
        
        # Numerator: dot product of R vector and target normal
        num = r * rho * (d * np.cos(alpha) - rho * np.cos(theta_s) * np.sin(alpha))
        
        # Denominator: Distance R^2 between the two points
        R_squared = (r**2 + rho**2 + d**2 
                     - 2 * d * r * np.cos(phi) * np.sin(alpha) 
                     - 2 * r * rho * (np.cos(phi) * np.cos(alpha) * np.cos(theta_s) 
                                      + np.sin(phi) * np.sin(theta_s)))
        
        # Return projected area / R^3
        return num / (R_squared**1.5)

    # 2. Set the integration boundaries for [phi, r, theta_s, rho]
    boundaries = [
        [0, 2 * np.pi],  # phi
        [0, Rt],         # r
        [0, 2 * np.pi],  # theta_s
        [0, Rs]          # rho
    ]
    
    # 3. Perform the 4D numerical integration
    # Note: epsrel is slightly relaxed to 1e-4 so it calculates in seconds rather than minutes
    # print(f"Calculating exact 4D integral for tilt = {alpha_degrees}° (This may take a few seconds)...")
    integral_result, error = nquad(integrand, boundaries, opts={'epsrel': 1e-4, 'epsabs': 1e-4})
    
    # 4. Divide by the total Area of the Source to get the average
    source_area = np.pi * (Rs**2)
    target_area = np.pi * (Rt**2)
    Omega_avg = integral_result / source_area

    G_factor = Omega_avg / target_area
    
    return Omega_avg, G_factor

# ==========================================
# Example Usage:
# ==========================================
# distance = 5.0      # cm
# source_R = 1.0      # cm
# target_R = 0.5      # cm

# # Calculate for a parallel disk (0 degrees)
# omega_parallel = tilted_solid_angle(distance, 0, source_R, target_R)
# print(f"Parallel Solid Angle  : {omega_parallel:.6f} sr")

# # Calculate for a disk tilted by 45 degrees
# omega_tilted = tilted_solid_angle(distance, 45, source_R, target_R)
# print(f"45° Tilt Solid Angle  : {omega_tilted:.6f} sr")