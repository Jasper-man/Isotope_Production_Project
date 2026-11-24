import numpy as np
from shapely.geometry import Polygon, Point

def squash_beam(
    polygon_coords, 
    r_foil, 
    distance, 
    angle_deg, 
    z_focal=0.0, 
    x_quadrant='+x', 
    num_samples=100_000
):
    """
    Calculates solid angle subtended by a target foil disc when its normal vector 
    points to a focal point (0, 0, z_focal) on the Z-axis, rather than the origin.

    Parameters:
    -----------
    polygon_coords : list of tuples [(x1, y1), ...]
        Vertices of the non-standard source shape in the xy-plane (z = 0).
    r_foil : float
        Radius of the target foil disc (cm).
    distance : float
        Distance (d0) from origin to the center of the foil disc (cm).
    angle_deg : float
        Angle (degrees) between +Z axis and foil center line-of-sight.
    z_focal : float, optional
        Z-coordinate of the focal point on the Z-axis that the foil normal points to.
        Defaults to 0.0 (origin).
    x_quadrant : str, optional
        '+x' or '-x' quadrant for foil position.
    num_samples : int, optional
        Number of Monte Carlo source points to sample inside the shape.

    Returns:
    --------
    omega_avg : float
        Average solid angle (steradians) across the source shape.
    error : float
        Estimated statistical error.
    source_area : float
        Area of the source polygon.
    """
    # 1. Create Shapely Polygon for the source, creates a valid area where the rejection samples will be kept
    poly = Polygon(polygon_coords)
    source_area = poly.area
    minx, miny, maxx, maxy = poly.bounds

    # 2. Rejection Sampling for uniform source points inside polygon, randomly picks points within the square, keeping only those that lie within the polygon
    points_x, points_y = [], []
    batch_size = num_samples * 2

    # This will continue to go theough points as long as the sample size is not met
    while len(points_x) < num_samples:
        rx = np.random.uniform(minx, maxx, batch_size)
        ry = np.random.uniform(miny, maxy, batch_size)
        
        for x, y in zip(rx, ry):
            if poly.contains(Point(x, y)):
                points_x.append(x)
                points_y.append(y)
                if len(points_x) == num_samples:
                    break

    # Source points array (N_src, 3) at z = 0
    # This generates an array of points on the source plane (z=0) that can later have vectors associated with them that point to the target foil
    Ps = np.column_stack([points_x[:num_samples], points_y[:num_samples], np.zeros(num_samples)])

    # 3. Geometry Setup with Focal Point Target (0, 0, z_focal)
    quadrant_sign = 1.0 if x_quadrant.lower() in ['+x', 'pos', 'positive', '+'] else -1.0
    theta = np.radians(abs(angle_deg))

    """
    This part of the code sets up a localized axis for the target foil based on the normal vector made between the center of the target foil and the
    center of the global coordinate system at (0,0,0) accomplished by taking the difference between the P_focal point and the local canter Pf_center.
    using this normal vector we can utilize the properties of the cross product to construct a vector that must be orthogonal to the normal vector and
    some arbitrary vector (the x-axis if the normal vector is not close to the x-axis and the y-axis if the normal vector is). This vector is normalized
    and called u_foil. A second vector that will represent the localized y-axis of the localized 2D surface can then be created using the n_vector and
    u_foil.

    """
    # Center position of foil
    Pf_center = np.array([
        quadrant_sign * distance * np.sin(theta), 
        0.0, 
        distance * np.cos(theta)
    ])

    # Focal Point on the Z-axis
    P_focal = np.array([0.0, 0.0, z_focal])

    # Foil Normal pointing from Pf_center to P_focal
    v_aim = P_focal - Pf_center
    n_foil = v_aim / np.linalg.norm(v_aim)

    # Construct orthonormal basis (u_foil, v_foil) on the foil surface
    if np.abs(n_foil[0]) < 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])
    else:
        arbitrary = np.array([0.0, 1.0, 0.0])

    u_foil = np.cross(n_foil, arbitrary)
    u_foil /= np.linalg.norm(u_foil)
    v_foil = np.cross(n_foil, u_foil)

    """
    The next part of the code generates a corresponding list of points on the target foil, utilizing the new local axis. 
    First, 1000 random points that are within the radius of the target foil at some position within the 2pi radians of the disc
    (This is scaled with a sqrt to prevent numerical clumping). These points then go through matrix multiplication with the u_vector
    and the v_vector to create a list of vectors that point from the center of our foil to the location of our 1000 random points
    These points are then coupled to the global geometry with the sum of the local pointing vector and the global pointing vector from
    the global orgin to the center of the foil then from there to our individual 1000 random points.
    """
    # 4. Integrate over Target Foil Disc (Sub-sampling)
    N_foil_samples = 1000
    rf = r_foil * np.sqrt(np.random.rand(N_foil_samples))
    theta_f = 2 * np.pi * np.random.rand(N_foil_samples)

    # Foil surface points (N_foil_samples, 3)
    Pf_offset = np.outer(rf * np.cos(theta_f), u_foil) + np.outer(rf * np.sin(theta_f), v_foil)
    Pf = Pf_center + Pf_offset

    """
    This part of the program preforms the solid angle calculations by determining the magnitude of the vector that points from the source to the target
    at every point. The dot product is then used to determine the angle between the incoming r_vec and the normal vector of the target to determine how
    much of our 'infinitesimal' area (Afoil/Nsamples) is 'visible' to the source plate by scaling the relationship between the angle in between these
    two vectors. These individual components are added together and then averaged to get the generalized solid angle of the entire setup.
    """

    # 5. Compute Solid Angle Contributions
    omega_per_source_point = np.zeros(num_samples)
    foil_area = np.pi * (r_foil**2)

    for i in range(num_samples):
        # Vector pointing FROM foil element TO source element
        r_vec = Ps[i] - Pf  # Shape: (N_foil_samples, 3)
        r_mag = np.linalg.norm(r_vec, axis=1)

        dot_prods = np.dot(r_vec, n_foil)
        dot_prods[dot_prods < 0] = 0.0  # Backside check

        d_omega = (dot_prods / (r_mag**3)) * (foil_area / N_foil_samples)
        omega_per_source_point[i] = np.sum(d_omega)

    omega_avg = np.mean(omega_per_source_point)
    error = np.std(omega_per_source_point) / np.sqrt(num_samples)

    """
    The average solid angle tells us how much of the total neutron field, at some distance, is being taken up by the target, if we want to instead determine
    what the solid angle is per unit of the target surface area we divide by the total target surface area. This value can now be used woth flux calculations
    """

    G = omega_avg / foil_area

    return omega_avg, error, source_area, G


# # ==========================================
# # Test Run & Comparison
# # ==========================================
# if __name__ == "__main__":
#     # Mock Wisconsin-shaped boundary profile
#     wisconsin_outline = [
#         (-2.0,  2.0), ( 1.0,  2.0), ( 1.5,  1.5),
#         ( 2.0, -1.0), ( 2.0, -2.5), (-1.5, -2.5), (-2.0,  0.0)
#     ]

#     R_foil = 1.0            # Foil radius in cm
#     d0 = 12.82949134        # Distance in cm
#     angle = 30.0            # 30° off-axis tilt
#     z = 0.629491342         # h' - h in cm

#     # Compare aiming at origin (z_focal = 0) vs. aiming further up the Z-axis (z_focal = 5.0 cm)
#     omega, _, _ = wisconsin_source_to_offset_foil(
#         wisconsin_outline, R_foil, d0, angle, z_focal=z
#     )

#     print(f"--- Aiming Parameter Comparison ---")
#     print(f"Aiming at Origin (z_focal = 0.0 cm) : {omega:.6f} sr")
