import numpy as np

def calculate_optimal_diameters(N1, N2, l1, l2, sin_alpha, E, R, allowable_disp):
    """
    Calculate optimal bar diameters that satisfy both strength and stiffness requirements
    
    Parameters:
    - N1, N2: Forces in bars (kN)
    - l1, l2: Lengths of bars (m)
    - sin_alpha: Sine of angle between bar 2 and horizontal
    - E: Young's modulus (GPa)
    - R: Design resistance (MPa)
    - allowable_disp: Allowable displacement (mm)
    
    Returns:
    - Dictionary containing optimal diameters and areas for both bars
    """
    # Step 1: Calculate minimum diameters based on strength requirements
    A1_m2_min = abs(N1 * 1000) / (R * 1e6)  # m²
    A2_m2_min = abs(N2 * 1000) / (R * 1e6)  # m²
    
    # Convert to cm² for easier handling
    A1_min = A1_m2_min * 10000  # cm²
    A2_min = A2_m2_min * 10000  # cm²
    
    # Calculate minimum diameters
    d1_min = np.sqrt(4 * A1_min / np.pi)  # cm
    d2_min = np.sqrt(4 * A2_min / np.pi)  # cm
    
    # Step 2: Check if minimum diameters meet stiffness requirement
    delta_l1 = (N1 * 1000 * l1) / (E * 1e9 * A1_m2_min)  # m
    delta_l2 = (N2 * 1000 * l2) / (E * 1e9 * A2_m2_min)  # m
    
    # Convert to mm for consistency
    delta_l1_mm = delta_l1 * 1000
    delta_l2_mm = delta_l2 * 1000
    
    # Calculate maximum displacement at point C with minimum areas
    delta_C = np.sqrt(delta_l1_mm**2 + (delta_l2_mm * sin_alpha)**2)
    
    # Step 3: If displacement is already within allowable limit
    if delta_C <= allowable_disp:
        return {
            "d1_optimal": d1_min,
            "d2_optimal": d2_min,
            "A1_optimal": np.pi * (d1_min/2)**2,
            "A2_optimal": np.pi * (d2_min/2)**2,
            "is_stiffness_critical": False,
            "delta_C": delta_C,
            "material_volume": (np.pi * (d1_min/2)**2) * l1 * 0.01 
                               + (np.pi * (d2_min/2)**2) * l2 * 0.01,
            "optimization_method": "No additional scaling needed"
        }
    
    # Step 4: Implement optimization strategy based on displacement contribution analysis
    # Calculate contribution of each bar to total displacement
    contrib1 = delta_l1_mm**2 / delta_C**2
    contrib2 = (delta_l2_mm * sin_alpha)**2 / delta_C**2
    
    # Calculate scaling factors needed for each bar individually
    k1 = delta_C / allowable_disp if contrib1 > 0 else 1.0
    k2 = delta_C / allowable_disp if contrib2 > 0 else 1.0
    
    # Option 1: Scale both bars proportionally (baseline)
    scale_both = delta_C / allowable_disp
    d1_both = d1_min * np.sqrt(scale_both)
    d2_both = d2_min * np.sqrt(scale_both)
    A1_both = np.pi * (d1_both/2)**2
    A2_both = np.pi * (d2_both/2)**2
    vol_both = A1_both * l1 * 0.01 + A2_both * l2 * 0.01  # in liters
    
    # Option 2: Scale bar 1 only
    scale1 = k1 * delta_C / allowable_disp
    d1_only = d1_min * np.sqrt(scale1)
    A1_only = np.pi * (d1_only/2)**2
    
    # Check if this configuration meets the displacement constraint
    delta_l1_new = delta_l1_mm / scale1
    delta_C_new = np.sqrt(delta_l1_new**2 + (delta_l2_mm * sin_alpha)**2)
    
    if delta_C_new <= allowable_disp:
        vol_option2 = A1_only * l1 * 0.01 + A2_min * l2 * 0.01
    else:
        # If scaling bar 1 alone isn't sufficient, this option is invalid
        vol_option2 = float('inf')
    
    # Option 3: Scale bar 2 only
    scale2 = k2 * delta_C / allowable_disp
    d2_only = d2_min * np.sqrt(scale2)
    A2_only = np.pi * (d2_only/2)**2
    
    # Check if this configuration meets the displacement constraint
    delta_l2_new = delta_l2_mm / scale2
    delta_C_new = np.sqrt(delta_l1_mm**2 + (delta_l2_new * sin_alpha)**2)
    
    if delta_C_new <= allowable_disp:
        vol_option3 = A1_min * l1 * 0.01 + A2_only * l2 * 0.01
    else:
        # If scaling bar 2 alone isn't sufficient, this option is invalid
        vol_option3 = float('inf')
    
    # Option 4: Weighted scaling based on contribution
    total_contrib = contrib1 + contrib2
    weight1 = contrib1 / total_contrib if total_contrib > 0 else 0.5
    weight2 = contrib2 / total_contrib if total_contrib > 0 else 0.5
    
    scale_factor = delta_C / allowable_disp
    
    # Apply weighted scaling, with higher scale to the bar that contributes more
    d1_weighted = d1_min * np.sqrt(1 + (scale_factor - 1) * weight1 * 1.5)
    d2_weighted = d2_min * np.sqrt(1 + (scale_factor - 1) * weight2 * 1.5)
    
    A1_weighted = np.pi * (d1_weighted/2)**2
    A2_weighted = np.pi * (d2_weighted/2)**2
    
    # Verify this approach meets the displacement constraint
    delta_l1_weighted = delta_l1_mm * (A1_min / A1_weighted)
    delta_l2_weighted = delta_l2_mm * (A2_min / A2_weighted)
    delta_C_weighted = np.sqrt(delta_l1_weighted**2 + (delta_l2_weighted * sin_alpha)**2)
    
    if delta_C_weighted <= allowable_disp:
        vol_option4 = A1_weighted * l1 * 0.01 + A2_weighted * l2 * 0.01
    else:
        # Apply additional scaling to ensure constraint is met
        extra_scale = delta_C_weighted / allowable_disp
        d1_weighted *= np.sqrt(extra_scale)
        d2_weighted *= np.sqrt(extra_scale)
        A1_weighted = np.pi * (d1_weighted/2)**2
        A2_weighted = np.pi * (d2_weighted/2)**2
        vol_option4 = A1_weighted * l1 * 0.01 + A2_weighted * l2 * 0.01
    
    # Choose the option with minimum material volume
    volumes = [vol_both, vol_option2, vol_option3, vol_option4]
    min_index = np.argmin(volumes)
    
    if min_index == 0:  # Proportional scaling
        return {
            "d1_optimal": d1_both,
            "d2_optimal": d2_both,
            "A1_optimal": A1_both,
            "A2_optimal": A2_both,
            "is_stiffness_critical": True,
            "delta_C": allowable_disp,
            "material_volume": vol_both,
            "optimization_method": "Proportional scaling of both bars"
        }
    elif min_index == 1:  # Scale bar 1 only
        return {
            "d1_optimal": d1_only,
            "d2_optimal": d2_min,
            "A1_optimal": A1_only,
            "A2_optimal": A2_min,
            "is_stiffness_critical": True,
            "delta_C": allowable_disp,
            "material_volume": vol_option2,
            "optimization_method": "Scaling bar 1 only"
        }
    elif min_index == 2:  # Scale bar 2 only
        return {
            "d1_optimal": d1_min,
            "d2_optimal": d2_only,
            "A1_optimal": A1_min,
            "A2_optimal": A2_only,
            "is_stiffness_critical": True,
            "delta_C": allowable_disp,
            "material_volume": vol_option3,
            "optimization_method": "Scaling bar 2 only"
        }
    else:  # Weighted scaling
        return {
            "d1_optimal": d1_weighted,
            "d2_optimal": d2_weighted,
            "A1_optimal": A1_weighted,
            "A2_optimal": A2_weighted,
            "is_stiffness_critical": True,
            "delta_C": allowable_disp,
            "material_volume": vol_option4,
            "optimization_method": "Weighted scaling based on displacement contribution"
        }
