import numpy as np

def calculate_bar_system(a, b, forces, distributed_loads, E, R, allowable_disp):
    """
    Calculate all parameters for the bar system with multiple forces and distributed loads
    
    Parameters:
    - a, b: geometry parameters (m)
    - forces: list of dicts, each with 'value', 'x_pos', 'y_pos' keys
    - distributed_loads: list of dicts, each with 'value', 'start_pos', 'end_pos' 'type' keys
    - E: Young's modulus (GPa)
    - R: Design resistance (MPa)
    - allowable_disp: Allowable displacement (mm)
    """
    results = {}
    
    # Calculate lengths and angles
    l1 = 2 * b
    l2 = np.sqrt((2*a)**2 + b**2)  # Using Pythagorean theorem
    sin_alpha = b / l2
    cos_alpha = 2*a / l2
    
    results['l1'] = l1
    results['l2'] = l2
    results['sin_alpha'] = sin_alpha
    results['cos_alpha'] = cos_alpha
    
    # Step 1: Forces in bar 1 (from moment equilibrium around A)
    # Sum the moment contributions from all loads
    moment_from_forces = 0
    for force in forces:
        if force['y_pos'] > 0:  # Force is above the x-axis
            direction = force.get('direction', [0, -1])
            fx = force['value'] * direction[0]
            fy = force['value'] * direction[1]
            
            # Moment = vertical component × horizontal distance - horizontal component × vertical distance
            moment_from_forces += (fy * force['x_pos'] - fx * force['y_pos'])
    
    moment_from_distributed_loads = 0
    for load in distributed_loads:
        if load['type'] == 'q1':  # Load on CD segment
            # Equivalent force at the middle of the segment
            equiv_force = load['value'] * (2 * a)
            moment_arm = a
            moment_from_distributed_loads += equiv_force * moment_arm
            
    # From moment equilibrium: N1 * a - moment_from_forces - moment_from_distributed_loads = 0
    N1 = (moment_from_forces + moment_from_distributed_loads) / a
    
    results['N1'] = N1
    results['N1_tension'] = N1 > 0
    
    # Step 2: Support reactions at A
    # Sum vertical forces (include all forces, not just y_pos>0)
    sum_vertical_forces = sum(
        force['value'] * force.get('direction', [0, -1])[1] 
        for force in forces
    )

    # Sum distributed loads (both q1 and q2)
    sum_distributed_q1 = sum(load['value'] * (2*a) for load in distributed_loads if load['type'] == 'q1')
    sum_distributed_q2 = sum(load['value'] * (2*a) for load in distributed_loads if load['type'] == 'q2')
    sum_distributed_loads = sum_distributed_q1 + sum_distributed_q2

    # Calculate horizontal reaction at A
    sum_horizontal_forces = sum(
        force['value'] * force.get('direction', [0, -1])[0] 
        for force in forces
    )
    Xa = -sum_horizontal_forces
    results['Xa'] = Xa

    # Vertical reaction at A:
    #    Ya + N1 - (sum of vertical forces + sum of distributed loads) = 0
    Ya = sum_vertical_forces + sum_distributed_loads - N1
    results['Ya'] = Ya
    
    # Step 3: Forces in bar 2
    # From moment equilibrium around B for element II
    moment_B_from_forces = 0.0
    moment_B_from_distributed_loads = 0.0
    
    for force in forces:
        direction = force.get('direction', [0, -1])
        fx = force['value'] * direction[0]
        fy = force['value'] * direction[1]
        # Accumulate moments around B
        moment_B_from_forces += fy * (2*a - force['x_pos']) - fx * force['y_pos']
        
    for load in distributed_loads:
        if load['type'] == 'q2':
            equiv_force = load['value'] * (2*a)
            moment_arm = a  # Center of distributed load is at a distance 'a' from B
            moment_B_from_distributed_loads += equiv_force * moment_arm
    
    # Now use them in the formula for N2
    N2 = (moment_B_from_forces + moment_B_from_distributed_loads - N1*(2*a)) / (sin_alpha*(2*a))
    results['N2'] = N2
    
    results['N2_tension'] = N2 > 0
    
    # Step 4: Support reactions at B
    # Calculate from global equilibrium
    sum_vertical_forces = sum(force['value'] * force.get('direction', [0, -1])[1] for force in forces)

    # Horizontal equilibrium: Xa + Xb + N2 * cos_alpha = 0
    Xb = -results['Xa'] - N2 * cos_alpha
    results['Xb'] = Xb

    # Vertical equilibrium: Ya + Yb + sum of vertical forces - N2 * sin_alpha + sum of distributed loads = 0
    Yb = -Ya - sum_vertical_forces + N2 * sin_alpha + sum_distributed_loads
    results['Yb'] = Yb
    
    # Step 5: Bar cross-section areas and diameters (same as before)
    A1_m2 = abs(N1 * 1000) / (R * 1e6)  # m²
    A2_m2 = abs(N2 * 1000) / (R * 1e6)  # m²
    
    A1_cm2 = A1_m2 * 10000  # cm²
    A2_cm2 = A2_m2 * 10000  # cm²
    
    d1 = np.sqrt(4 * A1_cm2 / np.pi)  # cm
    d2 = np.sqrt(4 * A2_cm2 / np.pi)  # cm
    
    results['A1_cm2'] = A1_cm2
    results['A2_cm2'] = A2_cm2
    results['d1'] = d1
    results['d2'] = d2
    
    # Step 6: Stiffness check (same as before)
    delta_l1 = (N1 * 1000 * l1) / (E * 1e9 * A1_m2)  # m
    delta_l2 = (N2 * 1000 * l2) / (E * 1e9 * A2_m2)  # m
    
    # Convert to mm for display
    delta_l1_mm = delta_l1 * 1000
    delta_l2_mm = delta_l2 * 1000
    
    results['delta_l1'] = delta_l1_mm
    results['delta_l2'] = delta_l2_mm
    
    # Maximum displacement at point C
    delta_C = np.sqrt(delta_l1_mm**2 + (delta_l2_mm * sin_alpha)**2)
    results['delta_C'] = delta_C
    
    return results
