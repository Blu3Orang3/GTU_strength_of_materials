import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import uuid

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
    # Sum vertical forces: Ya + N1 - sum of vertical forces - sum of distributed loads = 0
    sum_vertical_forces = sum(force['value'] * force.get('direction', [0, -1])[1] 
                         for force in forces if force['y_pos'] > 0)
    sum_distributed_q1 = sum(load['value'] * (2*a) for load in distributed_loads if load['type'] == 'q1')

    # Calculate horizontal reaction at A (needed for complete equilibrium)
    sum_horizontal_forces = sum(force['value'] * force.get('direction', [0, -1])[0] 
                           for force in forces)
    Xa = -sum_horizontal_forces  # Horizontal reaction at support A
    results['Xa'] = Xa

    Ya = sum_vertical_forces + sum_distributed_q1 - N1
    results['Ya'] = Ya
    
    # Step 3: Forces in bar 2
    # From moment equilibrium around B for element II
    moment_B_from_forces = 0
    for force in forces:
        if force['x_pos'] < 2*a:  # Force before point B
            direction = force.get('direction', [0, -1])
            fx = force['value'] * direction[0]
            fy = force['value'] * direction[1]
            
            # Vertical component creates moment around B
            moment_B_from_forces += fy * (2*a - force['x_pos'])
            
            # If the force has a horizontal component and is not at y=0
            if fx != 0 and force['y_pos'] > 0:
                moment_B_from_forces -= fx * force['y_pos']
    
    moment_B_from_distributed_loads = 0
    for load in distributed_loads:
        if load['type'] == 'q1':
            equiv_force = load['value'] * (2*a)
            moment_arm = a  # Distance from the middle of CD to B horizontally
            moment_B_from_distributed_loads += equiv_force * moment_arm
        elif load['type'] == 'q2':
            equiv_force = load['value'] * (2*a)
            moment_arm = a  # Distance from the middle of DB to B horizontally
            moment_B_from_distributed_loads += equiv_force * moment_arm
    
    # From moment equilibrium: N2 * sin_alpha * (2*a) - moment_B_from_forces - moment_B_from_distributed_loads + N1 * (2*a) = 0
    N2 = (moment_B_from_forces + moment_B_from_distributed_loads - N1 * (2*a)) / (sin_alpha * (2*a))
    
    results['N2'] = N2
    results['N2_tension'] = N2 > 0
    
    # Step 4: Support reactions at B
    # Calculate from global equilibrium
    sum_vertical_forces = sum(force['value'] * force.get('direction', [0, -1])[1] for force in forces)
    sum_distributed_loads = sum_distributed_q1 + sum(load['value'] * (2*a) for load in distributed_loads if load['type'] == 'q2')

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
    
    # Step 3: If displacement exceeds allowable, optimize diameters
    if delta_C <= allowable_disp:
        # Strength requirements already satisfy stiffness
        return {
            "d1_optimal": d1_min,
            "d2_optimal": d2_min,
            "A1_optimal": A1_min,
            "A2_optimal": A2_min,
            "is_stiffness_critical": False,
            "delta_C": delta_C,
            "material_volume": A1_min * l1 * 0.01 + A2_min * l2 * 0.01  # in liters
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

def draw_bar_system_with_multiple_loads(a, b, forces, distributed_loads, N1, N2, disp_scale, d1, d2):
    """Draw the bar system with deformation visualization and multiple loads"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate node positions
    A_x, A_y = 0, 0
    B_x, B_y = 2*a, 0
    C_x, C_y = 0, 2*b
    D_x, D_y = 2*a, b
    
    # Calculate bar properties
    l1 = np.sqrt((D_x - C_x)**2 + (D_y - C_y)**2)
    l2 = np.sqrt((D_x - A_x)**2 + (D_y - A_y)**2)
    
    sin_alpha = b / l2
    cos_alpha = 2*a / l2
    
    # Calculate bar elongations
    E = 210  # GPa
    A1 = np.pi * (d1/2)**2  # cm²
    A2 = np.pi * (d2/2)**2  # cm²
    
    delta_l1 = (N1 * 1000 * l1) / (E * 1e9 * A1 * 1e-4) * 1000  # mm
    delta_l2 = (N2 * 1000 * l2) / (E * 1e9 * A2 * 1e-4) * 1000  # mm
    
    # Scale for visualization
    delta_l1_scaled = delta_l1 * disp_scale / 1000  # m
    delta_l2_scaled = delta_l2 * disp_scale / 1000  # m
    
    # Calculate deformed positions
    dx_C = 0
    dy_C = delta_l1_scaled if N1 > 0 else -delta_l1_scaled
    
    dx_D = delta_l2_scaled * cos_alpha if N2 > 0 else -delta_l2_scaled * cos_alpha
    dy_D = delta_l2_scaled * sin_alpha if N2 > 0 else -delta_l2_scaled * sin_alpha
    
    C_x_def, C_y_def = C_x + dx_C, C_y + dy_C
    D_x_def, D_y_def = D_x + dx_D, D_y + dy_D
    
    # Draw undeformed system (solid lines)
    ax.plot([A_x, C_x], [A_y, C_y], 'k-', linewidth=2)
    ax.plot([C_x, D_x], [C_y, D_y], 'r-', linewidth=max(1, d1/10), label='Bar 1')
    ax.plot([D_x, B_x], [D_y, B_y], 'k-', linewidth=2)
    ax.plot([A_x, D_x], [A_y, D_y], 'b-', linewidth=max(1, d2/10), label='Bar 2')
    
    # Draw deformed system (dashed lines)
    ax.plot([A_x, C_x_def], [A_y, C_y_def], 'k--', linewidth=1)
    ax.plot([C_x_def, D_x_def], [C_y_def, D_y_def], 'r--', linewidth=1)
    ax.plot([D_x_def, B_x], [D_y_def, B_y], 'k--', linewidth=1)
    ax.plot([A_x, D_x_def], [A_y, D_y_def], 'b--', linewidth=1)
    
    # Draw multiple forces with custom directions
    arrow_length = 0.2 * max(a, b)
    for i, force in enumerate(forces):
        x_pos = force['x_pos']
        y_pos = force['y_pos']
        value = force['value']
        direction = force.get('direction', [0, -1])  # Default downward if not specified
        
        # Calculate arrow components
        dx = direction[0] * arrow_length
        dy = direction[1] * arrow_length
        
        # Draw arrow in the specified direction
        ax.arrow(x_pos, y_pos, dx, dy, head_width=0.05*max(a, b),
                head_length=0.05*max(a, b), fc='g', ec='g', width=0.02*max(a, b))
        
        # Position the text label based on arrow direction to avoid overlap
        text_offset_x = 0.1*a if direction[0] <= 0 else -0.3*a
        text_offset_y = 0.1*b if direction[1] <= 0 else -0.3*b
        
        # Add direction indicator to label
        direction_str = ""
        if direction[0] == 0 and direction[1] == -1:
            direction_str = "↓"
        elif direction[0] == 0 and direction[1] == 1:
            direction_str = "↑"
        elif direction[0] == 1 and direction[1] == 0:
            direction_str = "→"
        elif direction[0] == -1 and direction[1] == 0:
            direction_str = "←"
        
        ax.text(x_pos+text_offset_x, y_pos+text_offset_y, 
               f'F{i+1}={value} kN {direction_str}', fontsize=10)
    
    # Draw distributed loads
    q_scale = 0.15 * max(a, b)
    for i, load in enumerate(distributed_loads):
        value = load['value']
        
        if load['type'] == 'q1':  # Load on CD segment
            start_x, start_y = C_x, C_y
            end_x, end_y = D_x, D_y
        else:  # q2, load on DB segment
            start_x, start_y = D_x, D_y
            end_x, end_y = B_x, B_y
        
        # Draw 5 arrows to represent the distributed load
        for j in range(5):
            x = start_x + j*(end_x-start_x)/4
            y = start_y + j*(end_y-start_y)/4
            ax.arrow(x, y, 0, -q_scale, head_width=0.03*max(a, b),
                    head_length=0.03*max(a, b), fc='g', ec='g', width=0.01*max(a, b))
        
        # Add label
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        ax.text(mid_x, mid_y-q_scale, f'q{i+1}={value} kN/m', fontsize=10)
    
    # Support symbols
    rect_size = 0.1 * max(a, b)
    # Fixed support at A
    ax.add_patch(Rectangle((A_x-rect_size/2, A_y-rect_size),
                          rect_size, rect_size, fc='gray', ec='black'))
    
    # Roller support at B
    circle_radius = rect_size/2
    ax.add_patch(Circle((B_x, B_y-circle_radius),
                        circle_radius, fc='gray', ec='black'))
    
    # Labels
    ax.text(A_x-0.15*a, A_y+0.15*b, 'A', fontsize=12)
    ax.text(B_x+0.15*a, B_y+0.15*b, 'B', fontsize=12)
    ax.text(C_x-0.15*a, C_y+0.15*b, 'C', fontsize=12)
    ax.text(D_x+0.15*a, D_y+0.15*b, 'D', fontsize=12)
    
    # Forces labels
    if N1 > 0:
        ax.text((C_x+D_x)/2+0.1*a, (C_y+D_y)/2+0.1*b, f'N1={N1:.2f} kN (T)', color='red', fontsize=10)
    else:
        ax.text((C_x+D_x)/2+0.1*a, (C_y+D_y)/2+0.1*b, f'N1={abs(N1):.2f} kN (C)', color='red', fontsize=10)
        
    if N2 > 0:
        ax.text((A_x+D_x)/2+0.1*a, (A_y+D_y)/2-0.1*b, f'N2={N2:.2f} kN (T)', color='blue', fontsize=10)
    else:
        ax.text((A_x+D_x)/2+0.1*a, (A_y+D_y)/2-0.1*b, f'N2={abs(N2):.2f} kN (C)', color='blue', fontsize=10)
    
    # Add legend and labels
    ax.legend(loc='upper right')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Bar System - Deformation Scale: ' + str(disp_scale) + 'x')
    
    # Set axis limits with margins
    margin = 0.5 * max(a, b)
    ax.set_xlim(min(A_x, C_x)-margin, max(B_x, D_x)+margin)
    ax.set_ylim(min(A_y, B_y)-margin-q_scale, max(C_y, D_y)+margin)
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    ax.grid(True)
    
    return fig

def main():
    st.title("Bar System Analysis")
    
    # Session state initialization for managing multiple loads
    if 'forces' not in st.session_state:
        st.session_state.forces = [
            {'id': str(uuid.uuid4()), 'value': 20.0, 'x_pos': 0, 'y_pos': 2.0, 
             'direction': [0, -1]}  # Default force at point C (downward)
        ]
    
    if 'distributed_loads' not in st.session_state:
        st.session_state.distributed_loads = [
            {'id': str(uuid.uuid4()), 'value': 5.0, 'type': 'q1'},  # Default load on CD
            {'id': str(uuid.uuid4()), 'value': 10.0, 'type': 'q2'}   # Default load on DB
        ]
    
    # Input parameters
    with st.sidebar:
        st.header("Input Parameters")
        a = st.number_input("Length a (m):", value=0.8, min_value=0.1, max_value=10.0, step=0.1)
        b = st.number_input("Length b (m):", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
        E = st.number_input("Young's modulus E (GPa):", value=210.0, min_value=1.0, step=10.0)
        R = st.number_input("Design resistance R (MPa):", value=210.0, min_value=1.0, step=10.0)
        allowable_disp = st.number_input("Allowable displacement [δ] (mm):", value=20.0, min_value=0.1, step=1.0)
    
    # Tabs for different steps
    tabs = st.tabs(["System Overview", "Load Configuration", "Calculation Steps", "Results Summary", "Deformation Visualization"])
    
    # Create node positions for reference
    C_x, C_y = 0, 2*b
    D_x, D_y = 2*a, b
    
    with tabs[1]:
        st.markdown("## Configure Loads")
        
        st.subheader("Concentrated Forces")
        for i, force in enumerate(st.session_state.forces):
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 0.5])
            with col1:
                st.session_state.forces[i]['value'] = st.number_input(
                    f"Force {i+1} value (kN):",
                    value=force['value'],
                    min_value=0.1,
                    step=1.0,
                    key=f"force_value_{force['id']}"
                )
            with col2:
                st.session_state.forces[i]['x_pos'] = st.slider(
                    f"Force {i+1} X position (m):",
                    min_value=0.0,
                    max_value=float(2*a),
                    value=float(force['x_pos']),
                    step=0.1,
                    key=f"force_x_{force['id']}"
                )
            with col3:
                st.session_state.forces[i]['y_pos'] = st.slider(
                    f"Force {i+1} Y position (m):",
                    min_value=0.0,
                    max_value=float(2*b),
                    value=float(force['y_pos']),
                    step=0.1,
                    key=f"force_y_{force['id']}"
                )
            with col4:
                # Direction selection using angle (more intuitive for users)
                angle_options = {"Downward (↓)": 270, "Upward (↑)": 90, "Rightward (→)": 0, "Leftward (←)": 180}
                
                # Get current angle from direction vector if it exists
                current_direction = force.get('direction', [0, -1])
                current_angle = 270  # Default downward
                if current_direction[0] == 0 and current_direction[1] == 1:
                    current_angle = 90
                elif current_direction[0] == 1 and current_direction[1] == 0:
                    current_angle = 0
                elif current_direction[0] == -1 and current_direction[1] == 0:
                    current_angle = 180
                    
                # Convert angle name to actual value for comparison
                current_angle_name = next((k for k, v in angle_options.items() if v == current_angle), "Downward (↓)")
                
                selected_angle_name = st.selectbox(
                    f"Force {i+1} direction:",
                    options=list(angle_options.keys()),
                    index=list(angle_options.keys()).index(current_angle_name),
                    key=f"force_dir_{force['id']}"
                )
                
                # Convert selection to direction vector
                angle_rad = np.deg2rad(angle_options[selected_angle_name])
                st.session_state.forces[i]['direction'] = [
                    np.cos(angle_rad),
                    np.sin(angle_rad)
                ]
            
            with col5:
                if len(st.session_state.forces) > 1:  # Ensure at least one force remains
                    if st.button("Remove", key=f"remove_force_{force['id']}"):
                        st.session_state.forces.remove(force)
                        st.experimental_rerun()
        
        if st.button("Add Force"):
            st.session_state.forces.append({
                'id': str(uuid.uuid4()),
                'value': 10.0,
                'x_pos': 0,
                'y_pos': 2*b,
                'direction': [0, -1]  # Default downward direction
            })
            st.experimental_rerun()
        
        st.subheader("Distributed Loads")
        for i, load in enumerate(st.session_state.distributed_loads):
            col1, col2, col3 = st.columns([1, 1, 0.5])
            with col1:
                st.session_state.distributed_loads[i]['value'] = st.number_input(
                    f"Load {i+1} value (kN/m):",
                    value=load['value'],
                    min_value=0.0,
                    step=1.0,
                    key=f"load_value_{load['id']}"
                )
            with col2:
                st.session_state.distributed_loads[i]['type'] = st.selectbox(
                    f"Load {i+1} location:",
                    options=['q1', 'q2'],
                    index=0 if load['type'] == 'q1' else 1,
                    key=f"load_type_{load['id']}",
                    help="q1: load on segment CD, q2: load on segment DB"
                )
            with col3:
                if st.button("Remove", key=f"remove_load_{load['id']}"):
                    st.session_state.distributed_loads.remove(load)
                    st.experimental_rerun()
        
        if st.button("Add Distributed Load"):
            st.session_state.distributed_loads.append({
                'id': str(uuid.uuid4()),
                'value': 5.0,
                'type': 'q1'
            })
            st.experimental_rerun()
    
    # Calculate results with the updated functions
    results = calculate_bar_system(a, b, st.session_state.forces, st.session_state.distributed_loads, E, R, allowable_disp)
    
    # Update the rest of the tabs to use the new calculation results
    with tabs[0]:
        st.markdown("## Bar System Overview")
        st.markdown("""
        This analysis tool calculates the forces, stresses, and deformations in a statically determinate bar system 
        consisting of two steel bars and rigid elements. The system can be loaded with multiple concentrated forces
        and distributed loads as defined in the Load Configuration tab.
        """)
        
        # Display initial visualization with multiple loads
        disp_scale_initial = 10
        fig = draw_bar_system_with_multiple_loads(
            a, b, 
            st.session_state.forces, 
            st.session_state.distributed_loads,
            results['N1'], results['N2'], 
            disp_scale_initial, 
            results['d1'], results['d2']
        )
        st.pyplot(fig)
        
        st.markdown("### System Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Lengths:**")
            st.markdown(f"- a = {a} m")
            st.markdown(f"- b = {b} m")
            st.markdown(f"- Bar 1 length = {results['l1']:.2f} m")
            st.markdown(f"- Bar 2 length = {results['l2']:.2f} m")
        
        with col2:
            st.markdown(f"**Material Properties:**")
            st.markdown(f"- E = {E} GPa")
            st.markdown(f"- R = {R} MPa")
            st.markdown(f"- Allowable displacement = {allowable_disp} mm")

    # Continue with the remaining tabs similar to the original implementation
    # but update to use the new calculation results and visualization

    with tabs[2]:
        st.markdown("## Calculation Steps")
        
        # Step 1: Forces in bar 1
        with st.expander("Step 1: Forces in Bar 1"):
            st.markdown("""
            The force in bar 1 is calculated from moment equilibrium around point A for element I,
            now considering force directions:
            
            $\\sum M_A = 0$
            
            $N_1 \\cdot a - \\sum (F_i^y \\cdot x_i) + \\sum (F_i^x \\cdot y_i) - \\sum (q_j \\cdot L_j \\cdot x_j) = 0$
            
            Where:
            - $F_i^y$ is the vertical component of force $F_i$
            - $F_i^x$ is the horizontal component of force $F_i$
            
            Therefore:
            
            $N_1 = \\frac{\\sum (F_i^y \\cdot x_i) - \\sum (F_i^x \\cdot y_i) + \\sum (q_j \\cdot L_j \\cdot x_j)}{a}$
            """)
            
            # Display calculation details for each force and load with direction
            st.markdown("### Moment contribution from concentrated forces:")
            for i, force in enumerate(st.session_state.forces):
                if force['y_pos'] > 0:  # Force is above the x-axis
                    direction = force.get('direction', [0, -1])
                    fx = force['value'] * direction[0]
                    fy = force['value'] * direction[1]
                    
                    # Format moment contribution based on direction
                    x_contrib = f"F_{i+1}^y \\cdot x_{i+1} = {fy:.2f} \\cdot {force['x_pos']:.2f} = {fy * force['x_pos']:.2f}"
                    y_contrib = f"F_{i+1}^x \\cdot y_{i+1} = {fx:.2f} \\cdot {force['y_pos']:.2f} = {fx * force['y_pos']:.2f}"
                    
                    if direction[0] == 0:  # Pure vertical force
                        st.latex(f"{x_contrib} \\text{{ kN}} \\cdot \\text{{m}}")
                    elif direction[1] == 0:  # Pure horizontal force
                        st.latex(f"-({y_contrib}) \\text{{ kN}} \\cdot \\text{{m}}")
                    else:  # Both components
                        st.latex(f"{x_contrib} - ({y_contrib}) = {fy * force['x_pos'] - fx * force['y_pos']:.2f} \\text{{ kN}} \\cdot \\text{{m}}")

            st.markdown("### Moment contribution from distributed loads:")
            for i, load in enumerate(st.session_state.distributed_loads):
                if load['type'] == 'q1':
                    equiv_force = load['value'] * (2 * a)
                    moment_arm = a
                    st.latex(f"q_{i+1} \\cdot (2a) \\cdot a = {load['value']} \\cdot (2 \\cdot {a}) \\cdot {a} = {equiv_force * moment_arm:.2f} \\text{{ kN}} \\cdot \\text{{m}}")
            
            st.latex(f"N_1 = {results['N1']:.2f} \\text{{ kN}}")
            
            if results['N1_tension']:
                st.success(f"Bar 1 is in tension with N₁ = {results['N1']:.2f} kN")
            else:
                st.error(f"Bar 1 is in compression with N₁ = {results['N1']:.2f} kN")
        
        # Step 2: Support reactions at A
        with st.expander("Step 2: Support Reactions at A"):
            st.markdown("""
            The support reactions at A are calculated from force equilibrium:

            Vertical equilibrium:
            $Y_A + N_1 - \\sum F_i^y - \\sum q_j \\cdot L_j = 0$

            Horizontal equilibrium:
            $X_A + \\sum F_i^x = 0$

            Where:
            - $F_i^y$ is the vertical component of force $F_i$
            - $F_i^x$ is the horizontal component of force $F_i$
            """)

            st.latex(f"Y_A = \\sum F_i^y + \\sum q_j \\cdot L_j - N_1 = {results['Ya']:.2f} \\text{{ kN}}")
            st.latex(f"X_A = -\\sum F_i^x = {results['Xa']:.2f} \\text{{ kN}}")
        
        # Step 3: Forces in bar 2
        with st.expander("Step 3: Forces in Bar 2"):
            st.markdown("""
            The force in bar 2 is calculated from moment equilibrium around point B for element II:
            
            $N_2 \\cdot \\sin(\\alpha) \\cdot (2a) - \\text{{sum of moments from forces}} - \\text{{sum of moments from distributed loads}} + N_1 \\cdot (2a) = 0$
            """)
            
            st.latex(f"\\sin(\\alpha) = \\frac{{{b}}}{{{results['l2']:.2f}}} = {results['sin_alpha']:.4f}")
            
            st.latex(f"N_2 = \\frac{{\\text{{sum of moments from forces}} + \\text{{sum of moments from distributed loads}} - N_1 \\cdot (2a)}}{{\\sin(\\alpha) \\cdot (2a)}} = {results['N2']:.2f} \\text{{ kN}}")
            
            if results['N2_tension']:
                st.success(f"Bar 2 is in tension with N₂ = {results['N2']:.2f} kN")
            else:
                st.error(f"Bar 2 is in compression with N₂ = {results['N2']:.2f} kN")
        
        # Step 4: Support reactions at B
        with st.expander("Step 4: Support Reactions at B"):
            st.markdown("""
            The support reactions at B are calculated from force equilibrium for the entire system:
            
            Vertical equilibrium:
            $Y_A + Y_B - \\text{{sum of vertical forces}} + N_2 \\cdot \\sin(\\alpha) - \\text{{sum of distributed loads}} = 0$
            
            Horizontal equilibrium:
            $X_B - N_2 \\cdot \\cos(\\alpha) = 0$
            """)
            
            st.latex(f"\\cos(\\alpha) = \\frac{{2 \\cdot {a}}}{{{results['l2']:.2f}}} = {results['cos_alpha']:.4f}")
            
            st.latex(f"Y_B = -Y_A - \\text{{sum of vertical forces}} + N_2 \\cdot \\sin(\\alpha) + \\text{{sum of distributed loads}} = {results['Yb']:.2f} \\text{{ kN}}")
            
            st.latex(f"X_B = -N_2 \\cdot \\cos(\\alpha) = -{results['N2']:.2f} \\cdot {results['cos_alpha']:.4f} = {results['Xb']:.2f} \\text{{ kN}}")
        
        # Step 5: Bar cross-section areas and diameters
        with st.expander("Step 5: Bar Cross-section Areas and Diameters"):
            st.markdown("""
            The required cross-section areas for the bars are calculated based on the design resistance:
            
            $A_i = \\frac{|N_i|}{R}$
            
            For circular cross-sections, the diameter is:
            
            $d_i = \\sqrt{\\frac{4 \\cdot A_i}{\\pi}}$
            """)
            
            st.latex(f"A_1 = \\frac{{|{results['N1']:.2f}| \\cdot 1000}}{{{R} \\cdot 10^6}} = {results['A1_cm2']:.2f} \\text{{ cm}}^2")
            
            st.latex(f"d_1 = \\sqrt{{\\frac{{4 \\cdot {results['A1_cm2']:.2f}}}{{\\pi}}}} = {results['d1']:.2f} \\text{{ cm}}")
            
            st.latex(f"A_2 = \\frac{{|{results['N2']:.2f}| \\cdot 1000}}{{{R} \\cdot 10^6}} = {results['A2_cm2']:.2f} \\text{{ cm}}^2")
            
            st.latex(f"d_2 = \\sqrt{{\\frac{{4 \\cdot {results['A2_cm2']:.2f}}}{{\\pi}}}} = {results['d2']:.2f} \\text{{ cm}}")
        
        # Step 6: Stiffness check
        with st.expander("Step 6: Stiffness Check"):
            st.markdown("""
            The elongation of the bars is calculated using:
            
            $\\Delta l_i = \\frac{N_i \\cdot l_i}{E \\cdot A_i}$
            
            The maximum displacement at point C is calculated from geometric considerations:
            
            $\\delta_C = \\sqrt{\\Delta l_1^2 + (\\Delta l_2 \\cdot \\sin(\\alpha))^2}$
            """)
            
            st.latex(f"\\Delta l_1 = \\frac{{{results['N1']:.2f} \\cdot 1000 \\cdot {results['l1']:.2f}}}{{{E} \\cdot 10^9 \\cdot {results['A1_cm2']:.4f} \\cdot 10^{-4}}} = {results['delta_l1']:.2f} \\text{{ mm}}")
            
            st.latex(f"\\Delta l_2 = \\frac{{{results['N2']:.2f} \\cdot 1000 \\cdot {results['l2']:.2f}}}{{{E} \\cdot 10^9 \\cdot {results['A2_cm2']:.4f} \\cdot 10^{-4}}} = {results['delta_l2']:.2f} \\text{{ mm}}")
            
            st.latex(f"\\delta_C = \\sqrt{{{results['delta_l1']:.2f}^2 + ({results['delta_l2']:.2f} \\cdot {results['sin_alpha']:.4f})^2}} = {results['delta_C']:.2f} \\text{{ mm}}")
            
            if results['delta_C'] <= allowable_disp:
                st.success(f"Displacement check: {results['delta_C']:.2f} mm ≤ {allowable_disp} mm ✓")
            else:
                st.error(f"Displacement check: {results['delta_C']:.2f} mm > {allowable_disp} mm ✗")
    
    with tabs[3]:
        st.markdown("## Results Summary")
        
        # Create two columns for the results
        col1, col2 = st.columns(2)
        
        # First column for bar forces and reactions
        with col1:
            st.markdown("### Bar Forces and Support Reactions")
            data_forces = {
                "Parameter": ["N₁ (Bar 1)", "N₂ (Bar 2)", "Ya (Support A)", "Yb (Support B)", "Xb (Support B)"],
                "Value [kN]": [
                    f"{results['N1']:.2f}{' (T)' if results['N1_tension'] else ' (C)'}",
                    f"{results['N2']:.2f}{' (T)' if results['N2_tension'] else ' (C)'}",
                    f"{results['Ya']:.2f}",
                    f"{results['Yb']:.2f}",
                    f"{results['Xb']:.2f}"
                ]
            }
            st.table(pd.DataFrame(data_forces))
        
        # Second column for cross-section properties
        with col2:
            st.markdown("### Bar Properties")
            data_properties = {
                "Parameter": ["A₁ (Cross-section area of Bar 1)", "d₁ (Diameter of Bar 1)", 
                            "A₂ (Cross-section area of Bar 2)", "d₂ (Diameter of Bar 2)"],
                "Value": [f"{results['A1_cm2']:.2f} cm²", f"{results['d1']:.2f} cm",
                         f"{results['A2_cm2']:.2f} cm²", f"{results['d2']:.2f} cm"]
            }
            st.table(pd.DataFrame(data_properties))
            
        st.markdown("### Displacements")
        data_displacements = {
            "Parameter": ["Δl₁ (Elongation of Bar 1)", "Δl₂ (Elongation of Bar 2)", 
                         "δ_C (Maximum displacement at C)"],
            "Value [mm]": [f"{results['delta_l1']:.2f}", f"{results['delta_l2']:.2f}", 
                          f"{results['delta_C']:.2f}"]
        }
        st.table(pd.DataFrame(data_displacements))
        
        # Display stiffness check result
        st.markdown("### Stiffness Check")
        if results['delta_C'] <= allowable_disp:
            st.success(f"Maximum displacement {results['delta_C']:.2f} mm is within the allowable limit of {allowable_disp} mm.")
        else:
            st.error(f"Maximum displacement {results['delta_C']:.2f} mm exceeds the allowable limit of {allowable_disp} mm. The structure needs to be modified.")

    with tabs[4]:
        st.markdown("## Deformation Visualization")
        st.markdown("""
        Adjust the deformation scale factor to visualize the system's deformation. 
        The actual displacements are very small, so scaling is necessary for visualization.
        """)
        
        # Slider for deformation scale
        disp_scale = st.slider("Deformation Scale Factor:", min_value=1, max_value=100, value=10, key="deformation_scale_slider")
        
        # Draw the bar system with the selected scale and multiple loads
        fig = draw_bar_system_with_multiple_loads(
            a, b, 
            st.session_state.forces, 
            st.session_state.distributed_loads,
            results['N1'], results['N2'], 
            disp_scale, 
            results['d1'], results['d2']
        )
        st.pyplot(fig)
        
        # Display displacement values
        st.markdown("### Deformation Values")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"- Elongation of Bar 1: {results['delta_l1']:.2f} mm")
            st.markdown(f"- Elongation of Bar 2: {results['delta_l2']:.2f} mm")
        with col2:
            st.markdown(f"- Displacement at Point C: {results['delta_C']:.2f} mm")
            if results['delta_C'] <= allowable_disp:
                st.success(f"Within allowable limit ({allowable_disp} mm)")
            else:
                st.error(f"Exceeds allowable limit ({allowable_disp} mm)")
        
        # Optimization section
        st.markdown("### Bar Diameter Optimization")
        if st.button("Calculate Optimal Bar Diameters"):
            optimal = calculate_optimal_diameters(
                results['N1'], results['N2'], 
                results['l1'], results['l2'], 
                results['sin_alpha'], 
                E, R, allowable_disp
            )
            
            # Material volume reduction percentage
            current_volume = results['A1_cm2'] * results['l1'] * 0.01 + results['A2_cm2'] * results['l2'] * 0.01  # liters
            
            # Create comparison table
            comparison_data = {
                "Parameter": ["Diameter Bar 1 (cm)", "Diameter Bar 2 (cm)", 
                              "Area Bar 1 (cm²)", "Area Bar 2 (cm²)",
                              "Material Volume (L)"],
                "Current Design": [
                    f"{results['d1']:.2f}", 
                    f"{results['d2']:.2f}", 
                    f"{results['A1_cm2']:.2f}", 
                    f"{results['A2_cm2']:.2f}",
                    f"{current_volume:.2f}"
                ],
                "Optimal Design": [
                    f"{optimal['d1_optimal']:.2f}", 
                    f"{optimal['d2_optimal']:.2f}", 
                    f"{optimal['A1_optimal']:.2f}", 
                    f"{optimal['A2_optimal']:.2f}",
                    f"{optimal['material_volume']:.2f}"
                ]
            }
            st.table(pd.DataFrame(comparison_data))
            
            # Show optimization details
            if optimal['is_stiffness_critical']:
                st.info(f"Design is stiffness-critical. Optimization method: {optimal['optimization_method']}")
                
                # Calculate and display material savings
                volume_reduction = current_volume - optimal['material_volume']
                if volume_reduction > 0:
                    percentage_reduction = (volume_reduction / current_volume) * 100
                    st.success(f"Material savings: {volume_reduction:.2f}L ({percentage_reduction:.1f}%)")
                else:
                    st.warning("Current design is already optimal or nearly optimal.")
            else:
                st.success("Design is strength-critical. Current diameters are optimal.")

if __name__ == "__main__":
    main()