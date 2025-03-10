import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid
import sys
import os
import time  # For animation timing

# Add the project root to the Python path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from utility modules
from vizualizations.visualization import draw_bar_system_with_multiple_loads
from vizualizations.visualization_3d import create_3d_bar_system
from utils.calculations import calculate_bar_system
from utils.optimization import calculate_optimal_diameters

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
    
    # Calculate results using the imported function
    results = calculate_bar_system(a, b, st.session_state.forces, st.session_state.distributed_loads, E, R, allowable_disp)
    
    # Update the rest of the tabs to use the new calculation results
    with tabs[0]:
        st.markdown("## Bar System Overview")
        st.markdown("""
        This analysis tool calculates the forces, stresses, and deformations in a statically determinate bar system 
        consisting of two steel bars and rigid elements. The system can be loaded with multiple concentrated forces
        and distributed loads as defined in the Load Configuration tab.
        """)
        
        # Display initial visualization with multiple loads using imported function
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

        viz_tabs = st.tabs(["2D View", "3D View"])

        with viz_tabs[0]:
            st.markdown("### 2D View")
            # Display the bar system with the selected scale and multiple loads using imported function
            fig = draw_bar_system_with_multiple_loads(
                a, b, 
                st.session_state.forces, 
                st.session_state.distributed_loads,
                results['N1'], results['N2'], 
                disp_scale, 
                results['d1'], results['d2']
            )
            st.pyplot(fig)
        
        # Display displacement values with both actual and scaled values
        st.markdown("### Deformation Values")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"- Elongation of Bar 1: {results['delta_l1']:.2f} mm")
            st.markdown(f"- Elongation of Bar 2: {results['delta_l2']:.2f} mm")
        with col2:
            # Show both actual and scaled displacement values
            scaled_delta_C = results['delta_C'] * disp_scale
            st.markdown(f"- Actual Displacement at Point C: {results['delta_C']:.2f} mm")
            st.markdown(f"- Scaled Displacement at Point C: {scaled_delta_C:.2f} mm (×{disp_scale})")
            
            # Keep the original check against allowable displacement using actual value
            if results['delta_C'] <= allowable_disp:
                st.success(f"Within allowable limit ({allowable_disp} mm)")
            else:
                st.error(f"Exceeds allowable limit ({allowable_disp} mm)")
        
        # Add a note about the scaled visualization
        st.info(f"The visualization shows deformations scaled by {disp_scale}× for better visibility.")
        
        # Optimization section using imported function
        st.markdown("### Bar Diameter Optimization")
        if st.button("Calculate Optimal Bar Diameters"):
            # Use actual values for optimization
            optimal = calculate_optimal_diameters(
                results['N1'], results['N2'], 
                results['l1'], results['l2'], 
                results['sin_alpha'], 
                E, R, allowable_disp
            )
            
            # Material volume reduction percentage
            current_volume = results['A1_cm2'] * results['l1'] * 0.01 + results['A2_cm2'] * results['l2'] * 0.01  # liters
            
            # Calculate scaled displacement for display
            scaled_delta_C = optimal['delta_C'] * disp_scale
            
            # Create comparison table with both actual and scaled displacement
            comparison_data = {
                "Parameter": ["Diameter Bar 1 (cm)", "Diameter Bar 2 (cm)", 
                              "Area Bar 1 (cm²)", "Area Bar 2 (cm²)",
                              "Material Volume (L)",
                              "Actual Displacement (mm)",
                              f"Scaled Displacement ×{disp_scale} (mm)"],
                "Current Design": [
                    f"{results['d1']:.2f}", 
                    f"{results['d2']:.2f}", 
                    f"{results['A1_cm2']:.2f}", 
                    f"{results['A2_cm2']:.2f}",
                    f"{current_volume:.2f}",
                    f"{results['delta_C']:.2f}",
                    f"{results['delta_C'] * disp_scale:.2f}"
                ],
                "Optimal Design": [
                    f"{optimal['d1_optimal']:.2f}", 
                    f"{optimal['d2_optimal']:.2f}", 
                    f"{optimal['A1_optimal']:.2f}", 
                    f"{optimal['A2_optimal']:.2f}",
                    f"{optimal['material_volume']:.2f}",
                    f"{optimal['delta_C']:.2f}",
                    f"{scaled_delta_C:.2f}"
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