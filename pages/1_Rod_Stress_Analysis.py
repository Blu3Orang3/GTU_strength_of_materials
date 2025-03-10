import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Union, Optional
import pandas as pd
import os
import io
import sys
import pathlib

# Add the parent directory to sys.path to make relative imports work
parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
if (parent_dir not in sys.path):
    sys.path.append(parent_dir)

# Import tooltip utilities with error handling
try:
    from utils.tooltips import create_tooltip_style, tooltip, concept_tooltip, ENGINEERING_TOOLTIPS
    tooltips_available = True
except ImportError:
    # Define fallback functions if module not found
    def create_tooltip_style():
        pass
    
    def tooltip(text, concept=None):
        return text
    
    def concept_tooltip(concept):
        return concept
    
    ENGINEERING_TOOLTIPS = {}
    tooltips_available = False

# Set page configuration
st.set_page_config(
    page_title="Rod Stress Analysis Tool",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add CSS for better styling and tooltips
st.markdown("""
<style>
    .main {
        padding: 1rem 1rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stPlotlyChart {
        margin: 0 auto;
    }
    .rod-diagram {
        text-align: center;
        margin: 20px 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning {
        color: #ff4b4b;
        font-weight: bold;
    }
    .concept {
        color: #1E3A8A;
        font-weight: 500;
        text-decoration: underline dotted;
        cursor: help;
    }
</style>
""", unsafe_allow_html=True)

# At the top of your app after headers
if tooltips_available:
    with st.expander("Learn about Engineering Concepts"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Basic Concepts")
            st.markdown("**Axial Loading**: " + ENGINEERING_TOOLTIPS.get("Axial Loading", ""))
            st.markdown("**Stepped Rod**: " + ENGINEERING_TOOLTIPS.get("Stepped Rod", ""))
            st.markdown("**Cross-sectional Area**: " + ENGINEERING_TOOLTIPS.get("Cross-sectional Area", ""))
        with col2:
            st.subheader("Analysis Outputs")
            st.markdown("**Internal Forces**: " + ENGINEERING_TOOLTIPS.get("Internal Force", ""))
            st.markdown("**Normal Stresses**: " + ENGINEERING_TOOLTIPS.get("Normal Stress", ""))
            st.markdown("**Displacements**: " + ENGINEERING_TOOLTIPS.get("Displacement", ""))

@dataclass
class Force:
    def __init__(self, magnitude: float, position: float, direction: int):
        """
        Args:
            magnitude: Force magnitude
            position: Position along rod
            direction: +1 for tension, -1 for compression
        """
        self.magnitude = magnitude    
        self.position = position      
        self.direction = direction    # +1 or -1
        
@dataclass
class Section:
    """Represents a section of the rod"""
    area: float  # cm²
    length: float  # cm
    position: float  # cm from bottom

class SteppedRod:
    """Enhanced stepped rod analysis system with user inputs"""

    def __init__(self, sections: List[Section], E: float):
        """Initialize stepped rod with sections and material properties"""
        self.dtype = np.float64
        self.E = self.dtype(E) * self.dtype(1e9)  # Convert GPa to Pa
        self.sections = sections
        self._validate_sections()
        
    def _validate_sections(self):
        """Validate section geometry and connectivity"""
        if len(self.sections) != 3:
            raise ValueError("This implementation requires exactly 3 sections (a, b, c)")
    
    def analyze(self, forces: List[Force], plot: bool = True) -> Dict[str, np.ndarray]:
        """
        Analyze rod under given forces
        """
        # Sort forces from top to bottom for consistency
        sorted_forces = sorted(forces, key=lambda x: x.position, reverse=True)

        # Extract force magnitudes considering their directions
        F = [f.magnitude * f.direction for f in sorted_forces]

        # Pad with zeros to ensure we have at least 4 forces
        while len(F) < 4:
            F.append(0)

        # Calculate internal forces
        N = np.zeros(5, dtype=self.dtype)
        N[0] = F[0] * 1e3  # N12 = F1
        N[1] = (F[0] + F[1]) * 1e3  # N23 = F1 + F2
        N[2] = (F[0] + F[1] + F[2]) * 1e3  # N34 = F1 + F2 + F3
        N[3] = N[2]  # N45 = N34
        N[4] = N[2] + F[3] * 1e3  # N56 = N34 + F4

        # Create arrays for detailed section properties
        A = np.array([
            self.sections[0].area, 
            self.sections[1].area,
            self.sections[1].area,
            self.sections[2].area,
            self.sections[2].area
        ], dtype=self.dtype) * 1e-4  # Convert cm² to m²

        # Calculate lengths for each segment
        L = np.array([
            self.sections[2].length,
            self.sections[1].length / 2,
            self.sections[1].length / 2,
            self.sections[0].length / 2,
            self.sections[0].length / 2
        ], dtype=self.dtype) * 1e-2  # Convert cm to m

        # Calculate positions for plotting
        positions = np.array([
            0,
            self.sections[0].length,
            self.sections[0].length + self.sections[1].length / 2,
            self.sections[0].length + self.sections[1].length,
            self.sections[0].length + self.sections[1].length + self.sections[2].length
        ], dtype=self.dtype) * 1e-2  # Convert cm to m

        # Calculate stresses and deformations
        stresses = N / A
        deformations = (stresses * L) / self.E

        # Calculate cumulative displacements from bottom to top
        cumulative_displacements = np.zeros_like(deformations)
        cumulative_displacements[-1] = deformations[-1]
        for i in range(len(deformations) - 2, -1, -1):
            cumulative_displacements[i] = cumulative_displacements[i + 1] + deformations[i]

        # Prepare results
        results = {
            'internal_forces_kN': np.round(N * 1e-3, 3),
            'stresses_MPa': np.round(stresses * 1e-6, 3),
            'deformations_mm': np.round(deformations * 1e3, 3),
            'cumulative_displacements_mm': np.round(cumulative_displacements * 1e3, 3),
            'positions_m': positions
        }

        return results

    def create_plots(self, results: Dict[str, np.ndarray]):
        """Create engineering-style visualization plots matching the reference diagram"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
        
        # Convert positions from m to cm for plotting
        positions_cm = results['positions_m'] * 100
        
        # Common styling for all axes
        for ax in [ax1, ax2, ax3]:
            ax.grid(True, linestyle='-', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.invert_yaxis()  # Invert y-axis to match engineering convention
            ax.axvline(x=0, color='black', linewidth=0.5)  # Add vertical line at x=0
        
        self._plot_internal_forces(ax1, results, positions_cm)
        self._plot_stress_distribution(ax2, results, positions_cm)
        self._plot_displacement_distribution(ax3, results, positions_cm)

        # Adjust layout
        plt.tight_layout()
        return fig

    def _plot_internal_forces(self, ax, results, positions_cm):
        """Plot internal forces as segments with force values and dashed areas"""
        forces = results['internal_forces_kN']
        
        # Create vertical lines and dashed areas for each force section
        for i in range(len(positions_cm) - 1):
            y_pos = [positions_cm[i], positions_cm[i+1]]
            x_pos = [forces[i], forces[i]]
            
            # Plot vertical line
            ax.plot(x_pos, y_pos, 'b-', linewidth=2)
            
            # Create dashed area
            y_fill = np.linspace(y_pos[0], y_pos[1], 10)
            ax.fill_betweenx(y_fill, 0, forces[i], 
                            hatch='///', alpha=0.1, color='blue')
            
            # Add force values at midpoint
            mid_y = (y_pos[0] + y_pos[1]) / 2
            ax.text(forces[i], mid_y, f' {forces[i]:.0f}', 
                   verticalalignment='center')
            
            # Add plus/minus symbols
            if forces[i] > 0:
                ax.text(forces[i], y_pos[0], ' ⊕', verticalalignment='bottom')
            else:
                ax.text(forces[i], y_pos[0], ' ⊖', verticalalignment='bottom')
        
        # Set reasonable x-axis limits based on force values
        max_force = max(abs(forces))
        ax.set_xlim(-max_force * 1.2, max_force * 1.2)
        
        ax.set_title('N, kN', pad=20)
        ax.set_xlabel('Force')
        ax.set_ylabel('Height (cm)')

    def _plot_stress_distribution(self, ax, results, positions_cm):
        """Plot stress distribution"""
        stresses = results['stresses_MPa']

        max_abs_stress = max(abs(np.max(stresses)), abs(np.min(stresses)))
        text_padding = max_abs_stress * 0.05

        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        for i in range(len(positions_cm) - 1):
            y_pos = [positions_cm[i], positions_cm[i+1]]
            x_pos = [stresses[i], stresses[i]]

            line_color = 'blue' if stresses[i] >= 0 else 'red'
            ax.plot(x_pos, y_pos, color=line_color, linewidth=2)

            y_fill = np.linspace(y_pos[0], y_pos[1], 10)
            if stresses[i] >= 0:
                ax.fill_betweenx(y_fill, 0, stresses[i],
                                hatch='///', alpha=0.1, color='blue')
            else:
                ax.fill_betweenx(y_fill, stresses[i], 0,
                                hatch='\\\\\\', alpha=0.1, color='red')

            mid_y = (y_pos[0] + y_pos[1]) / 2
            if stresses[i] >= 0:
                ax.text(stresses[i] + text_padding, mid_y, f'{stresses[i]:.1f}',
                       verticalalignment='center',
                       horizontalalignment='left')
            else:
                ax.text(stresses[i] - text_padding, mid_y, f'{stresses[i]:.1f}',
                       verticalalignment='center',
                       horizontalalignment='right')

            if i in [0, len(stresses)//2, len(stresses)-1]:
                symbol = '⊕' if stresses[i] > 0 else '⊖'
                symbol_offset = text_padding * 8
                if stresses[i] >= 0:
                    ax.text(stresses[i] + symbol_offset, mid_y, symbol,
                           verticalalignment='center',
                           horizontalalignment='left')
                else:
                    ax.text(stresses[i] - symbol_offset, mid_y, symbol,
                           verticalalignment='center',
                           horizontalalignment='right')

        padding = max_abs_stress * 0.3
        ax.set_xlim(-max_abs_stress - padding, max_abs_stress + padding)

        ax.grid(True, linestyle=':', alpha=0.3)

        ax.set_title('Stress Distribution (σ, MPa)', pad=20)
        ax.set_xlabel('Stress')
        ax.set_ylabel('Height (cm)')

        ax.tick_params(axis='both', which='major', labelsize=10)

        import matplotlib.patches as patches
        pos_patch = patches.Patch(facecolor='blue', alpha=0.1, hatch='///', label='Positive stress')
        neg_patch = patches.Patch(facecolor='red', alpha=0.1, hatch='\\\\\\', label='Negative stress')
        ax.legend(handles=[pos_patch, neg_patch], loc='best')

    def _plot_displacement_distribution(self, ax, results, positions_cm):
        """Plot displacement distribution with improved styling"""
        displacements = results['cumulative_displacements_mm']

        # Create more points for smooth curve
        y_smooth = np.linspace(positions_cm[0], positions_cm[-1], 100)
        x_smooth = np.interp(y_smooth, positions_cm, displacements)

        # Plot smooth curve
        ax.plot(x_smooth, y_smooth, 'b-', linewidth=2)

        # Create separate hatched fills for positive and negative values
        ax.fill_betweenx(y_smooth, 0, x_smooth,
                         where=(x_smooth >= 0),
                         hatch='///', alpha=0.1, color='blue')
        ax.fill_betweenx(y_smooth, x_smooth, 0,
                         where=(x_smooth <= 0),
                         hatch='\\\\\\', alpha=0.1, color='red')

        # Calculate padding for text based on data range
        max_abs_disp = max(abs(np.max(displacements)), abs(np.min(displacements)))
        text_padding = max_abs_disp * 0.05  # 5% of max displacement for text padding

        # Add displacement values at original points with improved positioning
        for i, (x, y) in enumerate(zip(displacements, positions_cm)):
            # Adjust text position based on sign of displacement
            if x >= 0:
                ax.text(x + text_padding, y, f'{x:.2f}', 
                       verticalalignment='center',
                       horizontalalignment='left')
            else:
                ax.text(x - text_padding, y, f'{x:.2f}', 
                       verticalalignment='center',
                       horizontalalignment='right')

            # Add plus/minus symbols at key points with improved positioning
            if i in [0, len(displacements)//2, len(displacements)-1]:
                symbol = '⊕' if x > 0 else '⊖'
                symbol_offset = text_padding * 8
                if x >= 0:
                    ax.text(x + symbol_offset, y, symbol,
                           verticalalignment='center',
                           horizontalalignment='left')
                else:
                    ax.text(x - symbol_offset, y, symbol,
                           verticalalignment='center',
                           horizontalalignment='right')

        # Set axis limits with better spacing
        x_max = max_abs_disp
        x_min = -max_abs_disp
        padding = max_abs_disp * 0.3  # 30% padding on each side

        ax.set_xlim(x_min - padding, x_max + padding)

        # Add zero line for reference
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        # Improve grid
        ax.grid(True, linestyle=':', alpha=0.3)

        # Customize appearance
        ax.set_title('Displacement Distribution (W, mm)', pad=20)
        ax.set_xlabel('Displacement')
        ax.set_ylabel('Height (cm)')

        # Ensure ticks are readable
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Add legend to explain hatching
        import matplotlib.patches as patches
        pos_patch = patches.Patch(facecolor='blue', alpha=0.1, hatch='///', label='Positive displacement')
        neg_patch = patches.Patch(facecolor='red', alpha=0.1, hatch='\\\\\\', label='Negative displacement')
        ax.legend(handles=[pos_patch, neg_patch], loc='best')

def display_results(results: Dict[str, np.ndarray]):
    """Display results in a formatted way"""
    st.subheader("Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Internal Forces (kN)")
        if tooltips_available:
            st.markdown(tooltip("Internal Forces", "Internal Force"), unsafe_allow_html=True)
        for i, force in enumerate(results['internal_forces_kN']):
            st.write(f"N{i+1}{i+2}: {force:.3f} kN")
            
    with col2:
        st.markdown("##### Normal Stresses (MPa)")
        for i, stress in enumerate(results['stresses_MPa']):
            st.write(f"σ{i+1}{i+2}: {stress:.3f} MPa")
    
    with col3:
        st.markdown("##### Deformations (mm)")
        for i, deform in enumerate(results['deformations_mm']):
            st.write(f"Δl{i+1}{i+2}: {deform:.3f} mm")
            
    st.markdown("##### Cumulative Displacements (mm)")
    for i, disp in enumerate(results['cumulative_displacements_mm']):
        st.write(f"Total at section {i+1}: {disp:.3f} mm")

# Create a visual representation of the stepped rod
def draw_rod_diagram(lengths, areas):
    """Create a simple visualization of the rod configuration"""
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Normalize areas for visualization
    max_area = max(areas)
    norm_areas = [a/max_area * 0.5 for a in areas]
    
    # Calculate positions
    positions = [0]
    for i in range(len(lengths)):
        positions.append(positions[-1] + lengths[i])
    
    # Draw the rod
    for i in range(len(lengths)):
        # Draw section
        rect = plt.Rectangle(
            (positions[i], 0.5 - norm_areas[i]/2),
            lengths[i],
            norm_areas[i],
            facecolor='lightgray',
            edgecolor='black'
        )
        ax.add_patch(rect)
        
        # Add section label
        section_label = chr(97 + i)  # a, b, c
        ax.text(
            positions[i] + lengths[i]/2,
            0.25,
            f"{section_label}\n{lengths[i]} cm\n{areas[i]} cm²",
            ha='center',
            va='center'
        )
    
    # Set plot limits and remove axes
    ax.set_xlim(0, positions[-1])
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    return fig

# Main application
def main():
    # Sidebar for user inputs
    st.sidebar.header("Rod Properties")
    
    # Material properties
    if tooltips_available:
        st.sidebar.write("##### Material Properties")
        st.sidebar.markdown(tooltip("Young's Modulus", "Young's Modulus"), unsafe_allow_html=True)
    else:
        st.sidebar.markdown("##### Material Properties")
    
    # For Hooke's law section
    if tooltips_available:
        st.sidebar.write("<small><i>Learn more about:</i></small>", unsafe_allow_html=True)
        st.sidebar.markdown(tooltip("Hooke's Law", "Hooke's Law"), unsafe_allow_html=True)
        st.sidebar.markdown(tooltip("Stress-Strain Diagram", "Stress-Strain Diagram"), unsafe_allow_html=True)
    
    # Young's modulus with native tooltip
    if tooltips_available:
        modulus_help = ENGINEERING_TOOLTIPS.get("Young's Modulus", "")
    else:
        modulus_help = ""

    E = st.sidebar.number_input(
        "Young's Modulus (GPa)", 
        min_value=1.0, 
        value=200.0, 
        step=1.0,
        help=modulus_help
    )
    
    # Option 3: Use Unicode right single quotation mark
    hookes_law = "Hooke's Law"
    # Use the variable in the f-string instead of a backslash escape sequence
    if tooltips_available:
        st.sidebar.markdown(
            f"<small><i>Learn more about {tooltip(hookes_law, hookes_law)} and the "
            f"{tooltip('Stress-Strain Diagram', 'Stress-Strain Diagram')}</i></small>",
            unsafe_allow_html=True
        )
    
    # Rod section lengths
    st.sidebar.subheader("Section Lengths (cm)")
    length_a = st.sidebar.number_input("Length a", min_value=1.0, value=10.0, step=1.0)
    length_b = st.sidebar.number_input("Length b", min_value=1.0, value=15.0, step=1.0)
    length_c = st.sidebar.number_input("Length c", min_value=1.0, value=20.0, step=1.0)
    
    # Rod section areas
    # Rod section areas - use Streamlit's native tooltip
    st.sidebar.markdown("##### Section Cross-sectional Areas (cm²)")
    if tooltips_available:
        area_help = ENGINEERING_TOOLTIPS.get("Cross-sectional Area", "")
    else:
        area_help = ""
        
    area_a = st.sidebar.number_input("Area a", min_value=0.1, value=2.0, step=0.5, help=area_help)
    area_b = st.sidebar.number_input("Area b", min_value=0.1, value=3.0, step=0.5, help=area_help) 
    area_c = st.sidebar.number_input("Area c", min_value=0.1, value=4.0, step=0.5, help=area_help)
    
    # Force inputs - use Streamlit's native tooltip
    st.sidebar.markdown("##### Applied Forces")
    if tooltips_available:
        force_help = ENGINEERING_TOOLTIPS.get("Axial Loading", "")
    else:
        force_help = ""

    # Dynamically add/remove forces
    if 'num_forces' not in st.session_state:
        st.session_state.num_forces = 2  # Start with 2 forces by default

    if st.sidebar.button("Add Force", help="Add another force to the rod"):
        st.session_state.num_forces += 1

    if st.session_state.num_forces > 1 and st.sidebar.button("Remove Force", help="Remove the last force"):
        st.session_state.num_forces -= 1
    
    forces = []
    for i in range(st.session_state.num_forces):
        st.sidebar.markdown(f"#### Force {i+1}")
        
        # Force magnitude with native tooltip
        magnitude = st.sidebar.number_input(
            f"Magnitude (kN)", 
            min_value=0.0, 
            value=10.0 if i == 0 else 5.0, 
            step=1.0,
            key=f"magnitude_{i}",
            help="Force magnitude in kilonewtons (kN)"
        )
        
        # Force direction - Fix HTML display in selectbox
        if tooltips_available:
            direction_options = ["Tension (+)", "Compression (-)"]
            direction_display = st.sidebar.selectbox(
                "Direction",
                options=range(len(direction_options)),
                index=0 if i == 0 else 1,
                key=f"direction_{i}",
                format_func=lambda x: direction_options[x]
            )
            direction_value = 1 if direction_display == 0 else -1
        else:
            direction = st.sidebar.selectbox(
                "Direction",
                options=["Tension (+)", "Compression (-)"],
                index=0 if i == 0 else 1,
                key=f"direction_{i}"
            )
            direction_value = 1 if direction == "Tension (+)" else -1
        
        # Force position
        section = st.sidebar.selectbox(
            "Section", 
            options=["a", "b", "c"],
            index=i % 3,
            key=f"section_{i}"
        )
        
        position_type = st.sidebar.selectbox(
            "Position",
            options=["Start", "Middle", "End"],
            index=i % 3,
            key=f"position_type_{i}"
        )
        
        # Calculate position
        if section == "a":
            base_pos = 0
            section_length = length_a
        elif section == "b":
            base_pos = length_a
            section_length = length_b
        else:  # section c
            base_pos = length_a + length_b
            section_length = length_c
            
        if position_type == "Start":
            position = base_pos
        elif position_type == "Middle":
            position = base_pos + section_length/2
        else:  # End
            position = base_pos + section_length
            
        # Add force if magnitude is greater than 0
        if magnitude > 0:
            forces.append(Force(
                magnitude=magnitude,
                position=position,
                direction=direction_value
            ))
    
    # Create rod sections
    sections = [
        Section(area=area_a, length=length_a, position=0),
        Section(area=area_b, length=length_b, position=length_a),
        Section(area=area_c, length=length_c, position=length_a + length_b)
    ]
    
    # Show rod diagram
    if tooltips_available:
        # For Rod Configuration
        st.write("### Rod Configuration")
        st.markdown(tooltip('(Learn more about stepped rods)', 'Stepped Rod'), unsafe_allow_html=True)
    else:
        st.markdown("### Rod Configuration")
        
    lengths = [length_a, length_b, length_c]
    areas = [area_a, area_b, area_c]
    rod_fig = draw_rod_diagram(lengths, areas)
    st.pyplot(rod_fig)
    
    # Show forces on the rod
    if forces:
        st.subheader("Applied Forces")
        force_data = [(i+1, f.magnitude, "Tension" if f.direction > 0 else "Compression", f.position) 
                      for i, f in enumerate(forces)]
        force_df = pd.DataFrame(force_data, columns=["Force #", "Magnitude (kN)", "Direction", "Position (cm)"])
        st.dataframe(force_df)
    else:
        st.warning("No forces have been added. Please add at least one force.")
    
    # Analyze button
    if st.button("Analyze Rod"):
        if not forces:
            st.error("Cannot analyze rod without any forces. Please add at least one force.")
        else:
            try:
                # Create and analyze rod
                rod = SteppedRod(sections, E)
                results = rod.analyze(forces, plot=False)
                
                # Add tabs for better organization of results
                tab1, tab2 = st.tabs(["Numerical Results", "Visual Results"])

                with tab1:
                    display_results(results)
                    
                with tab2:
                    fig = rod.create_plots(results)
                    st.pyplot(fig)
                
                # Download results as CSV
                results_df = pd.DataFrame({
                    "Position (cm)": results["positions_m"] * 100,
                    "Internal Force (kN)": results["internal_forces_kN"],
                    "Stress (MPa)": results["stresses_MPa"],
                    "Deformation (mm)": results["deformations_mm"],
                    "Displacement (mm)": results["cumulative_displacements_mm"]
                })
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="rod_analysis_results.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    # Engineering Concepts Dictionary
    if tooltips_available and ENGINEERING_TOOLTIPS:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Engineering Concepts Dictionary")
        
        # Create accordion for each engineering concept
        for concept, description in ENGINEERING_TOOLTIPS.items():
            with st.sidebar.expander(concept):
                st.write(description)
    
    # About section
    with st.expander("About this Application"):
        if tooltips_available:
            st.write("### Rod Stress Analysis Tool")
            
            st.write("This application analyzes the internal forces, stresses, and displacements " 
                    "of a stepped rod under axial loading. The rod consists of three sections " 
                    "with different cross-sectional areas and lengths.")
            
            # Add tooltips for key terms
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Key Concepts:**")
                st.markdown(tooltip('Internal Forces', 'Internal Force'), unsafe_allow_html=True)
                st.markdown(tooltip('Normal Stresses', 'Normal Stress'), unsafe_allow_html=True)
            with col2:
                st.markdown(tooltip('Displacements', 'Displacement'), unsafe_allow_html=True)
                st.markdown(tooltip('Axial Loading', 'Axial Loading'), unsafe_allow_html=True)
                st.markdown(tooltip('Cross-sectional Areas', 'Cross-sectional Area'), unsafe_allow_html=True)
            
            # Regular markdown for the rest
            st.markdown("""
            #### How to use:
            1. Enter the material properties and geometry of the rod
            2. Define the forces acting on the rod
            3. Click "Analyze Rod" to see the results
            
            #### Analysis outputs:
            - Internal forces along the rod
            - Normal stresses in each section
            - Deformations and displacements
            - Visualization plots
            """)
            
            st.markdown("<small>*Hover over highlighted terms to see explanations of engineering concepts.*</small>", unsafe_allow_html=True)
        else:
            # Keep the existing non-tooltip version
            st.markdown("""
            ### Rod Stress Analysis Tool
            
            This application analyzes the internal forces, stresses, and displacements
            of a stepped rod under axial loading. The rod consists of three sections 
            with different cross-sectional areas and lengths.
            
            #### How to use:
            1. Enter the material properties and geometry of the rod
            2. Define the forces acting on the rod
            3. Click "Analyze Rod" to see the results
            
            #### Analysis outputs:
            - Internal forces along the rod
            - Normal stresses in each section
            - Deformations and displacements
            - Visualization plots
            """)

if __name__ == "__main__":
    main()

