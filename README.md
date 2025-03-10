# Rod Stress Analysis Application

This Streamlit application analyzes internal forces, stresses, and displacements in a stepped rod under axial loading. The application provides a user-friendly interface for engineers and students to perform structural analysis on rods with varying cross-sections.

## Application Purpose

The Rod Stress Analysis Tool serves as an educational and engineering utility for analyzing how axial forces affect a stepped rod (a rod with multiple sections of different cross-sectional areas). It calculates:

- Internal axial forces in each rod section
- Normal stresses in the material
- Deformations of each section
- Cumulative displacements throughout the rod

This tool visualizes these engineering concepts through interactive inputs and detailed graphical outputs.

## Key Features

- **Interactive Interface**: Users can directly input rod parameters through a clean sidebar interface
- **Dynamic Visualization**: Real-time visual representation of the rod configuration
- **Multiple Force Analysis**: Support for applying multiple forces at different positions and directions
- **Comprehensive Results**: Detailed numerical results and publication-quality plots
- **Data Export**: Download results as CSV for further analysis

## Technical Implementation

The application implements:

1. **Rod Modeling**: The rod is modeled as three connected sections, each with customizable length and cross-sectional area
2. **Force Application**: Users can apply any number of forces along the rod, specifying magnitude, direction, and position
3. **Mechanics Calculation**: Implements strength of materials equations to calculate internal forces, stresses, and displacements
4. **Interactive Visualization**: Three specialized charts showing:
   - Internal force distribution
   - Stress distribution (with tension/compression differentiation)
   - Displacement distribution along the rod

## How It Works

The application follows solid mechanics principles to analyze the stepped rod:
- Forces are resolved to calculate internal normal forces at each section interface
- Stresses are calculated based on force and cross-sectional area (σ = F/A)
- Deformations are computed using Hooke's law (ε = σ/E) and the section lengths
- Cumulative displacements are determined by summing deformations from fixed end

## Usage

1. Set material properties (Young's modulus)
2. Define geometry (section lengths and cross-sectional areas)
3. Add forces acting on the rod (magnitude, direction, position)
4. Click "Analyze Rod" to perform calculations
5. View results numerically and graphically
6. Export results if needed

## Dependencies

- streamlit
- numpy
- pandas
- matplotlib

## Educational Value

This tool helps engineering students and professionals understand:
- How forces propagate through a rod with changing cross-sections
- The relationship between geometry and stress distribution
- Principles of deformation and displacement under axial loading

The visualizations make it easier to understand these concepts compared to manual calculations, providing immediate feedback as parameters change.
