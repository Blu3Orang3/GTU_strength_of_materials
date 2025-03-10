import streamlit as st

st.set_page_config(
    page_title="Engineering Tools",
    page_icon="🔧",
    layout="wide"
)

st.title("Engineering Analysis Tools")
st.markdown("""
This platform provides various engineering analysis tools:

- **Rod Stress Analysis**: Analyze internal forces, stresses, and displacements in stepped rods
- **Beam Analysis**: Calculate deflections and internal forces in beams
- **Column Buckling**: Determine critical buckling loads for columns
- **Truss Analysis**: Analyze forces in truss structures

Select a tool from the sidebar to begin your analysis.
""")

# Show some common information
st.subheader("About")
st.markdown("""
These tools are designed for educational purposes and engineering analysis.
Each tool provides interactive inputs and visualizations to help understand 
structural mechanics concepts.
""")