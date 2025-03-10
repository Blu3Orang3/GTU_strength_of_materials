import streamlit as st
from typing import Dict

# Dictionary of engineering concepts and their explanations
ENGINEERING_TOOLTIPS: Dict[str, str] = {
    "Young's Modulus": "A mechanical property that measures the stiffness of a solid material. "
                      "It defines the relationship between stress (force per unit area) and strain "
                      "(proportional deformation) in a material in the linear elasticity regime.",
    
    "Normal Stress": "The internal force per unit area that acts perpendicular to the cross-sectional area. "
                    "Tensile stress is positive, and compressive stress is negative.",
    
    "Internal Force": "Forces that act between adjacent parts of a body, holding the parts together and "
                     "maintaining the shape of the body.",
    
    "Tension": "A pulling force that stretches a material. Tension forces result in elongation of the material "
              "in the direction of the applied force.",
    
    "Compression": "A pushing force that shortens a material. Compression forces result in shortening of the "
                  "material in the direction of the applied force.",
    
    "Displacement": "The change in position of a point relative to its original position. In rod analysis, "
                   "it's typically the movement in the axial direction.",
    
    "Deformation": "The change in the shape or size of an object due to an applied force. For a rod, "
                  "this is the change in length due to axial loading.",
    
    "Strain": "The measure of deformation representing the displacement between particles in a material "
             "relative to a reference length. Strain is dimensionless (change in length / original length).",
    
    "Cross-sectional Area": "The area of a plane surface obtained by cutting perpendicular to the longitudinal axis. "
                           "It's used to calculate stress from force (stress = force / area).",
    
    "Stepped Rod": "A rod with sections of different cross-sectional areas. This change in geometry affects "
                  "how stresses and displacements develop along the rod.",
    
    "Axial Loading": "Forces applied along the longitudinal axis of a structural member. "
                    "They cause tension or compression in the member.",
    
    "Hooke's Law": "States that the force needed to extend or compress a spring is proportional to the distance it's extended. "
                  "For materials, this means stress is proportional to strain in the elastic region (σ = Eε).",
    
    "Stress-Strain Diagram": "A graph showing the relationship between stress and strain in a material, "
                            "used to determine mechanical properties like elastic limit and yield strength."
}

def create_tooltip_style():
    """Add CSS for tooltips"""
    return st.markdown("""
    <style>
    .tooltip {
        position: relative;
        display: inline-block;
        color: #1E3A8A;
        font-weight: 500;
        text-decoration: underline dotted;
        cursor: help;
    }
    
    .tooltip .tooltip-text {
        visibility: hidden;
        width: 200px;
        background-color: #2E3440;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 999;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

def tooltip(text, concept):
    """Create a tooltip with given text and explanation"""
    if concept in ENGINEERING_TOOLTIPS:
        explanation = ENGINEERING_TOOLTIPS[concept]
        tooltip_html = f"""<span class="tooltip">{text}<span class="tooltip-text">{explanation}</span></span>"""
        return tooltip_html
    return text

def concept_tooltip(concept: str) -> str:
    """
    Create a tooltip using the concept name as both the display text and lookup key.
    
    Args:
        concept: The engineering concept name
        
    Returns:
        HTML string containing the concept with tooltip
    """
    return tooltip(concept, concept)
