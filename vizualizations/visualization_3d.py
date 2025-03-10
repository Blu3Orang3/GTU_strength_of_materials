import plotly.graph_objects as go
import numpy as np

def create_3d_bar_system(a, b, forces, distributed_loads, N1, N2, d1, d2, deformation_scale=0):
    """
    Create an interactive 3D visualization of the bar system with enhanced realism.
    
    Parameters:
    - a, b: geometry parameters (m)
    - forces: list of dicts with force data
    - distributed_loads: list of dicts with distributed load data
    - N1, N2: Forces in bars (kN)
    - d1, d2: Diameters of bars (cm)
    - deformation_scale: Scale factor for deformation visualization (0 = no deformation)
    
    Returns:
    - Plotly figure object
    """
    # Calculate node positions
    A_x, A_y, A_z = 0, 0, 0
    B_x, B_y, B_z = 2*a, 0, 0
    C_x, C_y, C_z = 0, 0, 2*b
    D_x, D_y, D_z = 2*a, 0, b
    
    # Calculate bar properties
    l1 = np.sqrt((D_x - C_x)**2 + (D_y - C_y)**2 + (D_z - C_z)**2)
    l2 = np.sqrt((D_x - A_x)**2 + (D_y - A_y)**2 + (D_z - A_z)**2)
    
    sin_alpha = b / l2
    cos_alpha = 2*a / l2
    
    # Calculate deformations if requested
    if deformation_scale > 0:
        E = 210  # GPa
        A1 = np.pi * (d1/2)**2  # cm²
        A2 = np.pi * (d2/2)**2  # cm²
        
        delta_l1 = (N1 * 1000 * l1) / (E * 1e9 * A1 * 1e-4) * 1000  # mm
        delta_l2 = (N2 * 1000 * l2) / (E * 1e9 * A2 * 1e-4) * 1000  # mm
        
        # Scale for visualization (mm to m with scale factor)
        delta_l1_scaled = delta_l1 * deformation_scale / 1000  # m
        delta_l2_scaled = delta_l2 * deformation_scale / 1000  # m
        
        # Calculate deformed positions
        # For Bar 1 (C to D)
        dir1_x = (D_x - C_x) / l1
        dir1_y = (D_y - C_y) / l1
        dir1_z = (D_z - C_z) / l1
        
        # For Bar 2 (A to D)
        dir2_x = (D_x - A_x) / l2
        dir2_y = (D_y - A_y) / l2
        dir2_z = (D_z - A_z) / l2
        
        # Calculate correct deformation directions based on tension/compression
        if N1 > 0:  # Tension
            C_x_def, C_y_def, C_z_def = C_x, C_y, C_z
            D_x_def = D_x + dir1_x * delta_l1_scaled
            D_y_def = D_y + dir1_y * delta_l1_scaled
            D_z_def = D_z + dir1_z * delta_l1_scaled
        else:  # Compression
            C_x_def, C_y_def, C_z_def = C_x, C_y, C_z
            D_x_def = D_x - dir1_x * delta_l1_scaled
            D_y_def = D_y - dir1_y * delta_l1_scaled
            D_z_def = D_z - dir1_z * delta_l1_scaled
            
        # Bar 2 deformation affects D position
        if N2 > 0:  # Tension
            D_x_def += dir2_x * delta_l2_scaled
            D_y_def += dir2_y * delta_l2_scaled
            D_z_def += dir2_z * delta_l2_scaled
        else:  # Compression
            D_x_def -= dir2_x * delta_l2_scaled
            D_y_def -= dir2_y * delta_l2_scaled
            D_z_def -= dir2_z * delta_l2_scaled
    else:
        # No deformation
        C_x_def, C_y_def, C_z_def = C_x, C_y, C_z
        D_x_def, D_y_def, D_z_def = D_x, D_y, D_z
    
    # Create a new 3D figure
    fig = go.Figure()
    
    # Enhanced color palette for better visualization
    colors = {
    'tension': '#FF3D00',       # Bright red-orange for tension
    'compression': '#2979FF',   # Bright blue for compression
    'rigid': '#455A64',         # Dark bluish grey
    'support_fixed': '#607D8B', # Blue grey
    'support_roller': '#90A4AE', # Lighter blue grey
    'force': '#00C853',         # Bright green
    'ground': '#ECEFF1',        # Very light grey for ground plane
    'node_a': '#B71C1C',        # Dark red for node A
    'node_b': '#0D47A1',        # Dark blue for node B
    'node_c': '#1B5E20',        # Dark green for node C
    'node_d': '#4A148C'         # Deep purple for node D
}
    
    # Add ground plane for better perspective
    x_range = max(2.5*a, 1)  # Ensure ground is large enough
    z_range = max(2.5*b, 1)
    
    fig.add_trace(go.Surface(
        x=np.array([[-x_range, x_range], [-x_range, x_range]]),
        y=np.array([[-0.1, -0.1], [-0.1, -0.1]]),
        z=np.array([[-0.1, -0.1], [z_range, z_range]]),
        colorscale=[[0, colors['ground']], [1, colors['ground']]],
        showscale=False,
        opacity=0.3,
        name='Ground'
    ))
    
    # Create 3D cylinders for bars instead of lines for more realistic representation
    
    # Function to create cylinder path points
    def create_cylinder_mesh(x1, y1, z1, x2, y2, z2, radius, segments=12):
        # Create a ring of points around the first endpoint
        theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
        # Compute direction vector
        direction = np.array([x2-x1, y2-y1, z2-z1])
        length = np.linalg.norm(direction)
        direction = direction / length if length > 0 else np.array([0, 0, 1])
        
        # Find perpendicular vectors to create the circle
        if abs(direction[2]) < 0.9:
            # Use z-axis for constructing perpendicular
            perp1 = np.cross(direction, [0, 0, 1])
        else:
            # Use x-axis for constructing perpendicular
            perp1 = np.cross(direction, [1, 0, 0])
            
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        
        # Create the points for both ends of the cylinder
        vertices = []
        for i in range(segments):
            # Points on the first end
            point1 = np.array([x1, y1, z1]) + radius * (perp1 * np.cos(theta[i]) + perp2 * np.sin(theta[i]))
            vertices.append(point1)
            
            # Points on the second end
            point2 = np.array([x2, y2, z2]) + radius * (perp1 * np.cos(theta[i]) + perp2 * np.sin(theta[i]))
            vertices.append(point2)
        
        # Convert to arrays for plotting
        vertices = np.array(vertices)
        x = vertices[:, 0].tolist()
        y = vertices[:, 1].tolist()
        z = vertices[:, 2].tolist()
        
        # Create i, j, k indices for triangular faces
        i, j, k = [], [], []
        for p in range(segments):
            # Connect points to form triangles
            i0 = 2 * p
            i1 = 2 * p + 1
            i2 = 2 * ((p + 1) % segments)
            i3 = 2 * ((p + 1) % segments) + 1
            
            # First triangle
            i.append(i0)
            j.append(i1)
            k.append(i2)
            
            # Second triangle
            i.append(i1)
            j.append(i3)
            k.append(i2)
        
        return x, y, z, i, j, k
    
    # Add rigid elements (AC and DB) as cylinders
    # Rigid element diameter (aesthetically 30% of average bar diameter)
    rigid_diam = 0.3 * (d1 + d2) / (2 * 100)  # Convert to meters
    
    # AC rigid element
    x, y, z, i, j, k = create_cylinder_mesh(A_x, A_y, A_z, C_x, C_y, C_z, rigid_diam)
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=colors['rigid'],
        opacity=0.9,
        name='Rigid Element AC'
    ))
    
    # DB rigid element
    x, y, z, i, j, k = create_cylinder_mesh(D_x_def, D_y_def, D_z_def, B_x, B_y, B_z, rigid_diam)
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=colors['rigid'],
        opacity=0.9,
        name='Rigid Element DB'
    ))
    
    # Add bar 1 (CD) as cylinder
    bar1_state = 'tension' if N1 > 0 else 'compression'
    bar1_diam = d1 / 100  # Convert cm to meters
    x, y, z, i, j, k = create_cylinder_mesh(C_x_def, C_y_def, C_z_def, D_x_def, D_y_def, D_z_def, bar1_diam/2)
    
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=colors[bar1_state],
        opacity=0.9,
        name=f'Bar 1 ({bar1_state.capitalize()}, N₁={N1:.2f} kN)'
    ))
    
    # Add bar 2 (AD) as cylinder
    bar2_state = 'tension' if N2 > 0 else 'compression'
    bar2_diam = d2 / 100  # Convert cm to meters
    x, y, z, i, j, k = create_cylinder_mesh(A_x, A_y, A_z, D_x_def, D_y_def, D_z_def, bar2_diam/2)
    
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=colors[bar2_state],
        opacity=0.9,
        name=f'Bar 2 ({bar2_state.capitalize()}, N₂={N2:.2f} kN)'
    ))
    
    # Add nodes with enhanced appearance
    node_size = max(0.05, min(a, b) * 0.08)  # Scale based on model size
    
    # Node A
    fig.add_trace(go.Mesh3d(
        x=[x for x in np.linspace(A_x-node_size, A_x+node_size, 10) for _ in range(10) for _ in range(10)],
        y=[y for _ in range(10) for y in np.linspace(A_y-node_size, A_y+node_size, 10) for _ in range(10)],
        z=[z for _ in range(10) for _ in range(10) for z in np.linspace(A_z-node_size, A_z+node_size, 10)],
        alphahull=0,
        color=colors['node_a'],
        opacity=0.8,
        name='Node A'
    ))

    
    # Node B
    fig.add_trace(go.Mesh3d(
        x=[x for x in np.linspace(B_x-node_size, B_x+node_size, 10) for _ in range(10) for _ in range(10)],
        y=[y for _ in range(10) for y in np.linspace(B_y-node_size, B_y+node_size, 10) for _ in range(10)],
        z=[z for _ in range(10) for _ in range(10) for z in np.linspace(B_z-node_size, B_z+node_size, 10)],
        alphahull=0,
        color=colors['node_b'],
        opacity=0.8,
        name='Node B'
    ))
    
    # Node C
    fig.add_trace(go.Mesh3d(
        x=[x for x in np.linspace(C_x_def-node_size, C_x_def+node_size, 10) for _ in range(10) for _ in range(10)],
        y=[y for _ in range(10) for y in np.linspace(C_y_def-node_size, C_y_def+node_size, 10) for _ in range(10)],
        z=[z for _ in range(10) for _ in range(10) for z in np.linspace(C_z_def-node_size, C_z_def+node_size, 10)],
        alphahull=0,
        color=colors['node_c'],
        opacity=0.8,
        name='Node C'
    ))
    
    # Node D
    fig.add_trace(go.Mesh3d(
        x=[x for x in np.linspace(D_x_def-node_size, D_x_def+node_size, 10) for _ in range(10) for _ in range(10)],
        y=[y for _ in range(10) for y in np.linspace(D_y_def-node_size, D_y_def+node_size, 10) for _ in range(10)],
        z=[z for _ in range(10) for _ in range(10) for z in np.linspace(D_z_def-node_size, D_z_def+node_size, 10)],
        alphahull=0,
        color=colors['node_d'],
        opacity=0.8,
        name='Node D'
    ))
    
    # Add node labels
    fig.add_trace(go.Scatter3d(
        x=[A_x, B_x, C_x_def, D_x_def],
        y=[A_y, B_y, C_y_def, D_y_def],
        z=[A_z + 0.15, B_z + 0.15, C_z_def + 0.15, D_z_def + 0.15],
        mode='text',
        text=['A', 'B', 'C', 'D'],
        textposition='top center',
        textfont=dict(size=14, color='black', family='Arial Black'),
        showlegend=False
    ))
    
    # Add forces with improved appearance
    for i, force in enumerate(forces):
        x_pos = force['x_pos']
        y_pos = 0  # Y is always 0 in our 3D space
        z_pos = force['y_pos']  # Y coordinate in 2D becomes Z in 3D
        value = force['value']
        direction = force.get('direction', [0, -1])
        
        # In 3D: [x, y, z] where y=0 and z is the vertical direction
        # Convert 2D direction to 3D
        dir_x = direction[0]
        dir_y = 0  # Always 0 in this system
        dir_z = direction[1]  # Vertical component
        
        # Scale arrow based on force value (relative to other forces)
        max_force = max([f['value'] for f in forces])
        arrow_length = (value / max_force) * max(a, b) * 0.2
        
        # Calculate end point of arrow
        end_x = x_pos + dir_x * arrow_length
        end_y = y_pos + dir_y * arrow_length
        end_z = z_pos + dir_z * arrow_length
        
        # Add arrow
        fig.add_trace(go.Scatter3d(
            x=[x_pos, end_x],
            y=[y_pos, end_y],
            z=[z_pos, end_z],
            mode='lines',
            line=dict(color=colors['force'], width=4),
            name=f'Force {i+1} ({value} kN)'
        ))
        
        # Add a cone to represent arrow head
        fig.add_trace(go.Cone(
            x=[end_x],
            y=[end_y],
            z=[end_z],
            u=[dir_x],
            v=[dir_y],
            w=[dir_z],
            sizemode="absolute",
            sizeref=arrow_length * 0.3,
            colorscale=[[0, colors['force']], [1, colors['force']]],
            showscale=False
        ))
    
    # Add distributed loads (simplified representation)
    for i, load in enumerate(distributed_loads):
        value = load['value']
        
        if load['type'] == 'q1':  # Load on CD segment
            start_x, start_y, start_z = C_x, C_y, C_z
            end_x, end_y, end_z = D_x, D_y, D_z
        else:  # q2, load on DB segment
            start_x, start_y, start_z = D_x, D_y, D_z
            end_x, end_y, end_z = B_x, B_y, B_z
        
        # Create 5 equally spaced points along the segment
        for j in range(5):
            x = start_x + j*(end_x-start_x)/4
            y = start_y + j*(end_y-start_y)/4
            z = start_z + j*(end_z-start_z)/4
            
            # Add small arrow for distributed load
            arrow_length = max(a, b) * 0.1
            fig.add_trace(go.Scatter3d(
                x=[x, x],
                y=[y, y],
                z=[z, z-arrow_length],
                mode='lines',
                line=dict(color=colors['force'], width=2),
                showlegend=(j == 0),  # Only show in legend for first arrow
                name=f'q{i+1} = {value} kN/m'
            ))
    
    # Add supports
    # Support at A (fixed)
    fig.add_trace(go.Mesh3d(
        x=[A_x-0.1*a, A_x+0.1*a, A_x+0.1*a, A_x-0.1*a, A_x-0.1*a, A_x+0.1*a, A_x+0.1*a, A_x-0.1*a],
        y=[A_y-0.1*a, A_y-0.1*a, A_y+0.1*a, A_y+0.1*a, A_y-0.1*a, A_y-0.1*a, A_y+0.1*a, A_y+0.1*a],
        z=[A_z, A_z, A_z, A_z, A_z-0.1*a, A_z-0.1*a, A_z-0.1*a, A_z-0.1*a],
        color=colors['support_fixed'],
        opacity=0.8,
        name='Fixed Support A'
    ))
    
    # Support at B (roller)
    fig.add_trace(go.Mesh3d(
        x=[B_x-0.1*a, B_x+0.1*a, B_x+0.1*a, B_x-0.1*a, B_x-0.1*a, B_x+0.1*a, B_x+0.1*a, B_x-0.1*a],
        y=[B_y-0.1*a, B_y-0.1*a, B_y+0.1*a, B_y+0.1*a, B_y-0.1*a, B_y-0.1*a, B_y+0.1*a, B_y+0.1*a],
        z=[B_z, B_z, B_z, B_z, B_z-0.05*a, B_z-0.05*a, B_z-0.05*a, B_z-0.05*a],
        color=colors['support_roller'],
        opacity=0.8,
        name='Roller Support B'
    ))
    
    # Add coordinate system
    length = min(a, b) * 0.3
    
    # X-axis (red)
    fig.add_trace(go.Scatter3d(
        x=[0, length],
        y=[0, 0],
        z=[0, 0],
        mode='lines',
        line=dict(color='red', width=3),
        name='X-axis'
    ))
    
    # Y-axis (green)
    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, length],
        z=[0, 0],
        mode='lines',
        line=dict(color='green', width=3),
        name='Y-axis'
    ))
    
    # Z-axis (blue)
    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, length],
        mode='lines',
        line=dict(color='blue', width=3),
        name='Z-axis'
    ))
    
    # Set layout
    fig.update_layout(
        title=f"Bar System Analysis - 3D View {'(Deformed)' if deformation_scale > 0 else ''}",
        scene=dict(
            xaxis_title='X [m]',
            yaxis_title='Y [m]',
            zaxis_title='Z [m]',
            aspectmode='data',
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    
    # Add deformation info if showing deformed shape
    if deformation_scale > 0:
        delta_C = np.sqrt(delta_l1**2 + (delta_l2*sin_alpha)**2)
        annotation_text = f"Max Displacement: {delta_C:.2f} mm (Scale: {deformation_scale}x)"
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=annotation_text,
            showarrow=False,
            font=dict(size=12, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )
    
    return fig
