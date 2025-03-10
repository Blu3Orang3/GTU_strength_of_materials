import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

def draw_bar_system_with_multiple_loads(a, b, forces, distributed_loads, N1, N2, deform_scale=1, 
                                       d1=None, d2=None, delta_l1=None, delta_l2=None, sin_alpha=None):
    """Draw the bar system with forces and deformation"""
    # If delta_l values are not provided, calculate them
    if delta_l1 is None or delta_l2 is None or sin_alpha is None:
        # Use default calculation method
        plt.style.use('seaborn')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Improved color scheme with better contrast
        colors = {
            'tension': '#FF5733',    # Bright orange-red
            'compression': '#3498DB', # Bright blue
            'rigid': '#2C3E50',      # Dark slate
            'deformed': '#7F8C8D',   # Gray
            'force': '#27AE60',      # Green
            'support': '#95A5A6'     # Light gray
        }
        
        # Determine bar states for visual coding
        bar1_state = 'tension' if N1 > 0 else 'compression'
        bar2_state = 'tension' if N2 > 0 else 'compression'
        
        # More visible line styles
        bar1_style = '-' if N1 > 0 else '--'
        bar2_style = '-' if N2 > 0 else '--'
        
        # Calculate stress for color intensity
        stress1 = abs(N1 * 1000) / (np.pi * (d1/2)**2)  # MPa
        stress2 = abs(N2 * 1000) / (np.pi * (d2/2)**2)  # MPa
        
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
        delta_l1_scaled = delta_l1 * deform_scale / 1000  # m
        delta_l2_scaled = delta_l2 * deform_scale / 1000  # m
        
        # Calculate deformed positions
        dx_C = 0
        dy_C = delta_l1_scaled if N1 > 0 else -delta_l1_scaled
        
        dx_D = delta_l2_scaled * cos_alpha if N2 > 0 else -delta_l2_scaled * cos_alpha
        dy_D = delta_l2_scaled * sin_alpha if N2 > 0 else -delta_l2_scaled * sin_alpha
        
        C_x_def, C_y_def = C_x + dx_C, C_y + dy_C
        D_x_def, D_y_def = D_x + dx_D, D_y + dy_D

        # Draw undeformed system with improved styling
        # Rigid elements
        ax.plot([A_x, C_x], [A_y, C_y], color=colors['rigid'], linewidth=3, solid_capstyle='round')
        ax.plot([D_x, B_x], [D_y, B_y], color=colors['rigid'], linewidth=3, solid_capstyle='round')
        
        # Bars with thickness proportional to diameter but with minimum visibility
        ax.plot([C_x, D_x], [C_y, D_y], color=colors[bar1_state], 
                linestyle=bar1_style, linewidth=max(2, d1/8), 
                label=f'Bar 1 ({bar1_state.capitalize()})')
        
        ax.plot([A_x, D_x], [A_y, D_y], color=colors[bar2_state], 
                linestyle=bar2_style, linewidth=max(2, d2/8), 
                label=f'Bar 2 ({bar2_state.capitalize()})')
        
        # Draw deformed system with dashed lines in gray color
        ax.plot([A_x, C_x_def], [A_y, C_y_def], color=colors['deformed'], linestyle='--', linewidth=1)
        ax.plot([C_x_def, D_x_def], [C_y_def, D_y_def], color=colors['deformed'], linestyle='--', linewidth=1)
        ax.plot([D_x_def, B_x], [D_y_def, B_y], color=colors['deformed'], linestyle='--', linewidth=1)
        ax.plot([A_x, D_x_def], [A_y, D_y_def], color=colors['deformed'], linestyle='--', linewidth=1)
        
        # Improved support symbols
        # Fixed support at A with hatching
        support_width = 0.15 * max(a, b)
        support_height = 0.12 * max(a, b)
        
        # Draw fixed support with hatching
        ax.add_patch(Rectangle((A_x-support_width/2, A_y-support_height), 
                     support_width, support_height, 
                     fc=colors['support'], ec='black', hatch='//'))
        
        # Add small triangles for the fixed support
        triangle_size = support_width/6
        for i in range(5):
            x = A_x - support_width/2 + i * support_width/4
            ax.fill([x, x + triangle_size, x - triangle_size], 
                    [A_y-support_height, A_y-support_height-triangle_size, A_y-support_height-triangle_size],
                    color='black')
        
        # Roller support at B with circle
        circle_radius = support_width/3
        ax.add_patch(Circle((B_x, B_y-circle_radius), 
                     circle_radius, fc=colors['support'], ec='black'))
        # Add a line above the roller
        ax.plot([B_x-support_width/2, B_x+support_width/2], 
                [B_y, B_y], color='black', linewidth=2)

        # Add reaction force arrows at supports
        reaction_scale = 0.2 * max(a, b)
        
        # Example reaction forces - replace with actual calculations in your app
        Ya = 20  # Placeholder - replace with actual value
        Xa = 10  # Placeholder - replace with actual value
        Yb = 15  # Placeholder - replace with actual value
        
        # Vertical reaction at A
        if Ya != 0:
            direction = 1 if Ya > 0 else -1
            ax.arrow(A_x, A_y, 0, direction * reaction_scale, 
                    head_width=0.05*max(a, b), head_length=0.05*max(a, b), 
                    fc='purple', ec='purple', width=0.02*max(a, b))
            ax.text(A_x+0.1*a, A_y+0.05*b, f'Ya = {abs(Ya):.2f} kN', 
                   fontsize=9, color='purple')
        
        # Horizontal reaction at A
        if Xa != 0:
            direction = 1 if Xa > 0 else -1
            ax.arrow(A_x, A_y, direction * reaction_scale, 0, 
                    head_width=0.05*max(a, b), head_length=0.05*max(a, b), 
                    fc='purple', ec='purple', width=0.02*max(a, b))
            ax.text(A_x+0.05*a, A_y-0.15*b, f'Xa = {abs(Xa):.2f} kN', 
                   fontsize=9, color='purple')
        
        # Vertical reaction at B
        if Yb != 0:
            direction = 1 if Yb > 0 else -1
            ax.arrow(B_x, B_y, 0, direction * reaction_scale, 
                    head_width=0.05*max(a, b), head_length=0.05*max(a, b), 
                    fc='purple', ec='purple', width=0.02*max(a, b))
            ax.text(B_x-0.2*a, B_y+0.05*b, f'Yb = {abs(Yb):.2f} kN', fontsize=9, color='purple')
        
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
                    head_length=0.05*max(a, b), fc=colors['force'], ec=colors['force'], width=0.02*max(a, b))
            
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
                        head_length=0.03*max(a, b), fc=colors['force'], ec=colors['force'], width=0.01*max(a, b))
            
            # Add label
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            ax.text(mid_x, mid_y-q_scale, f'q{i+1}={value} kN/m', fontsize=10)
        
        # Labels
        ax.text(A_x-0.15*a, A_y+0.15*b, 'A', fontsize=12, fontweight='bold')
        ax.text(B_x+0.15*a, B_y+0.15*b, 'B', fontsize=12, fontweight='bold')
        ax.text(C_x-0.15*a, C_y+0.15*b, 'C', fontsize=12, fontweight='bold')
        ax.text(D_x+0.15*a, D_y+0.15*b, 'D', fontsize=12, fontweight='bold')
        
        # Forces labels with improved formatting
        if N1 > 0:
            ax.text((C_x+D_x)/2+0.1*a, (C_y+D_y)/2+0.1*b, f'N1={N1:.2f} kN (T)', 
                    color=colors['tension'], fontsize=10, fontweight='bold')
        else:
            ax.text((C_x+D_x)/2+0.1*a, (C_y+D_y)/2+0.1*b, f'N1={abs(N1):.2f} kN (C)', 
                    color=colors['compression'], fontsize=10, fontweight='bold')
            
        if N2 > 0:
            ax.text((A_x+D_x)/2+0.1*a, (A_y+D_y)/2-0.1*b, f'N2={N2:.2f} kN (T)', 
                    color=colors['tension'], fontsize=10, fontweight='bold')
        else:
            ax.text((A_x+D_x)/2+0.1*a, (A_y+D_y)/2-0.1*b, f'N2={abs(N2):.2f} kN (C)', 
                    color=colors['compression'], fontsize=10, fontweight='bold')
        
        # Add deformation measurements
        # Calculate maximum displacement for reference
        delta_C = np.sqrt(delta_l1**2 + (delta_l2*sin_alpha)**2)
        scaled_delta_C = delta_C * deform_scale  # reflect deformation scale factor
        allowable_disp = 20  # Placeholder - replace with actual value

        # Add deformation scale reference
        ax.text(0.02, 0.98, f'Max Displacement: {scaled_delta_C:.2f} mm\nAllowable: {allowable_disp:.1f} mm', 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add a visual scale bar for deformation
        if deform_scale > 0:
            scale_length = 1.0  # 1m in real scale
            ax.plot([A_x, A_x+scale_length], [A_y-0.7*b, A_y-0.7*b], 'k-', linewidth=2)
            ax.plot([A_x, A_x+scale_length*deform_scale], [A_y-0.8*b, A_y-0.8*b], 
                   color='red', linewidth=2)
            ax.text(A_x+scale_length/2, A_y-0.75*b, 
                   f'Scale: {deform_scale}x', fontsize=9, ha='center')

        # Add a small colorbar showing stress levels
        # Define colormaps for tension and compression
        tension_cmap = LinearSegmentedColormap.from_list('tension', ['yellow', 'red'])
        compression_cmap = LinearSegmentedColormap.from_list('compression', ['lightblue', 'darkblue'])
        
        # Position for small colorbars
        cbar_len = 0.15
        cbar_height = 0.01
        
        # Assumption for max stress reference (R) - replace with actual value
        R = max(stress1, stress2)
        
        # Create a separate axes for the colorbar
        cax_tension = fig.add_axes([0.15, 0.05, cbar_len, cbar_height])
        cax_compression = fig.add_axes([0.15, 0.08, cbar_len, cbar_height])
        
        # Create gradient arrays
        gradient_tension = np.linspace(0, 1, 256).reshape(1, -1)
        gradient_compression = np.linspace(0, 1, 256).reshape(1, -1)
        
        # Plot the colormaps
        cax_tension.imshow(gradient_tension, aspect='auto', cmap=tension_cmap)
        cax_compression.imshow(gradient_compression, aspect='auto', cmap=compression_cmap)
        
        # Remove ticks
        cax_tension.set_xticks([])
        cax_tension.set_yticks([])
        cax_compression.set_xticks([])
        cax_compression.set_yticks([])
        
        # Add labels
        cax_tension.text(0, -0.5, '0 MPa', transform=cax_tension.transAxes,
                         ha='left', va='top', fontsize=8)
        cax_tension.text(1, -0.5, f'{R:.0f} MPa (Tension)', transform=cax_tension.transAxes,
                         ha='right', va='top', fontsize=8)
        cax_compression.text(0, -0.5, '0 MPa', transform=cax_compression.transAxes,
                             ha='left', va='top', fontsize=8)
        cax_compression.text(1, -0.5, f'{R:.0f} MPa (Compression)', transform=cax_compression.transAxes,
                             ha='right', va='top', fontsize=8)

        # Improved title and labels
        ax.set_title('Bar System Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('x [m]', fontsize=12)
        ax.set_ylabel('y [m]', fontsize=12)
        
        # Add grid with transparency
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Enhanced legend with custom elements
        legend_elements = [
            Line2D([0], [0], color=colors['tension'], lw=4, linestyle='-',
                  label='Tension'),
            Line2D([0], [0], color=colors['compression'], lw=4, linestyle='--',
                  label='Compression'),
            Line2D([0], [0], color=colors['rigid'], lw=3,
                  label='Rigid Element'),
            Line2D([0], [0], color=colors['deformed'], lw=2, linestyle='--',
                  label='Deformed Shape'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 framealpha=0.9, fontsize=10)
        
        # Set axis limits with margins
        margin = 0.5 * max(a, b)
        ax.set_xlim(min(A_x, C_x)-margin, max(B_x, D_x)+margin)
        ax.set_ylim(min(A_y, B_y)-margin-q_scale, max(C_y, D_y)+margin)
        
        # Equal aspect ratio
        ax.set_aspect('equal')
        
        return fig
    else:
        # Use the pre-calculated values for consistent visualization
        delta_l1_scaled = delta_l1 * deform_scale / 1000  # Convert mm to m and apply scale
        delta_l2_scaled = delta_l2 * deform_scale / 1000
        
        # Create plot with the same styling as above
        plt.style.use('seaborn')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Improved color scheme with better contrast
        colors = {
            'tension': '#FF5733',    # Bright orange-red
            'compression': '#3498DB', # Bright blue
            'rigid': '#2C3E50',      # Dark slate
            'deformed': '#7F8C8D',   # Gray
            'force': '#27AE60',      # Green
            'support': '#95A5A6'     # Light gray
        }
        
        # Determine bar states for visual coding
        bar1_state = 'tension' if N1 > 0 else 'compression'
        bar2_state = 'tension' if N2 > 0 else 'compression'
        
        # More visible line styles
        bar1_style = '-' if N1 > 0 else '--'
        bar2_style = '-' if N2 > 0 else '--'
        
        # Calculate stress for color intensity
        stress1 = abs(N1 * 1000) / (np.pi * (d1/2)**2)  # MPa
        stress2 = abs(N2 * 1000) / (np.pi * (d2/2)**2)  # MPa
        
        # Calculate node positions
        A_x, A_y = 0, 0
        B_x, B_y = 2*a, 0
        C_x, C_y = 0, 2*b
        D_x, D_y = 2*a, b
        
        # Calculate bar properties
        l1 = np.sqrt((D_x - C_x)**2 + (D_y - C_y)**2)
        l2 = np.sqrt((D_x - A_x)**2 + (D_y - A_y)**2)
        
        # Use provided sin_alpha or recalculate if needed
        if sin_alpha is None:
            sin_alpha = b / l2
        cos_alpha = 2*a / l2
        
        # Calculate deformed positions using the provided delta_l values
        dx_C = 0
        dy_C = delta_l1_scaled if N1 > 0 else -delta_l1_scaled
        
        dx_D = delta_l2_scaled * cos_alpha if N2 > 0 else -delta_l2_scaled * cos_alpha
        dy_D = delta_l2_scaled * sin_alpha if N2 > 0 else -delta_l2_scaled * sin_alpha
        
        C_x_def, C_y_def = C_x + dx_C, C_y + dy_C
        D_x_def, D_y_def = D_x + dx_D, D_y + dy_D

        # Draw undeformed system with improved styling
        # Rigid elements
        ax.plot([A_x, C_x], [A_y, C_y], color=colors['rigid'], linewidth=3, solid_capstyle='round')
        ax.plot([D_x, B_x], [D_y, B_y], color=colors['rigid'], linewidth=3, solid_capstyle='round')
        
        # Bars with thickness proportional to diameter but with minimum visibility
        ax.plot([C_x, D_x], [C_y, D_y], color=colors[bar1_state], 
                linestyle=bar1_style, linewidth=max(2, d1/8), 
                label=f'Bar 1 ({bar1_state.capitalize()})')
        
        ax.plot([A_x, D_x], [A_y, D_y], color=colors[bar2_state], 
                linestyle=bar2_style, linewidth=max(2, d2/8), 
                label=f'Bar 2 ({bar2_state.capitalize()})')
        
        # Draw deformed system with dashed lines in gray color
        ax.plot([A_x, C_x_def], [A_y, C_y_def], color=colors['deformed'], linestyle='--', linewidth=1)
        ax.plot([C_x_def, D_x_def], [C_y_def, D_y_def], color=colors['deformed'], linestyle='--', linewidth=1)
        ax.plot([D_x_def, B_x], [D_y_def, B_y], color=colors['deformed'], linestyle='--', linewidth=1)
        ax.plot([A_x, D_x_def], [A_y, D_y_def], color=colors['deformed'], linestyle='--', linewidth=1)
        
        # Improved support symbols
        # Fixed support at A with hatching
        support_width = 0.15 * max(a, b)
        support_height = 0.12 * max(a, b)
        
        # Draw fixed support with hatching
        ax.add_patch(Rectangle((A_x-support_width/2, A_y-support_height), 
                     support_width, support_height, 
                     fc=colors['support'], ec='black', hatch='//'))
        
        # Add small triangles for the fixed support
        triangle_size = support_width/6
        for i in range(5):
            x = A_x - support_width/2 + i * support_width/4
            ax.fill([x, x + triangle_size, x - triangle_size], 
                    [A_y-support_height, A_y-support_height-triangle_size, A_y-support_height-triangle_size],
                    color='black')
        
        # Roller support at B with circle
        circle_radius = support_width/3
        ax.add_patch(Circle((B_x, B_y-circle_radius), 
                     circle_radius, fc=colors['support'], ec='black'))
        # Add a line above the roller
        ax.plot([B_x-support_width/2, B_x+support_width/2], 
                [B_y, B_y], color='black', linewidth=2)

        # Add reaction force arrows at supports
        reaction_scale = 0.2 * max(a, b)
        
        # Example reaction forces - replace with actual calculations in your app
        Ya = 20  # Placeholder - replace with actual value
        Xa = 10  # Placeholder - replace with actual value
        Yb = 15  # Placeholder - replace with actual value
        
        # Vertical reaction at A
        if Ya != 0:
            direction = 1 if Ya > 0 else -1
            ax.arrow(A_x, A_y, 0, direction * reaction_scale, 
                    head_width=0.05*max(a, b), head_length=0.05*max(a, b), 
                    fc='purple', ec='purple', width=0.02*max(a, b))
            ax.text(A_x+0.1*a, A_y+0.05*b, f'Ya = {abs(Ya):.2f} kN', 
                   fontsize=9, color='purple')
        
        # Horizontal reaction at A
        if Xa != 0:
            direction = 1 if Xa > 0 else -1
            ax.arrow(A_x, A_y, direction * reaction_scale, 0, 
                    head_width=0.05*max(a, b), head_length=0.05*max(a, b), 
                    fc='purple', ec='purple', width=0.02*max(a, b))
            ax.text(A_x+0.05*a, A_y-0.15*b, f'Xa = {abs(Xa):.2f} kN', 
                   fontsize=9, color='purple')
        
        # Vertical reaction at B
        if Yb != 0:
            direction = 1 if Yb > 0 else -1
            ax.arrow(B_x, B_y, 0, direction * reaction_scale, 
                    head_width=0.05*max(a, b), head_length=0.05*max(a, b), 
                    fc='purple', ec='purple', width=0.02*max(a, b))
            ax.text(B_x-0.2*a, B_y+0.05*b, f'Yb = {abs(Yb):.2f} kN', fontsize=9, color='purple')
        
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
                    head_length=0.05*max(a, b), fc=colors['force'], ec=colors['force'], width=0.02*max(a, b))
            
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
                        head_length=0.03*max(a, b), fc=colors['force'], ec=colors['force'], width=0.01*max(a, b))
            
            # Add label
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            ax.text(mid_x, mid_y-q_scale, f'q{i+1}={value} kN/m', fontsize=10)
        
        # Labels
        ax.text(A_x-0.15*a, A_y+0.15*b, 'A', fontsize=12, fontweight='bold')
        ax.text(B_x+0.15*a, B_y+0.15*b, 'B', fontsize=12, fontweight='bold')
        ax.text(C_x-0.15*a, C_y+0.15*b, 'C', fontsize=12, fontweight='bold')
        ax.text(D_x+0.15*a, D_y+0.15*b, 'D', fontsize=12, fontweight='bold')
        
        # Forces labels with improved formatting
        if N1 > 0:
            ax.text((C_x+D_x)/2+0.1*a, (C_y+D_y)/2+0.1*b, f'N1={N1:.2f} kN (T)', 
                    color=colors['tension'], fontsize=10, fontweight='bold')
        else:
            ax.text((C_x+D_x)/2+0.1*a, (C_y+D_y)/2+0.1*b, f'N1={abs(N1):.2f} kN (C)', 
                    color=colors['compression'], fontsize=10, fontweight='bold')
            
        if N2 > 0:
            ax.text((A_x+D_x)/2+0.1*a, (A_y+D_y)/2-0.1*b, f'N2={N2:.2f} kN (T)', 
                    color=colors['tension'], fontsize=10, fontweight='bold')
        else:
            ax.text((A_x+D_x)/2+0.1*a, (A_y+D_y)/2-0.1*b, f'N2={abs(N2):.2f} kN (C)', 
                    color=colors['compression'], fontsize=10, fontweight='bold')
        
        # Add deformation measurements
        # Calculate maximum displacement
        delta_C = np.sqrt(delta_l1**2 + (delta_l2*sin_alpha)**2)
        scaled_delta_C = delta_C * deform_scale  # reflect deformation scale factor
        allowable_disp = 20  # Placeholder - replace with actual value
        
        # Add deformation scale reference
        ax.text(0.02, 0.98, f'Max Displacement: {scaled_delta_C:.2f} mm\nAllowable: {allowable_disp:.1f} mm', 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add a visual scale bar for deformation
        if deform_scale > 0:
            scale_length = 1.0  # 1m in real scale
            ax.plot([A_x, A_x+scale_length], [A_y-0.7*b, A_y-0.7*b], 'k-', linewidth=2)
            ax.plot([A_x, A_x+scale_length*deform_scale], [A_y-0.8*b, A_y-0.8*b], 
                   color='red', linewidth=2)
            ax.text(A_x+scale_length/2, A_y-0.75*b, 
                   f'Scale: {deform_scale}x', fontsize=9, ha='center')

        # Add a small colorbar showing stress levels
        # Define colormaps for tension and compression
        tension_cmap = LinearSegmentedColormap.from_list('tension', ['yellow', 'red'])
        compression_cmap = LinearSegmentedColormap.from_list('compression', ['lightblue', 'darkblue'])
        
        # Position for small colorbars
        cbar_len = 0.15
        cbar_height = 0.01
        
        # Assumption for max stress reference (R) - replace with actual value
        R = max(stress1, stress2)
        
        # Create a separate axes for the colorbar
        cax_tension = fig.add_axes([0.15, 0.05, cbar_len, cbar_height])
        cax_compression = fig.add_axes([0.15, 0.08, cbar_len, cbar_height])
        
        # Create gradient arrays
        gradient_tension = np.linspace(0, 1, 256).reshape(1, -1)
        gradient_compression = np.linspace(0, 1, 256).reshape(1, -1)
        
        # Plot the colormaps
        cax_tension.imshow(gradient_tension, aspect='auto', cmap=tension_cmap)
        cax_compression.imshow(gradient_compression, aspect='auto', cmap=compression_cmap)
        
        # Remove ticks
        cax_tension.set_xticks([])
        cax_tension.set_yticks([])
        cax_compression.set_xticks([])
        cax_compression.set_yticks([])
        
        # Add labels
        cax_tension.text(0, -0.5, '0 MPa', transform=cax_tension.transAxes,
                         ha='left', va='top', fontsize=8)
        cax_tension.text(1, -0.5, f'{R:.0f} MPa (Tension)', transform=cax_tension.transAxes,
                         ha='right', va='top', fontsize=8)
        cax_compression.text(0, -0.5, '0 MPa', transform=cax_compression.transAxes,
                             ha='left', va='top', fontsize=8)
        cax_compression.text(1, -0.5, f'{R:.0f} MPa (Compression)', transform=cax_compression.transAxes,
                             ha='right', va='top', fontsize=8)

        # Improved title and labels
        ax.set_title('Bar System Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('x [m]', fontsize=12)
        ax.set_ylabel('y [m]', fontsize=12)
        
        # Add grid with transparency
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Enhanced legend with custom elements
        legend_elements = [
            Line2D([0], [0], color=colors['tension'], lw=4, linestyle='-',
                  label='Tension'),
            Line2D([0], [0], color=colors['compression'], lw=4, linestyle='--',
                  label='Compression'),
            Line2D([0], [0], color=colors['rigid'], lw=3,
                  label='Rigid Element'),
            Line2D([0], [0], color=colors['deformed'], lw=2, linestyle='--',
                  label='Deformed Shape'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 framealpha=0.9, fontsize=10)
        
        # Set axis limits with margins
        margin = 0.5 * max(a, b)
        ax.set_xlim(min(A_x, C_x)-margin, max(B_x, D_x)+margin)
        ax.set_ylim(min(A_y, B_y)-margin-q_scale, max(C_y, D_y)+margin)
        
        # Equal aspect ratio
        ax.set_aspect('equal')
        
        return fig
