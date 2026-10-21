
import numpy as np  


def chimerax_add_sphere(center, radius=.3, color='red', alpha=1, name=None):
    """
    Generate ChimeraX command to add a sphere to the scene
    
    Args:
        center: (x, y, z) coordinates for sphere center
        radius: sphere radius (default: 1.0)
        color: sphere color name or RGB (default: 'red')
        alpha: transparency (0.0-1.0, default: 1.0)
        name: optional name for the sphere object
        
    Returns:
        str: ChimeraX command string
    """
    x, y, z = center
    
    # Base command
    cmd = f"shape sphere center {x:.3f},{y:.3f},{z:.3f} radius {radius:.3f}"
    
    # Add color
    cmd += f" color {color}"
    
    # Add transparency if not fully opaque
    if alpha < 1.0:
        cmd += f" transparency {1.0 - alpha:.2f}"
    
    # Add name if specified
    if name:
        cmd += f" name {name}"
    
    return cmd


def chimerax_add_arrow(start, vector, radius=0.2, tip_length=1.0, tip_radius=0.4,
                       color='red', alpha=1.0, name=None):
    """
    Generate ChimeraX commands to add an arrow (cylinder + cone) to the scene.
    
    Args:
        start: (x, y, z) coordinates of the arrow starting point
        vector: (vx, vy, vz) vector defining direction and length
        radius: cylinder radius (default: 0.2)
        tip_length: length of the arrow head cone (default: 1.0)
        tip_radius: radius of the arrow head cone base (default: 0.4)
        color: arrow color (default: 'red')
        alpha: transparency (0.0-1.0, default: 1.0)
        name: optional name for the arrow object
        
    Returns:
        str: ChimeraX command string (cylinder + cone)
    """
    start = np.array(start, dtype=float)
    end = start + np.array(vector, dtype=float)

    # Compute vector length
    vec = np.array(vector, dtype=float)
    length = np.linalg.norm(vec)
    if tip_length >= length:
        raise ValueError("Arrow head is longer than or equal to the vector length.")

    # Point where the cylinder ends and the cone starts
    base_cone = end - vec/length * tip_length

    # Cylinder command
    cyl_cmd = (
        f"shape cylinder fromPoint {start[0]:.3f},{start[1]:.3f},{start[2]:.3f} "
        f"toPoint {base_cone[0]:.3f},{base_cone[1]:.3f},{base_cone[2]:.3f} "
        f"radius {radius:.3f} color {color}"
    )

    # Cone command
    cone_cmd = (
        f"shape cone fromPoint {base_cone[0]:.3f},{base_cone[1]:.3f},{base_cone[2]:.3f} "
        f"toPoint {end[0]:.3f},{end[1]:.3f},{end[2]:.3f} "
        f"radius {tip_radius:.3f} color {color}"
    )

    # Add transparency if not fully opaque
    if alpha < 1.0:
        transparency = 1.0 - alpha
        cyl_cmd += f" transparency {transparency:.2f}"
        cone_cmd += f" transparency {transparency:.2f}"

    # Add name if specified
    if name:
        cyl_cmd += f" name {name}_shaft"
        cone_cmd += f" name {name}_head"

    return cyl_cmd + "\n" + cone_cmd