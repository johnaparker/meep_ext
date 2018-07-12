import meep
import math

def x_polarized_plane_wave(sim, src_time, amplitude=1.0):
    """add an x-polarized plane wave to sim with time profile and amplitude"""
    # assume first boundary layer is the full PML
    pml_thickness = sim.boundary_layers[0].thickness
    cell = sim.cell_size

    source = meep.Source(src_time,
                         component=meep.Ex,
                         center=meep.Vector3(0,0,-cell[2]/2 + pml_thickness),
                         size=meep.Vector3(cell[0],cell[1],0),
                         amplitude=amplitude)

    sim.add_source(source)

def y_polarized_plane_wave(sim, src_time, amplitude=1.0):
    """add an y-polarized plane wave to sim with time profile and amplitude"""
    # assume first boundary layer is the full PML
    pml_thickness = sim.boundary_layers[0].thickness
    cell = sim.cell_size

    source = meep.Source(src_time,
                         component=meep.Hx,
                         center=meep.Vector3(0,0,-cell[2]/2 + pml_thickness),
                         size=meep.Vector3(cell[0],cell[1],0),
                         amplitude=-amplitude)

    sim.add_source(source)

def rhc_polarized_plane_wave(sim, src_time, amplitude=1.0):
    """add an rhc-polarized plane wave to sim with time profile and amplitude"""

    x_polarized_plane_wave(sim, src_time, amplitude=amplitude)
    y_polarized_plane_wave(sim, src_time, amplitude=1j*amplitude)

def lhc_polarized_plane_wave(sim, src_time, amplitude=1.0):
    """add an lhc-polarized plane wave to sim with time profile and amplitude"""

    x_polarized_plane_wave(sim, src_time, amplitude=amplitude)
    y_polarized_plane_wave(sim, src_time, amplitude=-1j*amplitude)

def gaussian_amp(pos, width):
    rsq = pos.x**2 + pos.y**2
    return math.exp(-rsq/width**2)

def gaussian_beam(sim, src_time, width, polarization, amplitude=1.0):
    # assume first boundary layer is the full PML
    pml_thickness = sim.boundary_layers[0].thickness
    cell = sim.cell_size

    def amp_func(pos):
        return gaussian_amp(pos, width)

    source = meep.Source(src_time,
                         component=polarization,
                         center=meep.Vector3(0,0,-cell[2]/2 + pml_thickness),
                         size=meep.Vector3(cell[0],cell[1],0),
                         amplitude=amplitude,
                         amp_func=amp_func)

    sim.add_source(source)

def azimuthal_beam(sim, src_time, width, amplitude=1.0):
    # assume first boundary layer is the full PML
    pml_thickness = sim.boundary_layers[0].thickness
    cell = sim.cell_size

    def amp_func_x(pos):
        return gaussian_amp(pos, width)*pos.y

    def amp_func_y(pos):
        return -gaussian_amp(pos, width)*pos.x

    source = meep.Source(src_time,
                         component=meep.Ex,
                         center=meep.Vector3(0,0,-cell[2]/2 + pml_thickness),
                         size=meep.Vector3(cell[0],cell[1],0),
                         amplitude=amplitude,
                         amp_func=amp_func_x)
    sim.add_source(source)

    source = meep.Source(src_time,
                         component=meep.Ey,
                         center=meep.Vector3(0,0,-cell[2]/2 + pml_thickness),
                         size=meep.Vector3(cell[0],cell[1],0),
                         amplitude=amplitude,
                         amp_func=amp_func_y)
    sim.add_source(source)

def radial_beam(sim, src_time, width, amplitude=1.0):
    # assume first boundary layer is the full PML
    pml_thickness = sim.boundary_layers[0].thickness
    cell = sim.cell_size

    def amp_func_x(pos):
        return gaussian_amp(pos, 500e-9)*pos.y

    def amp_func_y(pos):
        return -gaussian_amp(pos, 500e-9)*pos.x

    source = meep.Source(src_time,
                         component=meep.Hx,
                         center=meep.Vector3(0,0,-cell[2]/2 + pml_thickness),
                         size=meep.Vector3(cell[0],cell[1],0),
                         amplitude=amplitude,
                         amp_func=amp_func_x)
    sim.add_source(source)

    source = meep.Source(src_time,
                         component=meep.Hy,
                         center=meep.Vector3(0,0,-cell[2]/2 + pml_thickness),
                         size=meep.Vector3(cell[0],cell[1],0),
                         amplitude=amplitude,
                         amp_func=amp_func_y)
    sim.add_source(source)
