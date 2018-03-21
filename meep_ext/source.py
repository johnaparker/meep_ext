import meep

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
