import meep

def freq_data(fmin, fmax):
    """get center frequency and df for given frequency range"""
    fcen = (fmin + fmax)/2
    df = abs(fmax - fmin)
    return fcen, df

def force_box(position, size, component):
    """create 6 force planes to calculate the component of a force"""
    position = meep.Vector3(*position)
    size = meep.Vector3(*size)

    force_x1 = meep.ForceRegion(center=position - meep.Vector3(size[0]/2, 0, 0),
                            size=meep.Vector3(0, size[1], size[2]), weight=-1,
                            direction=component)

    force_x2 = meep.ForceRegion(center=position + meep.Vector3(size[0]/2, 0, 0),
                            size=meep.Vector3(0, size[1], size[2]), weight=1,
                            direction=component)

    force_y1 = meep.ForceRegion(center=position - meep.Vector3(0, size[1]/2, 0),
                            size=meep.Vector3(size[0], 0, size[2]), weight=-1,
                            direction=component)

    force_y2 = meep.ForceRegion(center=position + meep.Vector3(0, size[1]/2, 0),
                            size=meep.Vector3(size[0], 0, size[2]), weight=1,
                            direction=component)

    force_z1 = meep.ForceRegion(center=position - meep.Vector3(0, 0, size[2]/2),
                            size=meep.Vector3(size[0], size[1], 0), weight=-1,
                            direction=component)

    force_z2 = meep.ForceRegion(center=position + meep.Vector3(0, 0, size[2]/2),
                            size=meep.Vector3(size[0], size[1], 0), weight=1,
                            direction=component)
    
    return [force_x1, force_x2, force_y1, force_y2, force_z1, force_z2]

def add_force_box(sim, fcen, df, nfreq, position, size, component):
    """add a force box to a meep simulation"""
    box = force_box(position, size, component)
    return sim.add_force(fcen, df, nfreq, *box)

def flux_box(position, size):
    """create 6 flux planes of a closed box with size and position"""
    position = meep.Vector3(*position)
    size = meep.Vector3(*size)

    flux_x1 = meep.FluxRegion(center=position - meep.Vector3(size[0]/2, 0, 0),
                            size=meep.Vector3(0, size[1], size[2]), weight=-1)

    flux_x2 = meep.FluxRegion(center=position + meep.Vector3(size[0]/2, 0, 0),
                            size=meep.Vector3(0, size[1], size[2]), weight=1)

    flux_y1 = meep.FluxRegion(center=position - meep.Vector3(0, size[1]/2, 0),
                            size=meep.Vector3(size[0], 0, size[2]), weight=-1)

    flux_y2 = meep.FluxRegion(center=position + meep.Vector3(0, size[1]/2, 0),
                            size=meep.Vector3(size[0], 0, size[2]), weight=1)

    flux_z1 = meep.FluxRegion(center=position - meep.Vector3(0, 0, size[2]/2),
                            size=meep.Vector3(size[0], size[1], 0), weight=-1)

    flux_z2 = meep.FluxRegion(center=position + meep.Vector3(0, 0, size[2]/2),
                            size=meep.Vector3(size[0], size[1], 0), weight=1)

    return [flux_x1, flux_x2, flux_y1, flux_y2, flux_z1, flux_z2]

def add_flux_plane(sim, fcen, df, nfreq, position, size):
    """add a flux plane to a meep simulation"""
    position = meep.Vector3(*position)
    size = meep.Vector3(*size)

    plane = meep.FluxRegion(center=position, size=size)
    return sim.add_flux(fcen, df, nfreq, plane)

def add_flux_box(sim, fcen, df, nfreq, position, size):
    """add a flux box to a meep simulation"""
    box = flux_box(position, size)
    return sim.add_flux(fcen, df, nfreq, *box)

### 2D simulations

def force_box_2d(position, size, component):
    """create 4 force planes (2D) to calculate the component of a force"""
    force_x1 = meep.ForceRegion(center=position - meep.Vector3(0,size[1]/2),
                            size=meep.Vector3(size[0], 0), 
                            direction=component, weight=-1)
    force_x2 = meep.ForceRegion(center=position + meep.Vector3(0,size[1]/2),
                            size=meep.Vector3(size[0], 0), 
                            direction=component, weight=1)

    force_y1 = meep.ForceRegion(center=position - meep.Vector3(size[0]/2, 0),
                            size=meep.Vector3(0, size[1]), 
                            direction=component, weight=-1)
    force_y2 = meep.ForceRegion(center=position + meep.Vector3(size[0]/2, 0),
                            size=meep.Vector3(0, size[1]), 
                            direction=component, weight=1)
    
    return [force_x1, force_x2, force_y1, force_y2]

def add_force_box_2d(sim, fcen, df, nfreq, position, size, component):
    """add a force box to a meep simulation"""
    box = force_box_2d(position, size, component)
    return sim.add_force(fcen, df, nfreq, *box)
    return sim.add_flux(fcen, df, nfreq, *box)

def flux_box_2d(position, size):
    """create 4 flux planes (2D) of a closed box with size and position"""
    position = meep.Vector3(*position)
    size = meep.Vector3(*size)

    flux_x1 = meep.FluxRegion(center=position - meep.Vector3(size[0]/2, 0),
                            size=meep.Vector3(0, size[1]), weight=-1)

    flux_x2 = meep.FluxRegion(center=position + meep.Vector3(size[0]/2, 0),
                            size=meep.Vector3(0, size[1]), weight=1)

    flux_y1 = meep.FluxRegion(center=position - meep.Vector3(0, size[1]/2),
                            size=meep.Vector3(size[0], 0), weight=-1)

    flux_y2 = meep.FluxRegion(center=position + meep.Vector3(0, size[1]/2),
                            size=meep.Vector3(size[0], 0), weight=1)

    return [flux_x1, flux_x2, flux_y1, flux_y2]

def add_flux_box_2d(sim, fcen, df, nfreq, position, size):
    """add a flux box to a meep simulation"""
    box = flux_box_2d(position, size)
    return sim.add_flux(fcen, df, nfreq, *box)


def near2far(position, size):
    """create 6 near2far planes of a closed box with size and position"""
    position = meep.Vector3(*position)
    size = meep.Vector3(*size)

    x1 = meep.Near2FarRegion(center=position - meep.Vector3(size[0]/2, 0, 0),
                       size=meep.Vector3(0, size[1], size[2]), weight=-1)

    x2 = meep.Near2FarRegion(center=position + meep.Vector3(size[0]/2, 0, 0),
                       size=meep.Vector3(0, size[1], size[2]), weight=1)

    y1 = meep.Near2FarRegion(center=position - meep.Vector3(0, size[1]/2, 0),
                       size=meep.Vector3(size[0], 0, size[2]), weight=-1)

    y2 = meep.Near2FarRegion(center=position + meep.Vector3(0, size[1]/2, 0),
                       size=meep.Vector3(size[0], 0, size[2]), weight=1)

    z1 = meep.Near2FarRegion(center=position - meep.Vector3(0, 0, size[2]/2),
                       size=meep.Vector3(size[0], size[1], 0), weight=-1)

    z2 = meep.Near2FarRegion(center=position + meep.Vector3(0, 0, size[2]/2),
                            size=meep.Vector3(size[0], size[1], 0), weight=1)

    return [x1, x2, y1, y2, z1, z2]

def add_near2far(sim, fcen, df, nfreq, position, size):
    """add a flux box to a meep simulation"""
    box = near2far(position, size)
    return sim.add_flux(fcen, df, nfreq, *box)
