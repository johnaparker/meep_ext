import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import matplotlib as mpl
import meep
import meep_ext
import pinboard

job = pinboard.pinboard()
nm = 1e-9
um = 1e-6

### geometry
radius = 75*nm
gold = meep_ext.material.Au()
# gold = meep.Medium(index=3.5)

### source
wavelength = 550*nm
fcen = 1/wavelength
src_time = meep.GaussianSource(frequency=1.3/um, fwidth=4.0/um)
source = lambda sim: meep_ext.rhc_polarized_plane_wave(sim, src_time)

### monitor info
pml_monitor_gap = 50*nm
particle_monitor_gap = 50*nm
norm_file_ext = 'norm_{}'

### grid
resolution = 1/(10*nm)
pml = meep.PML(100*nm)

@job.cache
def norm_sim(monitor_size, unique_id):
    """perform normalization simulation with a given box size"""
    monitor_size = np.asarray(monitor_size)
    cell_size = monitor_size + 2*pml_monitor_gap + 2*pml.thickness
    cell = meep.Vector3(*cell_size)

    norm = meep.Simulation(cell_size=cell,
                        boundary_layers=[pml],
                        geometry=[],
                        resolution=resolution)
    norm.init_fields()
    source(norm)

    flux_inc = meep_ext.add_flux_plane(norm, fcen, 0, 1, [0,0,0], [2*radius, 2*radius, 0])
    flux_box_inc = meep_ext.add_flux_box(norm, fcen, 0, 1, [0,0,0], monitor_size)

    norm.run(until_after_sources=meep.stop_when_fields_decayed(.5*um, meep.Ex,
                      pt=meep.Vector3(0,0,monitor_size[2]/2), decay_by=1e-3))

    norm.save_flux(norm_file_ext.format(unique_id), flux_box_inc)

    return {'area': (2*radius)**2, 'norm': np.asarray(meep.get_fluxes(flux_inc))}

@job.cache
def sim(separation, monitor_size, unique_id):
    """perform scattering simulation"""
    monitor_size = np.asarray(monitor_size)
    cell_size = monitor_size + 2*pml_monitor_gap + 2*pml.thickness
    cell = meep.Vector3(*cell_size)

    p1 = meep.Vector3(-separation/2, 0, 0)
    p2 = meep.Vector3(separation/2, 0, 0)
    geometry = [meep.Sphere(center=p1,
                         radius=radius, 
                         material=gold),
                meep.Sphere(center=p2,
                         radius=radius, 
                         material=gold)]

    scat = meep.Simulation(cell_size=cell,
                        boundary_layers=[pml],
                        geometry=geometry,
                        resolution=resolution)
    scat.init_fields()
    source(scat)

    flux_box_absorb = meep_ext.add_flux_box(scat, fcen, 0, 1, [0,0,0], monitor_size)
    flux_box_scat   = meep_ext.add_flux_box(scat, fcen, 0, 1, [0,0,0], monitor_size)
    scat.load_minus_flux(norm_file_ext.format(unique_id), flux_box_scat)

    # scat.run(until_after_sources=8*um)
    scat.run(until_after_sources=meep.stop_when_fields_decayed(.5*um, meep.Ex,
                pt=p2 - meep.Vector3(0,0,monitor_size[2]/2), decay_by=1e-3))

    return {'scattering': np.array(meep.get_fluxes(flux_box_scat)), 'absorption': -np.array(meep.get_fluxes(flux_box_absorb))}

@job.at_end
def vis():
    ### cross-sections
    fig, ax = plt.subplots()

    scat = np.zeros([len(separations)])
    absorb = np.zeros([len(separations)])
    for i,separation in enumerate(separations):
        norm = job.load(norm_sim, f'p{i}')

        var = job.load(sim, f'p{i}')
        scat[i] = var.scattering/norm.norm*norm.area
        absorb[i] = var.absorption/norm.norm*norm.area


    ax.plot(separations/nm, scat, 'o', color='C0', label='scattering (FDTD)')
    ax.plot(separations/nm, absorb, 'o', color='C1', label='absorption (FDTD)')
    ax.plot(separations/nm, scat + absorb, 'o', color='C2', label='extinction (FDTD)')

    import miepy
    eps = meep_ext.get_eps(gold)(wavelength)
    Au = miepy.constant_material(eps)
    source = miepy.sources.rhc_polarized_plane_wave()
    seps = np.linspace(300*nm, 900*nm, 100)

    scat = np.zeros([len(seps)])
    absorb = np.zeros([len(seps)])
    extinct = np.zeros([len(seps)])
    for i,sep in enumerate(seps):
        spheres = miepy.spheres([[-sep/2,0,0],[sep/2,0,0]], radius, Au)
        sol = miepy.gmt(spheres, source, wavelength, 2)
        scat[i], absorb[i], extinct[i] = sol.cross_sections()

    ax.plot(seps/nm, scat, color='C0', label='scattering (GMT)')
    ax.plot(seps/nm, absorb, color='C1', label='absorption (GMT)')
    ax.plot(seps/nm, extinct, color='C2', label='extinction (GMT)')

    ax.set(xlabel='separation (nm)', ylabel='cross-section')
    ax.legend()

    plt.show()

separations = np.linspace(300*nm, 900*nm, 10)
for i,separation in enumerate(separations):
    monitor_size = [separation + 2*radius + particle_monitor_gap, 
            2*radius + particle_monitor_gap, 2*radius + particle_monitor_gap]
    job.add_instance(norm_sim, f'p{i}', monitor_size=monitor_size, unique_id=i)
    job.add_instance(sim, f'p{i}', separation=separation, monitor_size=monitor_size, unique_id=i)

job.execute()
