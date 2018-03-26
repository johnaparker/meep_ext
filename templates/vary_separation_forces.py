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

### grid
resolution = 1/(10*nm)
pml = meep.PML(100*nm)

@job.cache
def norm_sim():
    """perform normalization simulation"""
    L = 2*radius + 2*pml_monitor_gap + 2*particle_monitor_gap + 2*pml.thickness
    cell = meep.Vector3(L,L,L)
    norm = meep.Simulation(cell_size=cell,
                        boundary_layers=[pml],
                        geometry=[],
                        resolution=resolution)
    norm.init_fields()
    source(norm)

    flux_inc = meep_ext.add_flux_plane(norm, fcen, 0, 1, [0,0,0], [2*radius, 2*radius, 0])

    norm.run(until_after_sources=meep.stop_when_fields_decayed(.5*um, meep.Ex,
                      pt=meep.Vector3(0,0,0), decay_by=1e-3))

    return {'area': (2*radius)**2, 'norm': np.asarray(meep.get_fluxes(flux_inc))}

@job.cache
def sim(separation):
    """perform scattering simulation"""
    L = separation + 2*radius + 2*pml_monitor_gap + 2*particle_monitor_gap + 2*pml.thickness
    cell = meep.Vector3(L,L,L)

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

    L = 2*radius + 2*particle_monitor_gap
    Fx = meep_ext.add_force_box(scat, fcen, 0, 1, p2, [L,L,L], meep.X)
    Fy = meep_ext.add_force_box(scat, fcen, 0, 1, p2, [L,L,L], meep.Y)
    Fz = meep_ext.add_force_box(scat, fcen, 0, 1, p2, [L,L,L], meep.Z)

    # scat.run(until_after_sources=8*um)
    scat.run(until_after_sources=meep.stop_when_fields_decayed(.5*um, meep.Ex,
                pt=p2-meep.Vector3(0,0,L/2), decay_by=1e-5))

    return {'Fx': np.array(meep.get_forces(Fx))[0], 'Fy': np.array(meep.get_forces(Fy))[0], 'Fz': np.array(meep.get_forces(Fz))[0]}

@job.at_end
def vis():
    ### forces
    fig, axes = plt.subplots(nrows=2, figsize=(7,6), sharex=True,
                  gridspec_kw=dict(height_ratios=[2,1], hspace=0.05))

    force = np.zeros([3,len(separations)])
    for i,separation in enumerate(separations):
        var = job.load(sim, f'p{i}')
        force[0,i] = var.Fx
        force[1,i] = var.Fy
        force[2,i] = var.Fz

    norm = job.load(norm_sim)

    for ax in axes:
        ax.axhline(0, linestyle='--', color='black')
        ax.plot(separations/nm, force[0]/norm.norm*norm.area*constants.epsilon_0/2*1e25, 'o', color='C0', label='Fx (FDTD)')
        ax.plot(separations/nm, force[1]/norm.norm*norm.area*constants.epsilon_0/2*1e25, 'o', color='C1', label='Fy (FDTD)')
        ax.plot(separations/nm, force[2]/norm.norm*norm.area*constants.epsilon_0/2*1e25, 'o', color='C2', label='Fz (FDTD)')

    import miepy
    eps = meep_ext.get_eps(gold)(wavelength)
    Au = miepy.constant_material(eps)
    # Au = miepy.constant_material(3.5**2)
    source = miepy.sources.rhc_polarized_plane_wave()
    seps = np.linspace(300*nm, 900*nm, 100)

    force = np.zeros([3,len(seps)])
    for i,sep in enumerate(seps):
        spheres = miepy.spheres([[-sep/2,0,0],[sep/2,0,0]], radius, Au)
        sol = miepy.gmt(spheres, source, wavelength, 2)
        F = sol.force_on_particle(1) 
        force[:,i] = F.squeeze()

    for ax in axes:
        ax.plot(seps/nm, force[0]*1e25, color='C0', label='Fx (GMT)')
        ax.plot(seps/nm, force[1]*1e25, color='C1', label='Fy (GMT)')
        ax.plot(seps/nm, force[2]*1e25, color='C2', label='Fz (GMT)')

    axes[0].legend()
    axes[0].set(ylabel='force')
    axes[1].set(xlabel='separation (nm)', ylabel='force', ylim=[-3e-2, 3e-2])

    plt.show()

separations = np.linspace(300*nm, 900*nm, 10)
for i,separation in enumerate(separations):
    job.add_instance(sim, f'p{i}', separation=separation)

job.execute()
