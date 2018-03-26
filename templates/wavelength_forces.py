import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import meep
import meep_ext
import pinboard

job = pinboard.pinboard()

### transformation optics
nb = 1.33
scale = nb
nm = 1e-9*scale
um = 1e-6*scale

### geometry
radius = 75*nm
gold = meep_ext.material.Au(multiplier=1/scale**2)
# gold = meep.Medium(index=3.5/scale)

sep = 400*nm
p1 = meep.Vector3(-sep/2, 0, 0)
p2 = meep.Vector3(sep/2, 0, 0)
geometry = [meep.Sphere(center=p1,
                     radius=radius, 
                     material=gold),
            meep.Sphere(center=p2,
                     radius=radius, 
                     material=gold)]

### source
fcen, df = meep_ext.freq_data(scale/(400*nm), scale/(1000*nm))
nfreq = 40
src_time = meep.GaussianSource(frequency=1.3*scale/um, fwidth=4.0*scale/um)
polarization = meep.Ex   # used in convergence check 'decay_by'
source = lambda sim: meep_ext.rhc_polarized_plane_wave(sim, src_time)

### monitor info
particle_monitor_gap = 50*nm
pml_monitor_gap = 50*nm

### grid
resolution = 1/(7.5*nm)
pml = meep.PML(100*nm)
lx = sep + 2*radius + 2*particle_monitor_gap + 2*pml_monitor_gap + 2*pml.thickness
ly = lz =  2*radius + 2*particle_monitor_gap + 2*pml_monitor_gap + 2*pml.thickness
cell = meep.Vector3(lx,ly,lz)
Nx, Ny, Nz = map(round, cell*resolution)


@job.cache
def dimer_fields():
    """capture the Ex fields in the xz plane"""
    sim = meep.Simulation(cell_size=cell,
                        boundary_layers=[pml],
                        geometry=geometry,
                        resolution=resolution)
    sim.init_fields()
    source(sim)

    while sim.fields.time() < 3*um:
        sim.fields.step()

        if sim.fields.t % 10 == 0:
            yield {'E': sim.get_array(meep.Vector3(0,0,0), meep.Vector3(cell[0], 0, cell[2]), polarization)}

@job.cache
def dimer_norm():
    """perform normalization simulation"""
    norm = meep.Simulation(cell_size=cell,
                        boundary_layers=[pml],
                        geometry=[],
                        resolution=resolution)
    norm.init_fields()
    source(norm)

    flux_inc = meep_ext.add_flux_plane(norm, fcen, df, nfreq, [0,0,0], [2*radius, 2*radius, 0])

    norm.run(until_after_sources=meep.stop_when_fields_decayed(.5*um, polarization,
                      pt=meep.Vector3(0,0,0), decay_by=1e-3))

    return {'frequency': np.array(meep.get_flux_freqs(flux_inc)), 'area': (2*radius)**2,
            'incident': np.asarray(meep.get_fluxes(flux_inc))}

@job.cache
def dimer_scat():
    """perform scattering simulation"""

    scat = meep.Simulation(cell_size=cell,
                        boundary_layers=[pml],
                        geometry=geometry,
                        resolution=resolution)
    scat.init_fields()
    source(scat)

    L = 2*radius + 2*particle_monitor_gap
    Fx = meep_ext.add_force_box(scat, fcen, df, nfreq, p2, [L,L,L], meep.X)
    Fy = meep_ext.add_force_box(scat, fcen, df, nfreq, p2, [L,L,L], meep.Y)
    Fz = meep_ext.add_force_box(scat, fcen, df, nfreq, p2, [L,L,L], meep.Z)

    # scat.run(until_after_sources=8*um)
    scat.run(until_after_sources=meep.stop_when_fields_decayed(.5*um, polarization,
                pt=p2-meep.Vector3(0,0,L/2), decay_by=1e-4))

    return {'Fx': np.array(meep.get_forces(Fx)), 'Fy': np.array(meep.get_forces(Fy)), 'Fz': np.array(meep.get_forces(Fz)),
            'frequency': np.array(meep.get_force_freqs(Fx))}

@job.at_end
def vis():
    nm = 1e-9

    ### forces
    fig, axes = plt.subplots(nrows=2, figsize=(7,6), sharex=True,
                  gridspec_kw=dict(height_ratios=[2,1], hspace=0.05))

    norm = job.load(dimer_norm)
    scat = job.load(dimer_scat)

    for ax in axes:
        ax.plot((1/nm)/norm.frequency, scat.Fx/norm.incident*norm.area*constants.epsilon_0/2*1e25, 'o', color='C0', label='Fx (FDTD)')
        ax.plot((1/nm)/norm.frequency, scat.Fy/norm.incident*norm.area*constants.epsilon_0/2*1e25, 'o', color='C1', label='Fy (FDTD)')
        ax.plot((1/nm)/norm.frequency, scat.Fz/norm.incident*norm.area*constants.epsilon_0/2*1e25, 'o', color='C2', label='Fz (FDTD)')

    import miepy
    wavelengths = np.linspace(400*nm, 1000*nm, 100)
    eps = meep_ext.get_eps(gold)(wavelengths)
    Au = miepy.data_material(wavelengths, eps*scale**2)
    water = miepy.constant_material(nb**2)
    # Au = miepy.constant_material(3.5**2)

    spheres = miepy.spheres([[-sep/2/scale,0,0],[sep/2/scale,0,0]], radius/scale, Au)
    source = miepy.sources.rhc_polarized_plane_wave()
    sol = miepy.gmt(spheres, source, wavelengths, 2, medium=water)
    F = sol.force_on_particle(1) 

    for ax in axes:
        ax.axhline(0, linestyle='--', color='black')
        ax.plot(wavelengths/nm, F[0]*1e25, color='C0', label='Fx (GMT)')
        ax.plot(wavelengths/nm, F[1]*1e25, color='C1', label='Fy (GMT)')
        ax.plot(wavelengths/nm, F[2]*1e25, color='C2', label='Fz (GMT)')

    axes[0].legend()
    axes[0].set(ylabel='force')
    axes[1].set(xlabel='wavelength (nm)', ylabel='force', ylim=[-0.035,0.01])

    ### field animation
    fig, ax = plt.subplots()

    x = np.linspace(0, cell[0]/nm, Nx)
    z = np.linspace(0, cell[1]/nm, Nz)
    X,Z = np.meshgrid(x,z, indexing='ij')

    var = job.load(dimer_fields)
    idx = np.s_[10:-10,10:-10]
    E = var.E[:,10:-10,10:-10]
    # E = var.E
    vmax = np.max(np.abs(E))/2
    im = ax.pcolormesh(X[idx], Z[idx], E[0], cmap='RdBu', animated=True, vmax=vmax, vmin=-vmax)

    ax.set_aspect('equal')
    def update(i):
        im.set_array(np.ravel(E[i][:-1,:-1]))
        return [im]

    ani = animation.FuncAnimation(fig, update, range(E.shape[0]), interval=50, blit=True)

    plt.show()

job.execute()
