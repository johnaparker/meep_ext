import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
gold = meep.Medium(index=3.5)

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
fcen, df = meep_ext.freq_data(1/(400*nm), 1/(1000*nm))
nfreq = 40
src_time = meep.GaussianSource(frequency=1.3/um, fwidth=4.0/um)
polarization = meep.Ex   # used in convergence check 'decay_by'
source = lambda sim: meep_ext.x_polarized_plane_wave(sim, src_time)

### monitor info
particle_monitor_gap = 50*nm
pml_monitor_gap = 50*nm
norm_file_ext = 'norm'
monitor_size = [sep + 2*radius + 2*particle_monitor_gap, 2*radius + 2*particle_monitor_gap,
            2*radius + 2*particle_monitor_gap]

### grid
resolution = 1/(10*nm)
pml = meep.PML(100*nm)
medium = meep.Medium(index=1)
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
                        default_material=medium,
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
                        default_material=medium,
                        resolution=resolution)
    norm.init_fields()
    source(norm)

    
    flux_box_inc = meep_ext.add_flux_box(norm, fcen, df, nfreq, [0,0,0], monitor_size)
    flux_inc = meep_ext.add_flux_plane(norm, fcen, df, nfreq, [0,0,0], [2*radius, 2*radius, 0])

    norm.run(until_after_sources=meep.stop_when_fields_decayed(.5*um, polarization,
                      pt=meep.Vector3(0,0,monitor_size[2]/2), decay_by=1e-3))

    norm.save_flux(norm_file_ext, flux_box_inc)

    return {'frequency': np.array(meep.get_flux_freqs(flux_inc)), 'area': (2*radius)**2,
            'incident': np.asarray(meep.get_fluxes(flux_inc))}

@job.cache
def dimer_scat():
    """perform scattering simulation"""

    scat = meep.Simulation(cell_size=cell,
                        boundary_layers=[pml],
                        geometry=geometry,
                        default_material=medium,
                        resolution=resolution)
    scat.init_fields()
    source(scat)

    flux_box_absorb = meep_ext.add_flux_box(scat, fcen, df, nfreq, [0,0,0], monitor_size)
    flux_box_scat   = meep_ext.add_flux_box(scat, fcen, df, nfreq, [0,0,0], monitor_size)
    scat.load_minus_flux(norm_file_ext, flux_box_scat)

    # scat.run(until_after_sources=8*um)
    scat.run(until_after_sources=meep.stop_when_fields_decayed(.5*um, polarization,
                pt=p2-meep.Vector3(0,0,monitor_size[2]/2), decay_by=1e-4))

    return {'scattering': np.array(meep.get_fluxes(flux_box_scat)),'absorption': -np.array(meep.get_fluxes(flux_box_absorb)),
            'frequency': np.array(meep.get_flux_freqs(flux_box_scat))}

@job.at_end
def vis():
    ### forces
    fig, ax = plt.subplots()

    norm = job.load(dimer_norm)
    scat = job.load(dimer_scat)

    ax.plot((1/nm)/norm.frequency, scat.scattering/norm.incident*norm.area, 'o', color='C0', label='scattering (FDTD)')
    ax.plot((1/nm)/norm.frequency, scat.absorption/norm.incident*norm.area, 'o', color='C1', label='absorption (FDTD)')
    ax.plot((1/nm)/norm.frequency, (scat.scattering + scat.absorption)/norm.incident*norm.area, 
            'o', color='C2', label='extinction (FDTD)')

    import miepy
    wavelengths = np.linspace(400*nm, 1000*nm, 100)
    eps = meep_ext.get_eps(gold)(wavelengths)
    Au = miepy.data_material(wavelengths, eps)
    # Au = miepy.constant_material(3.5**2)

    spheres = miepy.spheres([[-sep/2,0,0],[sep/2,0,0]], radius, Au)
    source = miepy.sources.x_polarized_plane_wave()
    sol = miepy.gmt(spheres, source, wavelengths, 2)
    C,A,E = sol.cross_sections()

    ax.axhline(0, linestyle='--', color='black')
    ax.plot(wavelengths/nm, C, color='C0', label='scattering (GMT)')
    ax.plot(wavelengths/nm, A, color='C1', label='absorption (GMT)')
    ax.plot(wavelengths/nm, E, color='C2', label='extinction (GMT)')

    ax.legend()
    ax.set(xlabel='wavelength (nm)', ylabel='cross-section')

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
