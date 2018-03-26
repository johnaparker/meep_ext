import numpy as np
import meep

eV_um_scale = 1/1.23984193*1e6

def drude_lorentz_material(freq, gamma, sigma, eps_inf=1, multiplier=1):
    """return a drude-lorentz material, where the first index is the Drude term"""
    Npoles = len(freq)
    susc = []

    for i in range(Npoles):
        func = meep.DrudeSusceptibility if i == 0 else meep.LorentzianSusceptibility
        susc.append(func(frequency=freq[i], gamma=gamma[i], sigma=sigma[i]*multiplier))

    material = meep.Medium(epsilon=eps_inf*multiplier, E_susceptibilities=susc)
    return material

def lorentz_material(freq, gamma, sig,a, eps_inf=1, multiplier=1):
    """return a lorentz material"""

    freq, gamma, sigma = map(np.asarray, [freq, gamma, sigma])

    Npoles = len(freq)
    func = meep.LorentzianSusceptibility
    susc = [func(frequency=freq[i], gamma=gamma[i], sigma=sigma[i]*multiplier) for i in range(Npoles)]

    material = meep.Medium(epsilon=eps_inf*multiplier, E_susceptibilities=susc)
    return material

def fit_drude_lorentz(eps, wavelength):
    """fit a drude-lorentz material model to complex permitivitty"""
    pass

def fit_lorentz_at_single_wavelength(eps, wavelength):
    """fit a lorentz material model to complex permitivitty at a fixed wavelength"""
    pass

def get_eps(material):
    """obtain the complex permitivitty eps(wavelength) function of a material"""

    # assume isotropic material
    def eps(wavelength):
        omega = 1/wavelength
        eps_val = material.epsilon_diag[0]

        for pole in material.E_susceptibilities:
            freq = pole.frequency
            gamma = pole.gamma
            sigma = pole.sigma_diag[0]
            if isinstance(pole, meep.geom.DrudeSusceptibility):
                eps_val += 1j*sigma*freq**2/(omega*(gamma - 1j*omega))
            elif isinstance(pole, meep.geom.LorentzianSusceptibility):
                eps_val += sigma*freq**2/(freq**2 - omega**2 - 1j*omega*gamma)

        return eps_val

    return eps

def Au(multiplier=1):
    """Gold material"""
    wp  = 9.01*eV_um_scale
    f   = eV_um_scale*np.array([1e-20, 4.25692])
    gam = eV_um_scale*np.array([0.0196841, 4.15975])
    sig = wp**2/f**2*np.array([0.970928, 1.2306])
    eps_inf = 3.63869

    return drude_lorentz_material(f, gam, sig, eps_inf, multiplier=multiplier)
