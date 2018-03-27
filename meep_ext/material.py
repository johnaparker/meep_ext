import numpy as np
import meep

eV_um_scale = 1/1.23984193*1e6

def drude_lorentz_material(freq, gamma, sigma, eps_inf=1, multiplier=1):
    """return a drude-lorentz material, where the first index is the Drude term"""

    freq, gamma, sigma = map(np.atleast_1d, [freq, gamma, sigma])

    Npoles = len(freq)
    susc = []

    for i in range(Npoles):
        func = meep.DrudeSusceptibility if i == 0 else meep.LorentzianSusceptibility
        susc.append(func(frequency=freq[i], gamma=gamma[i], sigma=sigma[i]*multiplier))

    material = meep.Medium(epsilon=eps_inf*multiplier, E_susceptibilities=susc)
    return material

def lorentz_material(freq, gamma, sigma, eps_inf=1, multiplier=1):
    """return a lorentz material"""

    freq, gamma, sigma = map(np.atleast_1d, [freq, gamma, sigma])

    Npoles = len(freq)
    func = meep.LorentzianSusceptibility
    susc = [func(frequency=freq[i], gamma=gamma[i], sigma=sigma[i]*multiplier) for i in range(Npoles)]

    material = meep.Medium(epsilon=eps_inf*multiplier, E_susceptibilities=susc)
    return material

def single_freq_material(eps, freq, multiplier=1):
    """fit a material model to complex permitivitty at a single given frequency (1/wavelength)"""

    # with positive eps, use simple material
    if eps.real > 0:
        return meep.Medium(epsilon=eps.real*multiplier, D_conductivity=2*np.pi*freq*eps.imag/eps.real*multiplier*1e18)
    # with negative eps, use Lorentz material
    else:
        eps_inf = 1
        sigma = 1
        gamma = freq*eps.imag/(eps.imag**2 + (eps.real-2)*(eps.real-1))
        fn_sq = 1/(2-eps.real)*(freq*gamma*eps.imag - freq**2*(eps.real-1))
        fn = fn_sq**0.5

        return lorentz_material(fn, gamma, sigma, eps_inf=eps_inf, multiplier=multiplier)

def fit_drude_lorentz(eps, freq):
    """fit a drude-lorentz material model to complex permitivitty"""
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

        factor = 1 + 1j*material.D_conductivity_diag[0]*wavelength/(2*np.pi)
        return eps_val*factor


    return eps

def Au(multiplier=1):
    """Gold material"""
    wp  = 9.01*eV_um_scale
    f   = eV_um_scale*np.array([1e-20, 4.25692])
    gam = eV_um_scale*np.array([0.0196841, 4.15975])
    sig = wp**2/f**2*np.array([0.970928, 1.2306])
    eps_inf = 3.63869

    return drude_lorentz_material(f, gam, sig, eps_inf, multiplier=multiplier)
