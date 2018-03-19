"""
Compare material to raw eps data
"""

import numpy as np
import meep_ext
import matplotlib.pyplot as plt
import miepy

nm = 1e-3
wavelengths = np.linspace(400*nm, 1000*nm, 100)
eps = meep_ext.get_eps(meep_ext.material.Au())(wavelengths)

plt.plot(wavelengths*1e3, eps.real, color='C0', label='Re (meep)')
plt.plot(wavelengths*1e3, eps.imag, color='C1', label='Im (meep)')

gold = miepy.materials.predefined.Au()
eps = gold.eps(wavelengths*1e-6)

plt.plot(wavelengths*1e3, eps.real, '--', color='C0', label='Re (JC)')
plt.plot(wavelengths*1e3, eps.imag, '--', color='C1', label='Im (JC)')

plt.legend()
plt.show()
