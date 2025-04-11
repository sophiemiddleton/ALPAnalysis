import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fig,ax = plt.subplots(1,1)

#plt.rcParams['text.usetex'] = True
#energy = [10,50,100,150,200,300,400,500]
#prima_decay = [39.67,0.317,0.0397,0.01176,0.00496,0.00147,0.0006199,0.000317]
coupling_10 = [1.00E-05, 2.50E-05, 5.00E-05, 7.50E-05,1.00E-04,2.50E-04,5.00E-04,7.50E-04,1.00E-03,2.50E-03,5.00E-03,7.50E-03,1.00E-02]
prima_decay_10 = [3.97E+05,6.35E+04,1.59E+04,7.05E+03,3.97E+03,6.35E+02,1.59E+02,7.05E+01,3.97E+01,6.35E+00,1.59E+00,7.05E-01,3.97E-01]
coupling_200 = [2.50E-06,5.00E-06,7.50E-06, 1.00E-05, 2.50E-05, 5.00E-05, 7.50E-05,1.00E-04,2.50E-04,5.00E-04,7.50E-04,1.00E-03,2.50E-03,5.00E-03,7.50E-03,1.00E-02]
prima_decay_200 = [7.94E+02,1.98E+02,8.82E+01,4.96E+01,7.94E+00,1.98E+00,8.82E-01,4.96E-01,7.94E-02,1.98E-02,8.82E-03,4.96E-03,7.94E-04,1.98E-04,8.82E-05,4.96E-05]
coupling_500 = [1.00E-07,2.50E-07,5.00E-07,7.50E-07,1.00E-06,2.50E-06,5.00E-06,7.50E-06,1.00E-05,2.50E-05,5.00E-05,7.50E-05,1.00E-04,2.50E-04,5.00E-04,7.50E-04,1.00E-03,2.50E-03,5.00E-03,7.50E-03,1.00E-02]
prima_decay_500 = [3.17E+04,5.07E+03,1.27E+03,5.64E+02,3.17E+02,5.07E+01,1.27E+01,5.64E+00,3.17E+00,5.07E-01,1.27E-01,5.64E-02,3.17E-02,5.07E-03,1.27E-03,5.64E-04,3.17E-04,5.07E-05,1.27E-05,5.64E-06,3.17E-06]
#PF_xsec = [428,261,145,85,52]
"""
plt.plot(energy, PF_xsec ,'o--', label = "Photon Fusion")
plt.legend(fontsize=8)
plt.ylabel(r'decay length [mm]')
plt.xlabel(r'$m_{ALP}$ [MeV/$c^{2}$]')
plt.legend()
plt.xlim([0,210])
plt.savefig("PFXSec.pdf")
plt.show()
"""
plt.plot(coupling_200, prima_decay_200 ,'o--', label = "Primakoff ($m_{ALP} = 200 MeV$)")
plt.legend(fontsize=8)
plt.ylabel(r'decay length [mm]')
#plt.xlabel(r'$m_{ALP}$ [MeV/$c^{2}$]')
plt.xlabel(r'$g_{a\gamma}$ [$GeV^{-1}$]')
plt.yscale('log')
plt.xscale('log')
plt.legend()
#plt.xlim([0,550])
plt.savefig("PKL.pdf")
plt.show()
