import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os

def main():
    processes = ['prima']

    N_prod_proc = {'Process': processes}
    N_prod_proc[10] = [134000]  #this will change based on coupling
    N_prod_proc[20] = [103444]
    N_prod_proc[30] = [84590]
    N_prod_proc[40] = [70960]
    N_prod_proc[50] = [62700]
    N_prod_proc[60] = [52382]
    N_prod_proc[70] = [45870]
    N_prod_proc[80] = [40590]
    N_prod_proc[90] = [36168]
    N_prod_proc[100] = [33700]
    N_prod_proc[110] = [29260]
    N_prod_proc[120] = [26532]
    N_prod_proc[150] = [21000]
    N_prod_proc[200] = [14000]
    N_prod_proc[300] = [6800]
    N_prod_proc[400] = [3600]
    N_prod_proc[500] = [2000]

    masses =[10,20,30,40,50,60,70,80,90,100,110,120,150,200,300,500]

    X = np.linspace(20,120, 11)
    Y = np.linspace(1e-4,1e-3,18)

    all_fracs = []
    all_coups = []
    all_masses = []
    fig2, ax2 = plt.subplots()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$g_{a \gamma} [ GeV^{-1}]$')
    plt.ylabel('# Events in vicinity of back HCal (1e16 EOT)')
    fracs = []
    coups = []
    masses_i = []

    start_coupling = 1e-5
    end_coupling = 1e-2

    c_lows = []
    c_his =[]
    for i, mass in enumerate(masses):
        c_low = 1000
        c_hi = 0
        file_path = "/Users/sophie/LDMX/software/NewClone/ALPs/displaced/m"+str(mass)+"_prima.lhe"
        valid_z = []
        with open(file_path, 'r') as lhe_file:
            for line in lhe_file:
                if line.startswith("#vertex"):  # Look for vertex lines
                    parts = line.split()  # Split line into parts
                    z_component = float(parts[3])  # Assuming z-component is the 5th
                    if  z_component > 0:
                        valid_z.append(z_component)
        valid_z = np.array(valid_z)
        coup = start_coupling
        while coup < end_coupling:

            factor = (1e-3 / coup)**2
            xsec_factor = (coup / 1e-3)**2
            z_optimal = (valid_z * factor)
            z_in_hcal = 0
            z_all = 0
            for i, z in enumerate(z_optimal):
                z_all +=1
                if z > 750 and z < 5700:
                    z_in_hcal += 1
            #print(mass,coup,N_prod_proc[mass][0]*xsec_factor*z_in_hcal/z_all)

            if(N_prod_proc[mass][0]*xsec_factor*z_in_hcal/z_all)> 2.44:#(0.88/22)
                fracs.append(N_prod_proc[mass][0]*xsec_factor*z_in_hcal/z_all)
                if coup < c_low :
                    c_low = coup

                if coup > c_hi:
                    c_hi = coup

                #coups.append(coup)
                #masses_i.append(mass)
            coup +=5e-7
        if(c_low!=1000 and c_hi !=0):
            c_lows.append(c_low)
            masses_i.append(mass)
            c_his.append(c_hi)
            print(mass/1000,c_low)
            print(mass/1000,c_hi)
    #ax2.scatter(masses_i, coups,s=500, color='green', marker='o', alpha=0.6, )
    #print(masses_i, c_lows, c_his)
    plt.fill_between(masses_i, c_lows, c_his, color='skyblue',alpha=0.6)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$g_{a \gamma} [ GeV^{-1}]$')
    plt.xlabel('mass [MeV]')
    plt.xlim(10, 500)
    plt.ylim(1e-7, 1e-2)
    plt.savefig("sens.pdf")
    plt.show()


if __name__ == '__main__':
    main()
