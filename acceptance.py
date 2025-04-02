import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os

def main():
    processes = ['prima']
    prev_limits_lower = {'Process': processes}
    prev_limits_lower[10] = (0.00000134, 1)
    prev_limits_lower[20] = (7.9e-7,0.0010462)
    prev_limits_lower[30] = (5.7e-7,0.00050747)
    prev_limits_lower[40] = (4.6e-7,0.00027886)
    prev_limits_lower[50] = (3.8e-7, 0.00014578) #(0.00479207,1)
    prev_limits_lower[60] = (3.7e-7,0.0001028)
    prev_limits_lower[70] = (3e-7,0.00007813)
    prev_limits_lower[80] = (2.8e-7,0.00005113)
    prev_limits_lower[90] = (2.6e-7,0.00004)
    prev_limits_lower[100] = (2.3e-7, 0.00002673)#(0.00491312,1)
    prev_limits_lower[110] = (2.3e-7,0.000022)
    prev_limits_lower[120] = (2.2e-7,0.00000466)
    prev_limits_lower[200] = (1.7e-7, 0.00000466)#(0.00115599,1)
    prev_limits_lower[300] = (1.5e-7, 0.00000127)#()
    prev_limits_lower[400] = (1.4e-7, 6.5e-7)#()
    prev_limits_lower[500] = (0.00148354,1)
    prev_limits_upper = {'Process': processes}
    prev_limits_upper[10] = (1, 10)
    prev_limits_upper[20] = (0.00155944,1)
    prev_limits_upper[30] = (0.00516448,1)
    prev_limits_upper[40] = (0.00491312,1)
    prev_limits_upper[50] = (0.00479207,1)
    prev_limits_upper[60] = (0.00479207,1)
    prev_limits_upper[70] = (0.00479207,1)
    prev_limits_upper[80] = (0.00479207,1)
    prev_limits_upper[90] = (0.00479207,1)
    prev_limits_upper[100] = (0.00491312,1)
    prev_limits_upper[110] = (0.00092351,1)
    prev_limits_upper[120] = (0.00172308,1)
    prev_limits_upper[200] = (0.00115599,1)
    prev_limits_upper[300] = (0.00071961,1)
    prev_limits_upper[400] = (0.00081521,1)
    prev_limits_upper[500] = (1,10)#0.00073779,1)
    optimal_couplings = {'Process': processes}
    optimal_couplings[20] = (3e-5,4e-5,5e-5,7e-5,1e-4,2e-4,3e-4,4e-4,5e-4,7e-4,9e-4,1e-3,1.2e-3,1.5e-3,1.7e-3,2e-3,2.5e-3,3e-3,3.5e-3,4e-3,4.2e-3,4.5e-3,5e-3)
    optimal_couplings[30] = (1e-6,2e-5,5e-5,7e-5,1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,1e-3,1.5e-3,2e-3,2.5e-3,3e-3)
    optimal_couplings[40] = (1e-5,2e-5,5e-5,7e-5,1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,1e-3,5e-3)
    optimal_couplings[50] = (9e-6,1e-5,2e-5,3e-5,4e-5,5e-5,7e-5,1e-4,1.2e-4,1.5e-4,1.7e-4,2.0e-4,2.5e-4,3e-4,3.5e-4,4e-4,4.5e-4,5.5e-4,6e-4,6.5e-4,7.5e-4,8e-4)
    optimal_couplings[60] = (1e-5,2e-5,5e-5,7e-5,1e-4,1.2e-4,1.5e-4,1.7e-4,2.0e-4,2.5e-4,3e-4,3.5e-4,4e-4)
    optimal_couplings[70] = (1e-5,2e-5,3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,1e-4,1.2e-4,1.5e-4,1.7e-4,2.0e-4,2.6e-4)
    optimal_couplings[80] = (9e-6,1e-5,2e-5,3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,1e-4,1.2e-4,1.5e-4,1.7e-4,2.0e-4)
    optimal_couplings[90] = (1e-5,2e-5,3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,1e-4,1.1e-4,1.2e-4,1.4e-4)
    optimal_couplings[100] = (1e-5,2e-5,3e-5,3.5e-5,4e-5,4.5e-5,5e-5,5.5e-5,6e-5,7e-5,8e-5,9e-5,1e-4)
    optimal_couplings[110] = (1e-5,1.5e-5,2e-5,2.5e-5,3e-5,3.5e-5,4e-5,4.5e-5,5e-5,5.5e-5,6e-5,7e-5,8e-5)
    optimal_couplings[120] = (1e-5,1.5e-5,2e-5,2.5e-5,3e-5,3.5e-5,4e-5,4.5e-5,5e-5,5.5e-5,6e-5)
    optimal_couplings[200] = (1e-5,5e-5,6e-5,7e-5,8e-5,9e-5,1e-3)
    optimal_couplings[300] = (1e-5,5e-6,1e-5,5e-5,1e-4,5e-4,7e-4)
    optimal_couplings[400] = (2e-5,5e-5,1e-4,5e-4,7e-4)
    optimal_couplings[500] = (5e-5,2e-3)

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

    masses =[20,30,40,50,60,70,80,90,100,110,120]

    all_fracs = []
    all_coups = []
    all_masses = []
    for i, mass in enumerate(masses):
        file_path = "/Users/sophie/LDMX/software/NewClone/ALPs/displaced/m"+str(mass)+"_prima.lhe"
        count_z_less_6000 = 0
        count_z_gt_800 = 0
        valid_z = []
        with open(file_path, 'r') as lhe_file:
            for line in lhe_file:
                if line.startswith("#vertex"):  # Look for vertex lines
                    parts = line.split()  # Split line into parts
                    z_component = float(parts[3])  # Assuming z-component is the 5th
                    if  z_component > 0:
                        count_z_less_6000 += 1
                        valid_z.append(z_component)
        valid_z = np.array(valid_z)
        print("Number of z-components less than 6000:", count_z_less_6000)
        processes = ['prima', 'pf']
        fracs = []
        coups = []
        masses_i = []
        for i, coup in enumerate(optimal_couplings[mass]):
        #coup =  5e-6
            factor = (1e-3 / coup)**2
            xsec_factor = (coup / 1e-3)**2
            z_optimal = (valid_z * factor)
            z_in_hcal = 0
            z_all = 0
            for i, z in enumerate(z_optimal):
                z_all +=1
                if z > 750 and z < 5700:
                    z_in_hcal += 1
            print(N_prod_proc[mass][0]*xsec_factor*z_in_hcal/z_all)
            fracs.append(N_prod_proc[mass][0]*xsec_factor*z_in_hcal/z_all)
            #masses_i.append(mass)
            #coups.append(coup)
            all_fracs.append(N_prod_proc[mass][0]*xsec_factor*z_in_hcal/z_all)
            all_coups.append(coup)
            all_masses.append(mass)
        fig, ax1 = plt.subplots()
        ax1.axvline(x=prev_limits_lower[mass][1], linestyle='--', color='black', label='Previous limit lower')
        ax1.axvline(x=prev_limits_upper[mass][0], linestyle='--', color='black', label='Previous limit upper')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$g_{a \gamma} [ GeV^{-1}]$')
        plt.ylabel('# Events in vicinity of back HCal (1e16 EOT)')

        # Plot the first graph with ax1
        line1, = ax1.plot(optimal_couplings[mass], fracs, color='blue', label='m_{ALP} ='+str(mass)+"MeV")

        plt.title('$m_{ALP}$ ='+str(mass)+"MeV")
        plt.savefig("acceptance"+str(mass)+".pdf")
        plt.show()

if __name__ == '__main__':
    main()
