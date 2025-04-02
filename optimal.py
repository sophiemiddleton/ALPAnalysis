import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def main():
    processes = ['prima']
    prev_limits = {'Process': processes}
    prev_limits[50] = (1e-4, 5e-3)
    prev_limits[100] = (2e-5, 5e-3)
    prev_limits[200] = (3e-6, 1e-3)
    prev_limits[300] = (1e-6, 1e-3)
    prev_limits[400] = (5e-6, 1e-3)
    prev_limits[500] = (1e-7, 1e-3)
    N_prod_proc = {'Process': processes}
    N_prod_proc[10] = [134000]  #this will change based on coupling
    N_prod_proc[100] = [33700]
    N_prod_proc[150] = [21000]
    N_prod_proc[200] = [14000]
    N_prod_proc[300] = [6800]
    N_prod_proc[400] = [3600]
    N_prod_proc[500] = [2000]


    mass = 10
    file_path = "/Users/sophie/LDMX/software/NewClone/ALPs/displaced/m10_prima.lhe"
    count_z_less_6000 = 0
    valid_z = []
    with open("/Users/sophie/LDMX/software/NewClone/ALPs/displaced/m10_prima.lhe", 'r') as lhe_file:
        for line in lhe_file:
            if line.startswith("#vertex"):  # Look for vertex lines
                parts = line.split()  # Split line into parts
                z_component = float(parts[3])  # Assuming z-component is the 5th
                if  z_component > 0:
                    count_z_less_6000 += 1
                    valid_z.append(z_component)
    valid_z = np.array(valid_z)
    print("Number of z-components less than 600:", count_z_less_6000)
    processes = ['prima', 'pf']
    #transferring signal yield here
    sig_yield = {'Process': processes}
    sig_yield[10] = [3071, 2097]
    sig_yield[50] = [2500, 800]
    sig_yield[100] = [4822, 888]
    sig_yield[150] = [4215, 616]
    sig_yield[200] = [3505, 448]
    sig_yield[300] = [2316, 249]
    sig_yield[400] = [1470, 140]
    sig_yield[500] = [927, 80]
    coup =  2.1e-5
    sig_yield = pd.DataFrame(sig_yield)
    factor = (1e-3 / coup)**2
    plt.hist(valid_z * factor, bins=70)
    plt.yscale('log')
    plt.xlabel('Vertex position (z) in mm')
    plt.title(f'Displacement for {mass} Prima, coupling = {coup}')
    #doing displacement scaling
    dz_median = np.median(valid_z)
    coup_d = []
    coup_d_med = []
    couplings = np.logspace(-6, -2, 50)
    alp_dz = np.array(valid_z)
    #dis scaling
    dis_factor = (1e-3 / couplings)**2
    xsec_factor = (couplings / 1e-3)**2
    for factor in dis_factor:
        coup_d_med.append(dz_median * factor)
    produced = N_prod_proc[mass][0] * xsec_factor
    #the_range = [1e-6, 3e-3]
    fig, ax1 = plt.subplots()
    # Plot the first graph with ax1
    line1, = ax1.plot(couplings, coup_d_med, color='blue', label='Median displacement')
    #ax1.axhline(y=1000)
    ax1.set_xlabel('Coupling')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.axvline(x=prev_limits[mass][0], linestyle='--', color='black', label='Previous limit lower')
    ax1.axvline(x=prev_limits[mass][1], linestyle='--', color='black', label='Previous limit upper')
    line3 = ax1.axhline(y=800, linestyle='--')
    line4 = ax1.axhline(y=5700, linestyle='--', label='Median D 800 and 5700 mm')
    ax1.set_ylabel('Median d', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    # Create the second y-axis with twinx and plot the second graph
    ax2 = ax1.twinx()
    line2, = ax2.plot(couplings, produced, color='red', label='Generated ALPs')
    line5 = ax2.axhline(y=1, linestyle='--', color='red', label='1 Generated ALP')  # Reference line
    #ax1.axhline(y=1e3, linestyle='--', color='grey', label='Median displacement = $10^3$')  # Reference line
    ax2.set_ylabel('Generated ALPs', color='red')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='red')
    light_grey_region = (couplings < prev_limits[mass][0]) | (couplings > prev_limits[mass][1])
    dark_grey_region = np.logical_or(
        np.array(coup_d_med) > 1e3,       # Median displacement > 10^3
        np.array(produced) < 1           # Reconstructed ALPs < 1
    )
    # Define proxy artists for the legend
    light_grey_patch = mpatches.Patch(color='lightgrey', alpha=0.5, label='Previously excluded')
    dark_grey_patch = mpatches.Patch(color='darkgrey', alpha=0.5, label='Displacement > $10^3$ or ALPs < 1')
    # Add legend with both proxy patches and plot labels
    ax1.fill_between(
        couplings,
        1e-5,  # Start of y-shade (smallest log value)
        1e6,   # End of y-shade (largest log value)
        where=light_grey_region,
        color='lightgrey',
        alpha=0.5,
        label='Below coupling limit'
    )
    ax1.fill_between(
        couplings,
        1e-5,  # Start of y-shade (smallest log value)
        1e6,   # End of y-shade (largest log value)
        where=dark_grey_region,
        color='darkgrey',
        alpha=0.5,
        label='Displacement > $10^3$ or ALPs < 1'
    )
    coupling_d_10e3 = couplings[np.abs(np.array(coup_d_med) - 1e3).argmin()]
    ax1.annotate(f'{coupling_d_10e3:.2e}',
                 xy=(coupling_d_10e3, 1e3),
                 xytext=(coupling_d_10e3 * 1.2, 1e4),  # Adjust label position
                 arrowprops=dict(arrowstyle='->', color='blue'),
                 fontsize=15, color='blue')
    # Reconstructed ALPs = 1
    coupling_reco_1 = couplings[np.abs(np.array(produced) - 1).argmin()]
    ax2.annotate(f'{coupling_reco_1:.2e}',
                 xy=(coupling_reco_1, 1),
                 xytext=(coupling_reco_1 * 1.2, 10),  # Adjust label position
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=15, color='red')
    handles = [line1, line2, line3, line4, line5, light_grey_patch, dark_grey_patch]
    # Add a title and show the plot
    plt.xlim(1e-6, 3e-3)
    plt.legend(handles=handles, loc='upper right', fontsize=10, markerscale=0.8, handlelength=1.5, labelspacing=0.8)
    plt.title(f'Displacement and # of ALPs for {mass} Prima')
    fig.tight_layout()  # Adjust layout to avoid overlap
    plt.show()
    for factor in dis_factor:
        coup_d.append(alp_dz * factor)
        coup_d_med.append(dz_median * factor)
    #ax.set_yscale('log')

if __name__ == '__main__':
    main()
