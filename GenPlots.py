# Example of how to use the LHE Reader for an ALP decay (diphoton) example
from lhereader import readLHEF
from ROOT import TCanvas, TH1F, TH2F, TLorentzVector, TF1
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.preprocessing import normalize
import matplotlib as mpl # for plotting
import matplotlib.pyplot as plt # common shorthand
import mplhep # style of plots

mpl.style.use(mplhep.style.ROOT) # set the plot style

path = "/Users/sophie/LDMX/software/NewClone/ALPs/displaced/"

def main(args):
    # Extract photons:

    pts = []
    pzs = []
    elecpzs=[]
    angles = []
    angles_z =[]
    alpenergies = []
    alpfracenergies = []
    alpsangles = []
    alppxs = []
    alppts = []
    alppys = []
    alppzs = []
    alpdises=[]
    alpdzs = []
    weightz = []
    weightsg = []
    weights1g = []
    for i, data in enumerate(args.fullfilename):
        data=[]
        print(str(args.fullfilename[i]))
        data_list=readLHEF(str(args.fullfilename[i]))
        photons=data_list.getParticlesByIDs([22])
        ALPs=data_list.getParticlesByIDs([666])
        electrons=data_list.getParticlesByIDs([11])

        # make an empty vector to fill with the quantity you want to plot
        pt = []
        pz = []
        angle = []
        anglez = []
        nphoton=0
        weight = []
        weightg = []
        weight1g = []
        elecpz=[]
        # TLorentzVector for the two photons
        gamma1_4mom = TLorentzVector()
        gamma2_4mom = TLorentzVector()
        # Loop over all photons:
        for g in photons:
            # Outgoing photons (status ==1):
            if (g.status == 1):
                nphoton+=1
                # all photons
                pt.append(g.p4.Pt())
                pz.append(g.p4.Pz())
                weight1g.append(1/nphoton)
                # to get details of each photon (assume two per event in even structure - this should be OK)
                if nphoton%2!=0:
                    print(nphoton, "this is the outgoing first photon in event")
                    gamma1_4mom = g.p4
                if nphoton%2==0:
                    print(nphoton,"this is the outgoing second photon in event")
                    gamma2_4mom = g.p4
                    # angle between the two photons
                    print("---------------------")
                    angle.append(gamma1_4mom.Angle(gamma2_4mom.Vect()))
                    weightg.append(1/nphoton)

        pts.append(pt)
        pzs.append(pz)
        angles.append(angle)
        weightsg.append(weightg)
        weights1g.append(weight1g)
        nALPs = 1
        alp_energy = []
        alp_frac_e = []
        alp_px = []
        alp_py = []
        alp_pz = []
        alp_angle = []
        alp_p = []
        alp_pt = []
        alp_vtims = []
        alp_mass = []
        alp_gammas = []
        alp_dz = []
        alp_dis = []
        velocities = []
        vtims = []
        for a in ALPs:
            alp_energy.append(a.energy)
            alp_frac_e.append(a.energy/8)
            alp_px.append(a.p4.Px())
            alp_py.append(a.p4.Py())
            alp_pz.append(a.p4.Pz())
            alp_pt.append(math.sqrt(a.p4.Px()*a.p4.Px() + a.p4.Py()*a.p4.Py()))
            alp_angle.append(math.atan(a.p4.Px()/a.p4.Pz()))
            alp_vtims.append(a.vtim)
            alp_mass.append(a.mass)
            alp_gammas.append(a.p4.Gamma())
            alp_p.append(a.p4.P())
            velocity = a.p4.Vect() / (a.p4.Gamma() * a.mass) #in units of c?
            velocities.append(velocity[2])
            displacement = velocity * a.vtim   #in mm
            vtims.append(a.vtim)
            #calculate magnitude
            alp_dis.append(np.sqrt(displacement[0]**2 + displacement[1]**2 + displacement[2]**2))
            alp_dz.append(displacement[2])
            weight.append(1/(nALPs))
            nALPs+=1
        alpenergies.append(alp_energy)
        alpfracenergies.append(alp_frac_e)
        alppxs.append(alp_px)
        alppys.append(alp_py)
        alppzs.append(alp_pz)
        alppts.append(alp_pt)
        alpsangles.append(alp_angle)
        alpdises.append(alp_dis)
        alpdzs.append(alp_dz)
        weightz.append(weight)
        for e in electrons:
            if (e.status == 1):
                elecpz.append(e.p4.Pz())
        elecpzs.append(elecpz)

    styles = ["dashed","solid"]#,"solid","solid","solid","solid"]
    colors = ["orange","orange"]#,"blue","orange","magenta","red"]
    masses = ["500MeV","500MeV"]#,"100MeV","200MeV","300MeV","500MeV"]
    procs = ["photon fusion","primakoff"]#,"prima","prima","prima","prima"]
    labels = ["photon fusion","primakoff"]#"masses[0],masses[1],masses[2],masses[3],masses[4],masses[5]]
    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(pts):

        plt.title("Photon Transverse Momentum")
        n, bins, patches = ax.hist(pts[i],
                                   bins=50,
                                   range=(0,2),
                                   histtype = 'step',
                                   linestyle = styles[i],
                                   color = colors[i],
                                   label=labels[i],
                                   weights=weights1g[i])
    #plt.legend()
    ax.set_yscale('log')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('events per bin')
    ax.set_xlabel('pt of all photons [MeV/c]')
    fig.savefig('pt.pdf')

    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(pzs):

        plt.title("Photon Pz Momentum")
        n, bins, patches = ax.hist(pzs[i],
                                   bins=100,
                                   range=(0,8),
                                   histtype = 'step',
                                   linestyle = styles[i],
                                   color = colors[i],
                                   label=labels[i],
                                   weights=weights1g[i])
    #plt.legend()
    ax.set_yscale('log')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('Arb. Units')
    ax.set_xlabel('pz of all photons [MeV/c]')
    fig.savefig('pz.pdf')



    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(angles):

        plt.title("Angle between photons")
        n, bins, patches = ax.hist(angles[i],
                                   bins=25,
                                   range=(0,math.pi),
                                   histtype = 'step',
                                   linestyle = styles[i],
                                   color = colors[i],
                                   label=labels[i],
                                   weights=weightsg[i])
                                   # Put a legend to the right of the current axis

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_yscale('log')
    ax.set_ylabel('Arb. Units')
    ax.set_xlabel('angle between photons [rad]')
    fig.savefig('angle.pdf')

    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(alpenergies):

        #plt.title("mALP= "+str(args.mass)+" MeV")
        #normalize([alpenergies[i]])[0],
        n, bins, patches = ax.hist( alpenergies[i],
                                    bins=25,
                                    range=(0,10),
                                    histtype = 'step',
                                    linestyle = styles[i],
                                    color = colors[i],
                                    label=labels[i],
                                    weights=weightz[i])

    #plt.legend()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_yscale('log')
    ax.set_ylabel('Arb. Units')
    ax.set_xlabel('ALP Energy [GeV]')
    fig.savefig('alp_energy.pdf')

    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(alpenergies):

        #plt.title("mALP= "+str(args.mass)+" MeV")
        #normalize([alpenergies[i]])[0],
        n, bins, patches = ax.hist( alpfracenergies[i],
                                    bins=25,
                                    range=(0,1),
                                    histtype = 'step',
                                    linestyle = styles[i],
                                    color = colors[i],
                                    label=labels[i],
                                    weights=weightz[i])

        #plt.legend()
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_yscale('log')
        ax.set_ylabel('Arb. Units')
        ax.set_xlabel('ALP Energy/Beam Energy')
        fig.savefig('alp_energy_frac.pdf')


    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(alpenergies):
        #plt.title("mALP= "+str(args.mass)+" MeV")
        #normalize([alpenergies[i]])[0],
        n, bins, patches = ax.hist( alppxs[i],
                                    bins=50,
                                    range=(-3,3),
                                    histtype = 'step',
                                    linestyle = styles[i],
                                    color = colors[i],
                                    label=labels[i],
                                    weights=weightz[i])
    #plt.legend()
    ax.set_yscale('log')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('Arb. Units')
    ax.set_xlabel('ALP Px [GeV/c]')
    fig.savefig('alp_px.pdf')

    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(alpenergies):
        #plt.title("mALP= "+str(args.mass)+" MeV")
        #normalize([alpenergies[i]])[0],
        n, bins, patches = ax.hist( alppts[i],
                                    bins=50,
                                    range=(0,3),
                                    histtype = 'step',
                                    linestyle = styles[i],
                                    color = colors[i],
                                    label=labels[i],
                                    weights=weightz[i])
    #plt.legend()
    ax.set_yscale('log')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('Arb. Units')
    ax.set_xlabel('ALP Pt [GeV/c]')
    fig.savefig('alp_pt.pdf')

    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(alpenergies):
        n, bins, patches = ax.hist( alppys[i],
                                    bins=50,
                                    range=(-1,1),
                                    histtype = 'step',
                                    linestyle = styles[i],
                                    color = colors[i],
                                    label=labels[i],
                                    weights=weightz[i])
    #plt.legend()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_yscale('log')
    ax.set_ylabel('Arb. Units')
    ax.set_xlabel('ALP Py [MeV/c]')
    fig.savefig('alp_py.pdf')

    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(alpenergies):
        n, bins, patches = ax.hist( alppzs[i],
                                    bins=50,
                                    range=(0,10),
                                    histtype = 'step',
                                    linestyle = styles[i],
                                    color = colors[i],
                                    label=labels[i],
                                    weights=weightz[i])
    #plt.legend()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_yscale('log')
    ax.set_ylabel('Arb. Units')
    ax.set_xlabel('ALP Pz [MeV/c]')
    fig.savefig('alp_pz.pdf')

    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(alpenergies):
        n, bins, patches = ax.hist( alpsangles[i],
                                    bins=25,
                                    range=(-math.pi/2,math.pi/2),
                                    histtype = 'step',
                                    linestyle = styles[i],
                                    color = colors[i],
                                    label=labels[i],
                                    weights=weightz[i])
    #plt.legend()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_yscale('log')
    ax.set_ylabel('Arb. Units')
    ax.set_xlabel('ALP Angle [rad]')
    fig.savefig('alp_angle.pdf')

    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(alpdises):

        plt.title("ALP Displacement Magnitude")
        n, bins, patches = ax.hist(alpdises[i],
                                   bins=100,
                                   range=(250,5500),
                                   histtype = 'step',
                                   linestyle = styles[i],
                                   color = colors[i],
                                   label=labels[i],
                                   weights=weightz[i])
    #plt.legend()
    ax.set_yscale('log')
    #ax.set_xscale('log')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('Arb. Units')
    ax.set_xlabel('ALP Displacement Mag [mm]')
    fig.savefig('alpdis.pdf')

    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(alpdzs):

        plt.title("ALP Displacement in Z")
        n, bins, patches = ax.hist(alpdzs[i],
                                   bins=25,
                                   range=(0,10),
                                   histtype = 'step',
                                   linestyle = styles[i],
                                   color = colors[i],
                                   label=labels[i],
                                   weights=weightz[i])
    #plt.legend()
    ax.set_yscale('log')
    #ax.set_xscale('log')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('Arb. Units')
    ax.set_xlabel('ALP Displacement Z [mm]')
    fig.savefig('alpdz.pdf')


    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(alppzs):

        #plt.title("Photon Pz Momentum")
        plt.hist2d(alppzs[i], alpsangles[i], bins=(50,50), cmap=plt.cm.jet)
        ax.set_ylabel('ALP Pz [MeV/c]')
        ax.set_xlabel('ALP Angle [rad]')
        fig.savefig('ALP2D_angle_energy'+str(i)+'.pdf')

    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(elecpzs):

        plt.title("Electron Pz Momentum")
        n, bins, patches = ax.hist(elecpzs[i],
                                   bins=100,
                                   range=(0,8),
                                   histtype = 'step',
                                   linestyle = styles[i],
                                   color = colors[i],
                                   label=labels[i])

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_yscale('log')
    ax.set_ylabel('Arb. Units')
    ax.set_xlabel('pz of electrons [MeV/c]')
    fig.savefig('elecpz.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    files = ["m500_PF.lhe","m500_prima.lhe"]#["m10_prima_vertex.lhe","m50_prima_vertex.lhe","m100_prima_vertex.lhe","m200_prima_vertex.lhe","m300_prima_vertex.lhe","m500_prima_vertex.lhe"]
    parser.add_argument("--fullfilename", help="full filename with path", default=files)
    parser.add_argument("--process", help="Primakoff or Photon Fusion")
    parser.add_argument("--mass", help="ALP mass")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)
