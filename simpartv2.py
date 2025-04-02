#!/usr/bin/python

# ldmx python3 MakeRootTree.py --ifile reco.root
import argparse
import importlib
import ROOT
from ROOT import TTree, TBranch, TH1F, TFile
ROOT.gSystem.Load("/Users/sophie/LDMX/software/NewClone/ldmx-sw/install/lib/libFramework.so")	;
import os
import math
import sys
import csv
import numpy as np
from array import array
from optparse import OptionParser
#import matplotlib.pyplot as plt
#sys.path.insert(0, '../')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class GetPart:

    def __init__(self, fn1, ofn, label, mass, tag):

        self.label = label
        self.mass = mass
        #input files:
        self.fin1 = ROOT.TFile(fn1);
        self.tin1 = self.fin1.Get("LDMX_Events")
        self.tag = int(tag);

        #list of branches:
        self.evHeader1 = ROOT.ldmx.EventHeader()
        self.hcalRecHits = ROOT.std.vector('ldmx::HcalHit')();
        self.simParticles = ROOT.std.map(int, 'ldmx::SimParticle')();
        self.tin1.SetBranchAddress("EventHeader",  ROOT.AddressOf( self.evHeader1 ));
        self.tin1.SetBranchAddress("HcalRecHits_v14",  ROOT.AddressOf( self.hcalRecHits ));
        self.tin1.SetBranchAddress("SimParticles_v14", ROOT.AddressOf( self.simParticles ));
        # loop and save:
        self.loop();


    def loop(self):
        f = TFile('test_ntuple.root', 'RECREATE')
        Features = TTree( 'Features', 'Information about events' )
        firstHitZ = array('f',[0])
        Features.Branch('firstHitZ', firstHitZ,"firstHitZ/F")
        nent = self.tin1.GetEntriesFast();

        passesTrue = 0
        last_z = 1000000
        decayz = 1000000

        z = 500
        decay_vert_z_true = []
        decay_vert_z_rec = []

        ALPTrueDecay = 0
        ALPRecod = 0

        for i in range(nent):

            self.tin1.GetEntry(i);
            firstHitZ[0] = 0
            sumE = 0

            decayz = 1000000
            for p, part in enumerate(self.simParticles):
                    parents = part.second.getParents()
                    for track_id in parents:
                        if track_id == 0 and part.second.getPdgID() == 22: # ALP decay vertex
                            #if part.second.getVertex()[2] > z and part.second.getVertex()[2] < 5700:
                            if decayz != 1000000:
                                ALPTrueDecay +=1
                                decay_vert_z_true.append(decayz)
                            decayz = part.second.getVertex()[2]


            #hcal colleciton
            for ih,hit in enumerate(self.hcalRecHits):
                firstHitZ[0] = hit.getZPos()

                if (hit.isNoise()==0):
                    if hit.getZPos() >= 870 :
                        if decayz < last_z and decayz>250:
                            last_z = decayz

                        sumE += hit.getEnergy() #12.2*

            if sumE!=0 and decayz!=1000000:
                decay_vert_z_rec.append(decayz)


        print(len(decay_vert_z_rec),len(decay_vert_z_true))
        print("has true ALP or photon that is Recos",passesTrue)
        print("earliest decay reconstructed",last_z)
        fig, ax = plt.subplots(1,1)
        ax.minorticks_on()
        plt.text(300, 200000, 'LDMX Simulation', fontsize=12)
        ntrue, bins, patches = ax.hist(decay_vert_z_true, bins=100,histtype='step',range=(0,5000),color='red',label="all vertices")
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        plt.errorbar(bin_centers, ntrue, yerr=np.sqrt(ntrue), fmt='r.')
        nrec, bins, patches = ax.hist(decay_vert_z_rec,bins=100,histtype='step',range=(0,5000),color='blue',label="vertices reconstructed")
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        plt.errorbar(bin_centers, nrec, yerr=np.sqrt(nrec), fmt='b.')
        plt.yscale('log')
        plt.xlim(300, 5000)
        ax.set_xlabel('true vertex position z [mm]')
        ax.legend()
        plt.savefig("truevertex.pdf")
        plt.show()
        """
        y = []
        y.append(decay_vert_z_true)
        y.append(decay_vert_z_rec)
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        plt.text(0, 200000, 'LDMX Simulation', fontsize=12)
        plt.yscale('log')
        ns, bins, patches = ax1.hist(y,
                              bins=100,
                              histtype='step',
                              color=["red","blue"],
                              label=['reco.','true']
                              )
        ax1.legend()
        """
        fig, ax = plt.subplots(1,1)
        ax.minorticks_on()
        plt.text(300, 2, 'LDMX Simulation', fontsize=12)
        ax.plot(bins[:-1],
                nrec / ntrue,c="black")
        plt.yscale('log')
        plt.xlim(300, 5000)
        ax.set_xlabel('true vertex z [mm]')
        ax.set_ylabel('reco. eff')

        plt.savefig("ratiovertex.pdf")
        plt.show()

        #make plot for coordinate position
        f.Write();
        f.Close();



def main(options,args) :
    sc = GetPart(options.ifile,options.ofile,options.label, options.mass, options.tag);
    #sc.fout.Close();

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option('-b', action='store_true', dest='noX', default=False, help='no X11 windows')
    parser.add_option('-a','--ifile', dest='ifile', default = 'file.root',help='directory with data1', metavar='idir')
    parser.add_option('-o','--ofile', dest='ofile', default = 'ofile.root',help='directory to write plots', metavar='odir')
    parser.add_option('--label', dest='label', default = 'primakoff',help='production model', metavar='label')
    parser.add_option('--mass', dest='mass', default = '10',help='mass of ALP', metavar='mass')
    parser.add_option('--tag', dest='tag', default = '1',help='file tag', metavar='tag')

    (options, args) = parser.parse_args()


    ROOT.gStyle.SetPadTopMargin(0.10)
    ROOT.gStyle.SetPadLeftMargin(0.16)
    ROOT.gStyle.SetPadRightMargin(0.10)
    ROOT.gStyle.SetPalette(1)
    ROOT.gStyle.SetPaintTextFormat("1.1f")
    ROOT.gStyle.SetOptFit(0000)
    ROOT.gROOT.SetBatch()
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)

    main(options,args);
