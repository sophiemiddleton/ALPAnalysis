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

        # output files:
        #self.fn_out = ofn;
        #self.fout = ROOT.TFile("hist_"+self.fn_out,"RECREATE");

        #list of branches:
        self.evHeader1 = ROOT.ldmx.EventHeader()
        self.hcalRecHits = ROOT.std.vector('ldmx::HcalHit')();
        self.ecalRecHits = ROOT.std.vector('ldmx::EcalHit')();
        self.ecalSPHits    = ROOT.std.vector('ldmx::SimTrackerHit')();
        self.ecalVeto = ROOT.ldmx.EcalVetoResult()
        self.simParticles = ROOT.std.map(int, 'ldmx::SimParticle')();
        self.tin1.SetBranchAddress("EventHeader",  ROOT.AddressOf( self.evHeader1 ));
        self.tin1.SetBranchAddress("HcalRecHits_v14",  ROOT.AddressOf( self.hcalRecHits ));
        self.tin1.SetBranchAddress("EcalRecHits_v14",  ROOT.AddressOf( self.ecalRecHits ));
        self.tin1.SetBranchAddress("EcalVeto_v14",  ROOT.AddressOf( self.ecalVeto ));
        self.tin1.SetBranchAddress("EcalScoringPlaneHits_v14", ROOT.AddressOf( self.ecalSPHits ));
        self.tin1.SetBranchAddress("SimParticles_v14", ROOT.AddressOf( self.simParticles ));
        # loop and save:
        self.loop();


    def loop(self):
        f = TFile('m20_g_1.8e-5_ntuple.root', 'RECREATE')
        Features = TTree( 'Features', 'Information about events' )

        NHits = array('i',[0])
        Features.Branch("NHits",  NHits,  'NHits/I')
        xMean = array('f',[0])
        Features.Branch("xMean", xMean,"xMean/F")
        yMean = array('f',[0])
        Features.Branch("yMean", yMean,"yMean/F")
        zMean = array('f',[0])
        Features.Branch('zMean', zMean,"zMean/F")
        HCALE = array('f',[0])
        Features.Branch("HCALE", HCALE,"HCALE/F")
        xStd = array('f',[0])
        Features.Branch("xStd", xStd,"xStd/F")
        yStd = array('f',[0])
        Features.Branch("yStd", yStd,"yStd/F")
        zStd = array('f',[0])
        Features.Branch('zStd', zStd,"zStd/F")
        rStd = array('f',[0])
        Features.Branch('rStd', rStd,"rStd/F")
        rMean = array('f',[0])
        Features.Branch('rMean', rMean,"rMean/F")
        isoHits = array('i',[0])
        Features.Branch('isoHits', isoHits,"isoHits/I")
        isoE =  array('f',[0])
        Features.Branch('isoE', isoE,"isoE/F")

        ECALE = array('f',[0])
        Features.Branch('ECALE', ECALE,"ECALE/F")
        ECALBDT = array('f',[0])
        Features.Branch('ECALBDT', ECALE,"ECALBDT/F")
        firstHitZ = array('f',[0])
        Features.Branch('firstHitZ', firstHitZ,"firstHitZ/F")

        isSignal = array('i',[0])
        Features.Branch("isSignal",  isSignal,  'isSignal/I')

        nent = self.tin1.GetEntriesFast();
        all = 0
        passesVeto=0
        passesECALE = 0
        passesTrack = 0
        passesBackHCALE = 0
        passesContain = 0
        passes_all = 0
        passesTrue = 0
        last_z = 1000000
        ALPTrueDecay = 0
        ALPRecod = 0
        decay_vert_z_true = []
        decay_vert_z_rec = []
        decay_vert_z_bdt = []
        for i in range(nent):

            self.tin1.GetEntry(i);
            layersHit = []

            isoHits[0] = 0
            isoE[0] = 0
            NHits[0]=0
            xMean[0] = 0
            yMean[0] = 0
            zMean[0] = 0
            rMean[0] = 0

            xStd[0] = 0
            yStd[0] = 0
            zStd[0] = 0
            rStd[0] = 0
            HCALE[0] = 0
            ECALE[0] = 0
            ECALBDT[0] = 0
            firstHitZ[0] = 0

            isSignal[0] = 0
            sumE = 0
            sumECALE =0


            for ih,hit in enumerate(self.ecalRecHits):
                sumECALE += hit.getEnergy() #12.2*
            decayz = 1000000
            for p, part in enumerate(self.simParticles):
                    parents = part.second.getParents()
                    for track_id in parents:
                        if track_id == 0 and part.second.getPdgID() == 22: # ALP decay vertex
                            #print(decayz)
                            #if part.second.getVertex()[2] > 750 and part.second.getVertex()[2] < 5500:
                            if decayz != 1000000:
                                ALPTrueDecay +=1
                                print(decayz)
                                decay_vert_z_true.append(decayz)
                            decayz = part.second.getVertex()[2]

            #hcal colleciton
            for ih,hit in enumerate(self.hcalRecHits):
                firstHitZ[0] = hit.getZPos()
                NHits[0] = len(self.hcalRecHits)
                if (hit.isNoise()==0):
                    if hit.getZPos() >= 870 :
                        if decayz < last_z and decayz>250:
                            last_z = decayz

                        x = hit.getXPos()
                        y = hit.getYPos()
                        z= hit.getZPos()

                        sumE += hit.getEnergy() #12.2*
                        r = math.sqrt(x*x + y*y)

                        xMean[0] += x*hit.getEnergy()
                        yMean[0] += y*hit.getEnergy()
                        zMean[0] += z*hit.getEnergy()
                        rMean[0] += r*hit.getEnergy()

                        if not z in layersHit :
                            layersHit.append(z)

                        closestpoint = 9999
                        for ih,hit2 in enumerate(self.hcalRecHits):
                            if abs(z - hit2.getZPos()) == 0 :
                                sepx = math.sqrt((x-hit2.getXPos())**2)
                                sepy = math.sqrt((y-hit2.getYPos())**2)
                                if sepx > 0 and sepx%50 == 0 :
                                    if sepx < closestpoint :
                                        closestpoint = sepx
                                elif sepy > 0 and sepy%50 == 0 :
                                    if sepy < closestpoint :
                                        closestpoint = sepy
                        if closestpoint > 50:
                            isoHits[0] += 1
                            isoE[0] += hit.getEnergy()
            HCALE[0] = sumE
            if(sumE != 0):
                xMean[0] /= sumE
                yMean[0] /= sumE
                zMean[0] /= sumE
                rMean[0] /= sumE

            for ih,hit in enumerate(self.hcalRecHits):
                if (hit.isNoise()==0):
                    if hit.getZPos() >= 870 :
                        x = hit.getXPos()
                        y = hit.getYPos()
                        z = hit.getZPos()
                        energy = hit.getEnergy()
                        r = math.sqrt(x*x + y*y)
                        xStd[0] += energy*(x-xMean[0])**2
                        yStd[0] += energy*(y-yMean[0])**2
                        zStd[0] += energy*(z-zMean[0])**2
                        rStd[0] += energy*(r-rMean[0])**2

            if sumE !=0:
                xStd[0] = math.sqrt(xStd[0]/sumE)
                yStd[0] = math.sqrt(yStd[0]/sumE)
                zStd[0] = math.sqrt(zStd[0]/sumE)
                rStd[0] = math.sqrt(rStd[0]/sumE)
            isSignal[0] = 1
            if sumE!=0 and decayz > 750 and decayz!=1000000:
                ALPRecod +=1
            #if (abs(last_z - first_z)) != 0 and len(z_positions) != 0 and len(energies) != 0:
            if sumE !=0:
                Features.Fill()
                all+=1

            if sumE!=0 and decayz!=1000000:
                decay_vert_z_rec.append(decayz)


            ECALBDT[0] =  self.ecalVeto.getDisc()
            if(sumE!=0 and decayz !=1000000 and ECALBDT[0] > 0.9996):
                passesTrue+=1
                passesVeto+=1
                decay_vert_z_bdt.append(decayz)

            if(ECALE[0] < 3160):
                passesECALE += 1
            if sumE !=0:
                passesBackHCALE +=1
            if firstHitZ[0] > 879.:
                passesContain += 1
            if(ECALBDT[0] > 0.9996 and ECALE[0] < 3160 and sumE !=0 ):
                passes_all +=1


        print("cuts",all)
        print("passes ECAL E", passesECALE/nent)
        print("passes ECAL BDT ", passesVeto/nent)
        print("passes contain", passesContain/nent)
        print("passes HCAL E", passesBackHCALE/nent)
        print("passes all",passes_all/nent)
        #print("has true ALP or photon",ALPRecod/ALPTrueDecay)
        print("has true ALP or photon that is Recos",passesTrue)
        print("earliest decay reconstructed",last_z)
        #make plot for coordinate position
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
        nbdt, bins, patches = ax.hist(decay_vert_z_bdt, bins=100,histtype='step',range=(0,5000),color='green',label="ecal bdt passed")
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        plt.errorbar(bin_centers, nbdt, yerr=np.sqrt(nbdt), fmt='g.')
        plt.yscale('log')
        plt.xlim(300, 5000)
        ax.set_xlabel('true vertex position z [mm]')
        ax.legend()
        plt.savefig("truevertex.pdf")
        plt.show()

        fig, ax = plt.subplots(1,1)
        ax.minorticks_on()
        plt.text(300, 2, 'LDMX Simulation', fontsize=12)
        ax.plot(bins[:-1],
                nrec / ntrue,c="black")
        ax.plot(bins[:-1],
                nbdt / ntrue,c="red")
        plt.yscale('log')
        plt.xlim(300, 5000)
        ax.set_xlabel('true vertex z [mm]')
        ax.set_ylabel('eff')

        plt.savefig("ratiovertex.pdf")
        plt.show()

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
