#!/usr/bin/python
import argparse
import importlib
import ROOT
import numpy as np
from ROOT import TTree, TBranch, TFile
ROOT.gSystem.Load("/Users/sophie/LDMX/software/NewClone/ldmx-sw/install/lib/libFramework.so")	;
import os
import math
import sys
#import numpy as np
import matplotlib.pyplot as plt
from array import array
from optparse import OptionParser
sys.path.insert(0, '../')
import csv
layer_weights = ([
    2.312, 4.312, 6.522, 7.490, 8.595, 10.253, 10.915, 10.915, 10.915, 10.915, 10.915,
    10.915, 10.915, 10.915, 10.915, 10.915, 10.915, 10.915, 10.915, 10.915, 10.915,
    10.915, 10.915, 14.783, 18.539, 18.539, 18.539, 18.539, 18.539, 18.539, 18.539,
    18.539, 18.539, 9.938
])
mip_si_energy = 0.130 #MeV - corresponds to ~3.5 eV per e-h pair <- derived from 0.5mm thick Si

def weights( eot, size, nent, xsec):
    factor = 1
    list = []
    for i in range(0,size):
        if eot == 1e16:
            factor = (22*int(xsec)*74)/int(nent)
        if eot == 4e14:
            factor = (0.88*int(xsec)*74)/int(nent)
        list.append(factor)
    return list

class Event():
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []
        self.e = []
        self.pid = []

class WabEvent:

    def __init__(self, fn1):

        self.fin = ROOT.TFile(fn1);
        self.tin = self.fin.Get("LDMX_Events")
        # Access the Data:
        #self.evHeader1 = ROOT.ldmx.EventHeader()
        self.simParticles = ROOT.std.map(int, 'ldmx::SimParticle')();
        self.hcalHits      = ROOT.std.vector('ldmx::HcalHit')()
        self.ecalHits      = ROOT.std.vector('ldmx::EcalHit')()
        self.tin.SetBranchAddress("HcalRecHits_PF", ROOT.AddressOf( self.hcalHits ));
        self.tin.SetBranchAddress("EcalRecHits_PF", ROOT.AddressOf( self.ecalHits ));
        self.tin.SetBranchAddress("SimParticles_PF", ROOT.AddressOf( self.simParticles ));
        #self.fin.Close();



    def loop(self,niter):
        nentries = self.tin.GetEntriesFast();
        layers=[]
        nhits=[]
        numerical_pdg=[]
        numerical_proc=[]
        for i in range(nentries):
            #print("=========",i,"============")
            self.tin.GetEntry(i);
            hasElecInECal = False
            hasPhotInECal= False
            hasBack = False
            sumEcal=0
            sumHcal=0
            sumBack=0

            for p, part in enumerate(self.simParticles):
                if (abs(part.second.getEndPoint()[0])>450 or abs(part.second.getEndPoint()[1]) < 450 and part.second.getEndPoint()[2]< 750 and part.second.getEndPoint()[2]> 250):

                    if(part.second.getProcessType()==13):
                        numerical_proc.append("primary")
                    elif(part.second.getProcessType()==5):
                        numerical_proc.append("eBrem")
                    elif(part.second.getProcessType()==6):
                        numerical_proc.append("eIoni")
                    elif(part.second.getProcessType()==3):
                        numerical_proc.append("conv")
                    elif(part.second.getProcessType()==9):
                        numerical_proc.append("PN")
                    else:
                        numerical_proc.append("other")

                    if part.second.getPdgID() == -11:
                        numerical_pdg.append("e+")
                    elif part.second.getPdgID() == 11:
                        numerical_pdg.append("e-")
                    elif part.second.getPdgID() == 22:
                        numerical_pdg.append("gamma")
                    elif abs(part.second.getPdgID()) == 13 or abs(part.second.getPdgID()) == 14:
                        numerical_pdg.append("muon")
                    elif part.second.getPdgID() == 2112:
                        numerical_pdg.append("neutron")
                    else:
                        numerical_pdg.append("other")
                if part.second.getPdgID() == 11 and part.second.getProcessType() == 13:
                    #if (abs(part.second.getEndPoint()[0])<400 and abs(part.second.getEndPoint()[1]) < 300 and part.second.getEndPoint()[2]< 750 and part.second.getEndPoint()[2]> 250):
                    hasElecInECal = True
                    #print("electron",part.second.getEnergy(),"at ",part.second.getEndPoint()[0],part.second.getEndPoint()[1],part.second.getEndPoint()[2])
                if part.second.getPdgID() == 22 and part.second.getProcessType() == 13:
                    #if (abs(part.second.getEndPoint()[0])<400 and abs(part.second.getEndPoint()[1]) < 300 and part.second.getEndPoint()[2]< 750 and part.second.getEndPoint()[2]> 250):
                    #print("photon",part.second.getEnergy(),"at ",part.second.getEndPoint()[0],part.second.getEndPoint()[1],part.second.getEndPoint()[2])
                    hasPhotInECal = True
            """
            for hit in self.hcalHits:
                if(abs(hit.getSection())==0):
                    hasBack = True
            """
            for hit in self.ecalHits:
                if (hit.isNoise() is False):
                    sumEcal +=  hit.getEnergy()
            #print("total in ecal", sumEcal)
            for hit in self.hcalHits:
                if (hit.isNoise()==0):
                    if(abs(hit.getSection())!=0):
                        if hasElecInECal == True:
                            layers.append(hit.getLayer())
                            nhits.append(len(self.hcalHits))
                            sumHcal += hit.getEnergy()
                    if(abs(hit.getSection())==0):
                        if hasElecInECal == True:
                            sumBack+=1
            #print("total in side hcal",sumHcal)
            #print("total in back hcal",sumBack)
            #print("total edep",sumEcal+sumHcal+sumBack)
        fig, ax = plt.subplots()
        # Sample categorical data
        category_pdg = [ 'e-', 'gamma','e+','neutron','muon', 'other']
        category_proc = ["primary", 'conv', 'eBrem','eIoni','PN','other']#

        histogram = np.zeros((len(category_pdg ), len(category_proc)), dtype=int)

        # Populate the histogram
        for x, y in zip(numerical_pdg, numerical_proc):
            x_index = category_pdg.index(x)
            y_index = category_proc.index(y)
            histogram[y_index, x_index] += 1

        # Create the plot
        fig, ax = plt.subplots()

        # Display the histogram as an image
        im = ax.imshow(histogram, cmap="rainbow",vmin=20000)

        # Set axis labels and ticks
        ax.set_xticks(np.arange(len(category_pdg)))
        ax.set_yticks(np.arange(len(category_proc)))
        ax.set_xticklabels(category_pdg)
        ax.set_yticklabels(category_proc)

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add a colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Frequency", rotation=-90, va="bottom")

        # Add title
        ax.set_title("Particles in Side HCal")
        fig.tight_layout()
        plt.savefig("simparts"+str(niter)+".pdf")
        plt.show()
        return nentries, layers, nhits

def main(options,args) :
    all_layers = []
    all_hits = []
    all_xsecs = []
    all_nentries = []
    proc=[]
    colors=["red","blue","green"]
    err=["r.","b.","g."]
    with open(options.ifile, 'r') as file:
        csv_reader = csv.reader(file)
        for i, line in enumerate(csv_reader):
            print(line[1])
            sc = WabEvent(line[1]) ;
            proc.append(line[0])
            nentries,layers, nhits = sc.loop(i);
            all_nentries.append(nentries)
            all_layers.append(layers)
            all_hits.append(nhits)
            all_xsecs.append(line[2])




    fig, ax = plt.subplots()

    w1e16_perevent_all = []

    for l, layers in enumerate(all_layers):
        w1e16_perevent=[]
        w1e16= weights(1e16,len(layers),nentries,all_xsecs[l])
        for i, w in enumerate(w1e16):
            if len(nhits) != 0:
                w1e16_perevent.append(w/nhits[l])
            else:
                w1e16_perevent.append(0)
        w1e16_perevent_all.append(w1e16_perevent)
    n, bins, patches = ax.hist(all_layers, bins=25,histtype='step', stacked=True,color=colors,range=(0,26),weights=w1e16_perevent_all,label=proc)
    plt.xlabel('layer number in side HCal')
    plt.title("1e16 EOT, all wide-angle backgrounds")
    #plt.text(17, 5e6, 'LDMX Simulation', fontsize=12)
    plt.ylabel('#events per layer')
    plt.yscale('log')
    ax.legend()
    plt.savefig("HasECalcompare1e16.pdf")
    plt.show()

    fig, ax = plt.subplots()
    w4e14_perevent_all = []

    for l, layers in enumerate(all_layers):
        w4e14_perevent=[]
        w4e14= weights(4e14,len(layers),nentries,all_xsecs[l])
        for i, w in enumerate(w4e14):
            if len(nhits) != 0:
                w4e14_perevent.append(w/nhits[l])
            else:
                w4e14_perevent.append(0)
        w4e14_perevent_all.append(w4e14_perevent)
    n, bins, patches = ax.hist(all_layers, bins=25,histtype='step', stacked=True,color=colors,range=(0,26),weights=w4e14_perevent_all,label=proc)
    plt.xlabel('layer number in side HCal')
    plt.title("4e14 EOT, all wide-angle backgrounds")
    #plt.text(17, 2e5, 'LDMX Simulation', fontsize=12)
    plt.ylabel('#events per layer')
    plt.yscale('log')
    ax.legend()
    plt.savefig("HasECalcompare4e14.pdf")
    plt.show()

    print("finished main")


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option('-b', action='store_true', dest='noX', default=False, help='no X11 windows')
    # input data files (4)
    parser.add_option('-a','--ifile', dest='ifile', default = 'file1.root',help='directory with data1', metavar='idir1')
    parser.add_option('-o','--ofile', dest='ofile', default = 'ofile.root',help='directory to write plots', metavar='odir')
    parser.add_option('--type', dest='type', default = '1',help='type of process', metavar='type')
    parser.add_option('--tag', dest='tag', default = '1',help='file tag', metavar='tag')
    parser.add_option('--event_tag', dest='event_tag', default = '1',help='file tag', metavar='event_tag')

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
    # Get the Event library
    ROOT.gSystem.Load("/Users/sophie/LDMX/software/ldmx-sw/install/lib/libFramework.so")	;
    main(options,args);
