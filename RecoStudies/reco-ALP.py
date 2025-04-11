#from Tyler
from LDMX.Framework import ldmxcfg
from LDMX.SimCore import simulator
import sys

#Example command:
#ldmx fire ap_producer.py <run number> <dark brem file> <ap decay file> <output file> <number of events>

proc = 'v14'
p = ldmxcfg.Process(proc)
p.outputFiles = ["m20_g_0.00352.root"]
p.maxEvents = 50000
p.logFrequency = 1
p.termLogLevel = 4
p.fileLogLevel = 4
p.run = 1

sim = simulator.simulator('visible_ALP')
sim.description = "ALP -> gg visible signal decay"
sim.setDetector('ldmx-det-v14-8gev',True)

# Generators
from LDMX.SimCore import generators

# Enable the LHE generator
sim.generators.append(generators.lhe( "Signal Generator", "m20_g_0.00352.lhe"))

# Producers
from LDMX.Ecal import EcalGeometry
from LDMX.Hcal import HcalGeometry
import LDMX.Ecal.ecal_hardcoded_conditions as ecal_conditions
import LDMX.Hcal.hcal_hardcoded_conditions as hcal_conditions
import LDMX.Ecal.digi as ecal_digi
import LDMX.Ecal.vetos as ecal_vetos
import LDMX.Hcal.hcal as hcal_py
import LDMX.Hcal.digi as hcal_digi
from LDMX.Recon.simpleTrigger import simpleTrigger

p.sequence=[
        sim,
        ecal_digi.EcalDigiProducer(),
        ecal_digi.EcalRecProducer(),
        ecal_vetos.EcalVetoProcessor(),
        hcal_digi.HcalDigiProducer(),
        hcal_digi.HcalRecProducer(),
        hcal_py.HcalVetoProcessor(),
        ]

#p.pause()
