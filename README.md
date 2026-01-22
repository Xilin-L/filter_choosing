# How to use this software

## To calculate the rule-of-thumb (ROT) thickness of the scan
Fill in the kvp, filter material, sample material, and sample thickness. Then click the 'calculate ROT thickness' button. 
You can apply this value directly by clicking the 'apply ROT thickness' button, or manually input another value.

## To calculate other quantities
The machine model and source mode options try to make the power within the operational range, do not worry too much about them.
Cone angle can affect the result for scattering. The default value is given based on the ANU4 geometry, which estimates the scattering from just the sample itself.
After you fill in all the values (except the optional power), you can click the 'calculate all' button. The results will be displayed in the output box. You may have to scroll down a bit to see the full results.


# Q&A for the current version

## Q: Does the scatter/transmission mean that if it is 10% that 10% of the 0.5% transmission is probably scatter?
A: Yes, you are right. 

## Q: Is the scatter estimate for the source with an aperture or without? Or is the scatter component only scatter from the sample, or does it include other scatter from the CT cabinet? 
A: The scattering calculated only includes the scattering from the sample itself,  with the part that been reflected by the cabinet and detector table. Assuming the geometry of ANU4. (this can be significantly smaller compared to the actual value, but very close to the topas simulation)

## Q: We should also have a filter setting for some of the "composite equivalent" filters as an option. 
A: Yes, I guess this will not have too much difference on bhc and transmission, but more on scattering, if we include the scattering from the filter.

## Q: When selecting a sample material you are assuming that material solid or has some porosity?
A: Here it is solid, I have not added the porosity option yet.

## Q: For ANU4 there are two source modes S and L, but this system does not have two modes?
A: I will fix this in the next version, just use L mode for now.



# Basic usage

## Rule-of-thumb filter thickness

```python
from filterPerformance import *

# Replace with your values:
kvp = 120  # kVp
filterMaterial = "Al"  # can be "Fe" as well
sampleMaterial = "sandstone"
sampleDiameterMm = 25  # mm

thickness_mm = getRuleOfThumbFilterThickness(
    kvp,
    filterMaterial=filterMaterial,
    sampleMaterial=sampleMaterial,
    sampleDiameterMm=sampleDiameterMm,
)
print(thickness_mm)
```

## For transmission:

```python
import numpy as np
import xraySimulation as xs
import materialPropertiesData as mpd
import filterPerformance as fp

# Replace with your values:
sampleMaterial = "sandstone" # options listed below in the next section
sampleDiameterMm = 25   # mm
kvp = 120               # kVp
filterMaterial = "Al"
filterThicknessMm = 1.5 # mm

materialWeights, materialSymbols, dens = mpd.getMaterialProperties(sampleMaterial)
energyKeV, spectrum = fp.setSpectrum(kvp, filterMaterial=filterMaterial, filterThicknessMm=filterThicknessMm)
trans = xs.calcTransmission(
    energyKeV,
    materialWeights,
    materialSymbols,
    dens,
    sampleDiameterMm/10,   # convert mm to cm
    kind='total'
)
transmission = np.sum(trans * spectrum) / np.sum(spectrum)
print(transmission)
```

## For SNR analysis:

```python
from automation.qualityMeasures import *
import matplotlib.pyplot as plt
import os
import re
base_path = 'your/path/to/the/folder/that/contains/images' # such as '/home/xilin/projects/W7-20p33-21p30m__290kV_DeeTee'
sample = QualityMeasuresAnalyzer(directoryPath=base_path, sampleMaterial="sampleMaterial", shape=(2920,2920)) # change "sampleMaterial" to anything within the list below if you don't simulate bhc or scattering, shape is size of detector images

sample.computeSnr()
"""
it print the results in a few minuates (longer for larger images, 30 mins for the example size on my desktop) like:
#### SNR Result ####
SNR X = 10.3240
SNR Y = 53.6396
SNR Z = 9.8023
"""
```


## table of material list (26/08/2025):
Valid materials are: sandstone, limestone, haematite, goethite, iron ore, feo, peek, al, xe, ti64, hardwood, softwood, ti, pmma, greenalite, dolomite, pyrex, teflon, siderite, pyrite, chalcopyrite, magnetite, xe500psi, air, nai1.5mol, spt, spt0.63mol, spt0.27mol, clastic, glass, carbonate, marble, caco3, titanium, wustite, acrylic.
