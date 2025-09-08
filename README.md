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
## table of material list (26/08/2025):
Valid materials are: sandstone, limestone, haematite, goethite, iron ore, feo, peek, al, xe, ti64, hardwood, softwood, ti, pmma, greenalite, dolomite, pyrex, teflon, siderite, pyrite, chalcopyrite, magnetite, xe500psi, air, nai1.5mol, spt, spt0.63mol, spt0.27mol, clastic, glass, carbonate, marble, caco3, titanium, wustite, acrylic.
