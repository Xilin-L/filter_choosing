# Basic usage

## Rule-of-thumb filter thickness

```python
from filterPerformance import *

# Replace with your values:
kvp = 120  # kVp
filterMaterial = "Al"
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
sampleMaterial = "sandstone"
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
