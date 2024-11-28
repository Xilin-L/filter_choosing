import numpy as np
import matplotlib.pyplot as plt

import skimage as ski

import spekpy
import larch
from larch import xray
from skimage.transform import radon

import xraySimulation as xs
import xrayImagingPerformance as xip
import beamHardeningSimulation as bhs

def getRadiusMm(img, voxelSizeMm=0.1):
    imgIdealised=bhs.idealiseTomo(bhs.enforceBinaryImage(img),threshold=0.4)
    radius=np.sqrt(np.sum(imgIdealised)/np.pi)*voxelSizeMm
    return radius

def normaliseProj(proj,clearField,darkField):
    projNorm= (proj - darkField) / (clearField - darkField)
    projNorm[projNorm < 0]=0
    projNorm[projNorm > 1]=1
    return projNorm

if __name__ == '__main__':

    tomo=

    kvp=120
    filterMaterial='Cu'
    filterThicknessMm=0.5
    materialName='feo'
    sampleThicknessMm= getRadiusMm(tomo)

    energyKeV, spectrum= xs.setSpectrum(kvp,filterMaterial,filterThicknessMm)
    materialWeights, materialSymbols, dens = xip.getMaterialProperties(materialName)
    sampleAttPerCm = xs.getMaterialAttenuation(energyKeV, materialWeights, materialSymbols, dens)
    sampleTransmission = np.exp(-sampleAttPerCm*20*sampleThicknessMm)
