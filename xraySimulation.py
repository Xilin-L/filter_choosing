#!/usr/bin/env python3

import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
import spekpy
import larch
# from larch import xray
import xraydb as xray

### SOME IMPORTANT INFO ON SPEKPY AND LARCH FUNCTIONS ###

"""
spekpy.Spek()

        Constructor method for the Spek class

        SEE THE WIKI PAGES FOR MORE INFO:
        https://bitbucket.org/spekpy/spekpy_release/wiki/Further_information

        :param float kvp: tube potential [kV] (default: depends on target)
        :param float th: anode angle [degrees] (default: 12 degrees)
        :param float dk: energy bin width [keV] (default: 0.5 keV)
        :param string: mu_data_source (default: depends on physics model)
            options: ('pene' or 'nist')
        :param float physics: physics model (default: 'casim')
            options: ('casim', 'kqp', 'spekpy-v1', 'spekcalc',
                      'diff', 'uni', 'sim', 'classical')
        :param float x: displacement from central axis in anode-cathode
            direction [cm] (default: 0 cm)
        :param float y: displacement from central axis in orthogonal
            direction [cm] (default: 0 cm)
        :param float z: focus-to-detector distance [cm] (default: 100 cm)
        :param float mas: the tube current-time product [mAs] (default: 1 mAs)
        :param logical brem: whether bremsstrahlung x rays requested
            (default: true)
        :param logical char: whether characteristic x rays requested
            (default: true)
        :param logical obli: whether increased oblique paths through filtration
            are assumed for off axis positions (default: true)
        :param string comment: any text annotation the user wishes to add
        :param string targ: the anode target material (default: 'W')
            options: ('W', 'Mo', or 'Rh')
        :param float shift: optional fraction of an energy bin to shift the
            energy bins (useful when matching to measurements) (default: 0.0)
"""

"""
spekpy.get_spectrum(self,edges=False, flu=True, diff=True, sig=None,
                     addend=False,**kwargs):

        A method to get the energy and spectrum for the parameters in the
        current spekpy state

        :param bool edges: Keyword argument to determine whether midbin or edge
            of bins data are returned
        :param bool addend: Keyword argument to determine whether a zero
            end point is added to the spectrum
        :param bool flu: Whether to return fluence or energy-fluence
        :param bool diff: Whether to return spectrum differential in energy
        :param kwargs: Keyword arguments to change parameters that are used for
            the calculation
        :return array k: Array with photon energies (mid-bin or edge values)
            [keV]
        :return array spk: Array with corresponding photon fluences
            [Photons cm^-2 keV^-1], [Photons cm^-2] or [Photons cm^-2 keV^1]
            depending of values of flu and diff inputs
"""

"""
 _xray.mu_elam(z_or_symbol, energyEV, kind='total')

    return X-ray mass attenuation coefficient in cm^2/gr for an atomic number or symbol at specified energyEV values.

Parameters:
    z_or_symbol – Integer atomic number or symbol for elemen
    energyEV – energyEV (single value, list, array) in eV at which to calculate .
    kind – one of ‘total’ (default), ‘photo’, ‘coh’, and ‘incoh’ for total, photo-absorption, coherent scattering, and incoherent scattering cross sections, respectively.
"""

def generateEmittedSpectrum(kvp=100,filterMaterial='Al',filterThicknessMm=0.5,energyResolutionKeV=2.355):
    # Generate initial unfiltered spectrum
    s = spekpy.Spek(kvp=kvp,th=90.,dk=1.,z=0.1,targ='W',shift=0.5)
    # Filter the spectrum (unit is 'mm')
    s.filter('W',0.002) # self absorption by target
    s.filter('C',0.3) # diamond window
    if isinstance(filterMaterial,list): # for multiple filters
        if len(filterMaterial) != len(filterThicknessMm):
            raise Exception("provided a list of filter materials: %s\n but list of material thicknesses does not correspond: %s"%(filterMaterial,filterThicknessMm))
        for mc in range(len(filterMaterial)):
            s.filter(filterMaterial[mc],filterThicknessMm[mc]) # 0.5mm Al filter
    else:
        s.filter(filterMaterial,filterThicknessMm) # 0.5mm Al filter
    # Get energy and energy-fluence arrays (return values at bin-centres)
    energyKeV, spectrum = s.get_spectrum(edges=False,flu=False)
    # normalise the spectrum
    #spectrum /= np.sum(spectrum)
    # reduce energy resolution by blurring spectrum
    if energyResolutionKeV is not None:
        sigma = energyResolutionKeV / 2.355
        spectrum = sp.ndimage.gaussian_filter(spectrum,sigma)
    return energyKeV, spectrum

def getSpekpyMaterialList():
    # if want to see a list of available materials in spekpy
    return spekpy.Spek.show_matls()

def calcMassAttenuation(energyKeV, materialWeights, materialSymbols, kind='total'):
    # NOTE: kind can be: 'total' (default), 'photo', 'coh', 'incoh' for total,
    # photo-absorption, coherent scattering, and incoherent scattering cross
    # sections, respectively.
    M = len(materialWeights)
    if M < 1 or M != len(materialSymbols):
        raise Exception("Either no material weights, or not matching material symbols")
    energyEV = 1000. * energyKeV
    #normalise weights
    materialWeights /= np.sum(materialWeights)
    muPerCm = np.zeros(len(energyKeV),dtype=float)
    for mc in range(M):
        muPerCm += materialWeights[mc]*xray.mu_elam(materialSymbols[mc],energyEV)
    return muPerCm # list of mu factors per cm per density at different energies

def getMaterialAttenuation(energyKeV, materialWeights, materialSymbols, dens):
    sampleAttPerCm = dens * calcMassAttenuation(energyKeV, materialWeights, materialSymbols)
    return sampleAttPerCm

def calcTransmission(energyKeV, materialWeights, materialSymbols, density, thicknessCm, kind='total'):
    muPerCm = calcMassAttenuation(energyKeV, materialWeights, materialSymbols, kind)
    return np.exp(-muPerCm*density*thicknessCm)

def calcAbsorption(energyKeV, materialWeights, materialSymbols, density, thicknessCm):
    return 1.0-calcTransmission(energyKeV, materialWeights, materialSymbols, density, thicknessCm)

def calcPhotoTransmission(energyKeV, materialWeights, materialSymbols, density, thicknessCm):
    return calcTransmission(energyKeV, materialWeights, materialSymbols, density, thicknessCm, kind='photo')

def calcPhotoAbsorption(energyKeV, materialWeights, materialSymbols, density, thicknessCm):
    return 1.0-calcTransmission(energyKeV, materialWeights, materialSymbols, density, thicknessCm, kind='photo')

def calcIncohTransmission(energyKeV, materialWeights, materialSymbols, density, thicknessCm):
    return calcTransmission(energyKeV, materialWeights, materialSymbols, density, thicknessCm, kind='incoh')

def calcIncohScatter(energyKeV, materialWeights, materialSymbols, density, thicknessCm):
    return 1.0-calcTransmission(energyKeV, materialWeights, materialSymbols, density, thicknessCm, kind='incoh')

def calcCohTransmission(energyKeV, materialWeights, materialSymbols, density, thicknessCm):
    return calcTransmission(energyKeV, materialWeights, materialSymbols, density, thicknessCm, kind='coh')

def calcCohScatter(energyKeV, materialWeights, materialSymbols, density, thicknessCm):
    return 1.0-calcTransmission(energyKeV, materialWeights, materialSymbols, density, thicknessCm, kind='coh')

def attenuateSpectrum(energyKeV, spectrum, materialWeights, materialSymbols, density, thicknessCm):
    trans = calcTransmission(energyKeV, materialWeights, materialSymbols, density, thicknessCm)
    return spectrum*trans

def detectedSpectrum(energyKeV, spectrumIn, cameraLengthMm=300, scintType="CsI"):
    #filter by air, carbon fibre, and aluminum; then scale by abs(CsI) = 1-trans(CsI)
    ### filter by Air (cameraLengthMm)
    # Air, Dry (near sea level), Z/A = 0.49919, I(eV) = 85.7, dens = 1.205E-03
    # Z=6   : 0.000124
    # Z=7   : 0.755268
    # Z=8   : 0.231781
    # Z=18  : 0.012827
    materialWeights = [0.000124, 0.755268, 0.231781, 0.012827] # composition of air
    materialSymbols = ["C", "N", "O", "Ar"]
    dens = 1.225E-03 # 1.225 from the ISA
    thicknessCm = 0.1*cameraLengthMm
    spectrumOut = attenuateSpectrum(energyKeV, spectrumIn, materialWeights, materialSymbols, dens, thicknessCm)
    ### filter by Carbon Fibre (3.7mm)
    # (65%V, 70%W C p=1.7, 35%V 30%W C21H29O3 p=1.35)
    # Z=1	: 0.088722 
    # Z=6	: 0.765590
    # Z=8	: 0.145688
    materialWeights = [1.0]
    materialSymbols = ["C"]
    dens = 1.7
    thicknessCm = 0.4
    trans = calcTransmission(energyKeV, materialWeights, materialSymbols, dens, 0.65*thicknessCm)
    dens = 1.35
    materialWeights = [0.088722, 0.765590, 0.145688]
    materialSymbols = ["H","C","O"]
    trans *= calcTransmission(energyKeV, materialWeights, materialSymbols, dens, 0.35*thicknessCm)
    spectrumOut *= trans
    ### filter by Aluminium (0.1mm)
    materialWeights = [1.0]
    materialSymbols = ["Al"]
    dens = 2.7
    thicknessCm = 0.01
    spectrumOut = attenuateSpectrum(energyKeV, spectrumOut, materialWeights, materialSymbols, dens, thicknessCm)
    ### scale by abs(scint) = 1 - trans(scint)
    if "CsI" in scintType:
        ### CsI (0.7mm)
        # Z=53	: 0.488451
        # Z=55	: 0.511549
        materialWeights = [0.488451, 0.511549]
        materialSymbols = ["I", "Cs"]
        dens = 4.51
    elif "LuAG" in scintType:
        ### LuAG---Lu3Al5O12 (0.7mm)
        # Z=8	: 0.225396
        # Z=13	: 0.158379
        # Z=71	: 0.616225
        materialWeights = [0.225396, 0.158379, 0.616225]
        materialSymbols = ["O", "Al", "Lu"]
        dens = 6.67
    else:
        raise Exception("Unknown scintillator material specified: %s, (should be CsI, LuAG)"%scintType)
    thicknessCm = 0.07
    absorp = calcAbsorption(energyKeV, materialWeights, materialSymbols, dens, thicknessCm)
    spectrumOut *= absorp
    return spectrumOut



def plotSpectrum(energyKeV,spectrum,title=None):
    plt.plot(energyKeV,spectrum)
    plt.xlabel("energy (keV)")
    if title is not None:
        plt.title(title)
    plt.show()
    return

def plotSpectra(energyKeV,spectrumIn,spectrumOut,title=None):
    plt.plot(energyKeV,spectrumIn,label="Input Spectrum")
    plt.plot(energyKeV,spectrumOut,label="Detected Spectrum")
    plt.legend()
    plt.xlabel("energy (keV)")
    if title is not None:
        plt.title(title)
    plt.show()
    return

def estimateMeanEnergy(energyKeV,spectrum):
    return np.sum(spectrum*energyKeV)/np.sum(spectrum)


if __name__ == "__main__":

    energyKeV,spectrumIn = generateEmittedSpectrum(kvp=125.,filterThicknessMm=2.5)
    print("Mean energy of the emitted spectrum (keV): %s"%estimateMeanEnergy(energyKeV,spectrumIn))
    spectrumOut = detectedSpectrum(energyKeV,spectrumIn)
    plotSpectra(energyKeV,spectrumIn,spectrumOut)
    DEE = 100.0*np.sum(spectrumOut)/np.sum(spectrumIn)
    print("DEE = %s"%DEE)
    DQE = 100.0*np.sum(spectrumOut/energyKeV)/np.sum(spectrumIn/energyKeV)
    print("DQE = %s"%DQE)

    exit()

