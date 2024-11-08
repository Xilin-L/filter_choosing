#!/usr/bin/env python3

import numpy as np

import scipy as sp
from scipy import ndimage
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

import spekpy

import larch
from larch import xray

import xraySimulation as xs

### What are the requirements for the ideal filter? ###
# 1 -X- rule of thumb (ROT), want filter to be equivalent to exp(-1) = 0.367 x diameter of sample * sample material * sample density. Maybe start here?
# 2 -X- want BHC power to be as close to 1.0 as possible. Ideally less than 1.1, can deal with less than 1.2...
# 3 --- want flux to be as close to 100% as possible. SNR decreases as 1/sqrt(flux). How does this relate to BH artefacts and scatter contribution? NEED TO THINK ABOUT THIS (RELATED TO 5 & 6?)
# 4 -X- want max attenuation to be around 1 [exp(-1) = 37%] but could go down to 2 [exp(-2) = 13.5%]; absolute limit is 3 [exp(-3) = 5%]
# 5 --- how does flux change with kV? Assume constant power? NEED TO MEASURE THIS
# 6 -X- include some sort of scatter estimate? use larch %flux attenuated by pe vs cs/ts? Want scatter < 2% (at least scatter < half transmisson?)


def estimateMeasuredScatterAsPercentOfFlux(energyKeV,spectrum,sampleMaterial,sampleDiameterMm,coneAngleDeg=60.):
    # estimate fraction of spectrum flux is detected as scatter
    # get material properties
    thicknessAvgCm = 0.075*sampleDiameterMm
    materialWeights,materialSymbols,dens = getMaterialProperties(sampleMaterial)
    # estiamte amount of total scatter
    # Start with Compton ('absorption' component)
    sampIncohScat = xs.calcIncohScatter(energyKeV,materialWeights,materialSymbols,dens,thicknessAvgCm)
    spectrumScat = spectrum*sampIncohScat
    # Plus Raleigh ('absorption' component after transmission from Compton)
    sampIncohTrans = xs.calcIncohTransmission(energyKeV,materialWeights,materialSymbols,dens,thicknessAvgCm)
    sampCohScat = xs.calcCohScatter(energyKeV,materialWeights,materialSymbols,dens,thicknessAvgCm)
    spectrumScat += spectrum*(sampIncohTrans)*sampCohScat
    # attenuate total scatter by photo-electric half(thicknessAvg)
    sampPhotoTrans = xs.calcPhotoTransmission(energyKeV,materialWeights,materialSymbols,dens,0.5*thicknessAvgCm)
    spectrumScat *= sampPhotoTrans
    # total scatter energy is:
    scatEnergyFluence = np.sum(spectrumScat) 
    # assume scatter in all directions, so fraction detected is:
    scatFracDet = (coneAngleDeg**2)/(360.0**2)
    # '''
    # change spectrum to spectrum_trans as we are comparing signal measured to noise in return
    # '''
    # sample_trans = xs.calcTransmission(energyKeV, materialWeights, materialSymbols, dens, thicknessAvgCm)
    # spectrum_trans = spectrum*sample_trans
    return 100.0*scatEnergyFluence*scatFracDet/np.sum(spectrum)


def generateBeamHardeningCurve(spectrum,sampleAttPerCm,sampleDiameterMm):
    #bhcData[:,0] is thickness of material in mm
    #bhcData[:,1] is corresponding measured transmission
    tStepMm = 0.1
    Nt = int(10.0*sampleDiameterMm/tStepMm)
    bhcData = np.zeros((Nt,2),dtype=float)
    for tc in range(Nt):
        tCurrCm = 0.1*(tc+1)*tStepMm
        sampleTransCurr = np.exp(-sampleAttPerCm*tCurrCm)
        spectrumCurr = spectrum*sampleTransCurr
        measTransCurr = np.sum(spectrumCurr)/np.sum(spectrum)
        bhcData[tc,0] = 10.0*tCurrCm
        bhcData[tc,1] = measTransCurr
    return bhcData


def func_powerlaw(x, A, n):
    return A*(x**n)


def fitPowerLawToBhcData(bhcData):
    #bhcData[:,0] is thickness of material in mm
    #bhcData[:,1] is corresponding measured transmission
    x = bhcData[:,0]
    y = -np.log(bhcData[:,1])
    popt, pcov = curve_fit(func_powerlaw, x, y)
    A = popt[0]
    n = popt[1]
    return A,n


def estimateBeamHardening(spectrum,sampleAttPerCm,sampleDiameterMm,plot=False):
    # return parameters A,n that fit the beam hardening curve y as y=Ax^n
    # where x is the material thickness
    #
    #bhcData[:,0] is thickness of material in mm
    #bhcData[:,1] is corresponding measured transmission
    bhcData = generateBeamHardeningCurve(spectrum,sampleAttPerCm,sampleDiameterMm)
    A,n = fitPowerLawToBhcData(bhcData)
    if plot is True:
        plt.plot(bhcData[:,0],-np.log(bhcData[:,1]),label="data")
        plt.plot(bhcData[:,0],A*(bhcData[:,0]**n),label="fit")
        plt.legend()
        plt.show()
    return A,n



def getMaterialProperties(material):
    """
    Retrieves material properties including elemental weight fractions, element symbols, and density.
    Supports both single materials and composite materials with strict, case-sensitive material names.

    Parameters:
    - material (str or list): 
        - If str: Name of the material (e.g., "sandstone").
        - If list: 
            - First element: List of material names (exact case).
            - Second element: List of corresponding percentages (should sum to 1).

    Returns:
    - materialWeights (list): Combined weight fractions of elements.
    - materialSymbols (list): Corresponding element symbols.
    - dens (float): Combined density of the material.

    Raises:
    - Exception: If an unknown material is provided, input format is incorrect, or percentages are invalid.
    """

    def get_single_material_properties(mat):
        """
        Helper function to get properties of a single material with case-sensitive matching.
        this part is the old getMaterialProperty() function
        """
        if mat.lower() == "sandstone" or mat.lower() == "clastic" or mat.lower() == "glass":
            # Z=8	: 0.532565
            # Z=14	: 0.467435
            materialWeights = [0.532565, 0.467435]
            materialSymbols = ["O", "Si"]
            dens = 2.1
        elif mat.lower() == "limestone" or mat.lower() == "carbonate" or mat.lower() == "marble" or mat.lower() == "caco3":
            # Z=6	: 0.120005
            # Z=8	: 0.479564
            # Z=20	: 0.400431
            materialWeights = [0.120005, 0.479564, 0.400431]
            materialSymbols = ["C", "O", "Ca"]
            dens = 2.65
        elif mat.lower() == "haematite":
            # Z=8	: 0.300567
            # Z=26	: 0.699433
            materialWeights = [0.300567, 0.699433]
            materialSymbols = ["O", "Fe"]
            dens = 5.3
        elif mat.lower() == "goethite":
            # Z=1	: 0.011344
            # Z=8	: 0.360129
            # Z=26	: 0.628527
            materialWeights = [0.011344, 0.360129, 0.628527]
            materialSymbols = ["H", "O", "Fe"]
            dens = 3.8  # 3.3 - 4.3
        elif mat.lower() == "iron ore":
            densSum = 5.3 + 3.8
            materialWeights = (5.3 * np.array([0., 0.300567, 0.699433]) + 3.8 * np.array(
                [0.011344, 0.360129, 0.628527])) / densSum
            materialSymbols = ["H", "O", "Fe"]
            dens = 0.5 * densSum
        elif mat.lower() == "feo" or mat.lower() == "wustite":
            materialWeights = [0.223, 0.777]
            materialSymbols = ["O", "Fe"]
            dens = 5.745
        elif mat.lower() == "peek":
            materialWeights = [0.041948, 0.791569, 0.166483]
            materialSymbols = ["H", "C", "O"]
            dens = 1.32
        elif mat.lower() == "al":
            materialWeights = [1.0]
            materialSymbols = ["Al"]
            dens = 2.70
        elif mat.lower() == "xe":
            materialWeights = [1.0]
            materialSymbols = ["Xe"]
            dens = 0.00589
        elif mat.lower() == "ti64":  # the titanium alloy with 6% Al and 4% V by weight
            materialWeights = [0.06, 0.90, 0.04]
            materialSymbols = ["Al", "Ti", "V"]
            dens = 4.43
        elif mat.lower() == "hardwood":  # assuming not dried
            materialWeights = [0.06, 0.52, 0.42]
            materialSymbols = ["H", "C", "O"]
            dens = 0.85  # Jarrah wood
        elif mat.lower() == "softwood":  # assuming not dried
            materialWeights = [0.06, 0.52, 0.42]
            materialSymbols = ["H", "C", "O"]
            dens = 0.50  # Pine
        elif mat.lower() == "ti" or mat.lower() == "titanium":
            materialWeights = [1.0]
            materialSymbols = ["Ti"]
            dens = 4.51
        elif mat.lower() == "pmma" or mat.lower() == "acrylic":
            materialWeights = [0.080541, 0.599846, 0.319613]
            materialSymbols = ["H", "C", "O"]
            dens = 1.18
        else:
            raise Exception(f"Unknown sample material: '{mat}'. Valid materials are: 'sandstone', 'clastic', "
                            f"'limestone', 'carbonate', 'haematite', 'goethite', 'iron ore', 'PEEK', 'Al', 'Xe',"
                            f" 'Ti64', 'wood', 'titanium', 'glass', 'acrylic', 'feo', 'wustite', 'hardwood', 'softwood'.")
        return materialWeights, materialSymbols, dens

    if isinstance(material, str):
        # Single material
        materialWeights, materialSymbols, dens = get_single_material_properties(material)
        return materialWeights, materialSymbols, dens

    elif isinstance(material, list):
        if len(material) != 2:
            raise Exception("Composite material input must be a list of two lists: [materials, percentages].")

        materials_list, percentages_list = material

        if not (isinstance(materials_list, list) and isinstance(percentages_list, list)):
            raise Exception("Composite material input must be a list of two lists: [materials, percentages].")

        if len(materials_list) != len(percentages_list):
            raise Exception("Materials list and percentages list must have the same length.")

        total_percentage = sum(percentages_list)
        if not np.isclose(total_percentage, 1.0):
            raise Exception(f"Percentages must sum to 1.0, but sum to {total_percentage}.")

        # Initialize dictionaries to accumulate element weights
        combined_elements = {}
        combined_density = 0.0

        for mat, perc in zip(materials_list, percentages_list):
            mat_weights, mat_symbols, mat_dens = get_single_material_properties(mat)
            
            # Accumulate density as weighted average
            combined_density += perc * mat_dens

            # Accumulate element weights
            for w, sym in zip(mat_weights, mat_symbols):
                if sym in combined_elements:
                    combined_elements[sym] += perc * w
                else:
                    combined_elements[sym] = perc * w

        # Normalize the combined element weights to sum to 1
        total_element_weight = sum(combined_elements.values())
        if total_element_weight == 0:
            raise Exception("Total element weight is zero. Check material weights and percentages.")
        for sym in combined_elements:
            combined_elements[sym] /= total_element_weight

        # Sort elements alphabetically for consistency
        sorted_elements = sorted(combined_elements.items())
        materialSymbols = [sym for sym, _ in sorted_elements]
        materialWeights = [w for _, w in sorted_elements]

        return materialWeights, materialSymbols, combined_density

    else:
        raise Exception("Input material must be either a string or a list of two lists [materials, percentages].")



def getFilterAttPerCm(energyKeV,filterMaterial="Cu"): # added Cu as an option
    if "Al" in filterMaterial:
        filtAttPerCm = 2.7*xs.calcMassAttenuation(energyKeV,[1.0],["Al"])
    elif "Fe" in filterMaterial:
        filtAttPerCm = 7.85*xs.calcMassAttenuation(energyKeV,[1.0],["Fe"])
    elif "Cu" in filterMaterial:
        filtAttPerCm = 8.94*xs.calcMassAttenuation(energyKeV,[1.0],["Cu"])
    elif "304sstl" in filterMaterial:
        filtAttPerCm = 7.93*xs.calcMassAttenuation(energyKeV,[0.74, 0.18, 0.08 ],["Fe", "Cr", "Ni"])
    else:
        raise Exception("unknown filter material specified: %s (should be Al, Cu, Fe or 304sstl)"%filterMaterial)
    return filtAttPerCm


def getFilterTransmission(energyKeV, filterMaterial="Al", filterThicknessMm=0.5):
    filtAttPerCm = getFilterAttPerCm(energyKeV,filterMaterial)
    tFiltCm = 0.1*filterThicknessMm
    filtTrans = np.exp(-filtAttPerCm*tFiltCm)
    return filtTrans


def getFilteredSpectrum(energyKeV, spectrumIn, filterMaterial="Al", filterThicknessMm=0.5):
    filtTrans = getFilterTransmission(energyKeV,filterMaterial,filterThicknessMm)
    return spectrumIn*filtTrans


def getRuleOfThumbFilterThickness(kVpeak,filterMaterial="Al", sampleMaterial="sandstone", sampleDiameterMm=25.0, 
                                  max_tFiltMm = 9.0):
    # rule of thumb for filter thickness is that which gives equivalent attenuation as D/e thickness of the sample
    # where D is sampleDiameter and e is 2.7183
    energyKeV,spectrumIn = xs.generateEmittedSpectrum(kVpeak)
    tSampFiltCm = 0.1*np.exp(-1.)*sampleDiameterMm # ROT filter attenuation 
    materialWeights,materialSymbols,dens = getMaterialProperties(sampleMaterial)
    sampFiltTrans = xs.calcTransmission(energyKeV, materialWeights, materialSymbols, dens, tSampFiltCm)
    spectrumOut = spectrumIn*sampFiltTrans
    measTransSampFilt = np.sum(spectrumOut)/np.sum(spectrumIn)

    filtAttPerCm = getFilterAttPerCm(energyKeV,filterMaterial)
    if "Al" in filterMaterial:
        stepSzCm = 0.05
    elif "Fe" in filterMaterial:
        stepSzCm = 0.01
    elif "Cu" in filterMaterial:
        stepSzCm = 0.01
    elif "304sstl" in filterMaterial:
        filtAttPerCm = 7.93*xs.calcMassAttenuation(energyKeV,[0.74, 0.18, 0.08 ],["Fe", "Cr", "Ni"])
        stepSzCm = 0.01
    else:
        raise Exception("unknown filter material specified: %s (should be Al, Cu, Fe or 304sstl)"%filterMaterial)
    tFiltCm = 0.0
    max_tFiltCm = 10.0*max_tFiltMm # max filter thickness is 9mm by default
    measTransFilt = 1.0
    while measTransFilt>measTransSampFilt and tFiltCm<max_tFiltCm:
        tFiltCm += stepSzCm
        filtTrans = np.exp(-filtAttPerCm*tFiltCm)
        spectrumOut = spectrumIn*filtTrans
        measTransFilt = np.sum(spectrumOut)/np.sum(spectrumIn)
    return 10.0*tFiltCm


def getImagingStatistics(energyKeV, spectrumIn, filterMaterial="Al", filterThicknessMm=2.0,
                         sampleMaterial="sandstone",sampleDiameterMm=15.,coneAngleDeg=50.):
    # get filtered spectrum
    spectrumFilt = getFilteredSpectrum(energyKeV, spectrumIn, filterMaterial, filterThicknessMm)
    # estimate fraction of original flux incident on sample [return 1]
    fluxPerc = 100.0*np.sum(spectrumFilt)/np.sum(spectrumIn)
    # incorporate attenuation by sample
    materialWeights,materialSymbols,dens = getMaterialProperties(sampleMaterial)
    tSampCm = 0.1*sampleDiameterMm
    sampTrans = xs.calcTransmission(energyKeV, materialWeights, materialSymbols, dens, tSampCm)
    spectrumOut = spectrumFilt*sampTrans
    # estimate SNR [return 2]
    # snr = estimateSnr(energyKeV,spectrumOut)
    snr = SnrBySummingFlux(energyKeV, spectrumFilt, sampleMaterial, sampleDiameterMm, coneAngleDeg)
    # estimate sample transmission/attenuation
    measTransSamp = np.sum(spectrumOut)/np.sum(spectrumFilt)
    sampTPerc = 100.0*measTransSamp # [return 2]
    sampA = -np.log(measTransSamp) # [return 3]
    # estimate BHC factor
    sampAttPerCm = -np.log(sampTrans)/tSampCm
    A,n = estimateBeamHardening(spectrumFilt,sampAttPerCm,sampleDiameterMm)
    bhcFactor = 1.0/n # [return 4]
    # estimate scatter from sample that hits detector [return 5]
    
    scatContribPerc = estimateMeasuredScatterAsPercentOfFlux(energyKeV,spectrumFilt,sampleMaterial,sampleDiameterMm,coneAngleDeg)
    return fluxPerc,snr,sampTPerc,sampA,bhcFactor,scatContribPerc


def estimateSnr(energyKeV,spectrum,scale=0.00030435,gain=0.003):
    # get mean energy of spectrum
    meanEnergy = xs.estimateMeanEnergy(energyKeV,spectrum)
    # estimate #photons as (flux/gain)/meanEnergy
    flux = scale * np.sqrt(np.sum(spectrum))
    numPhotons = (flux / gain) / meanEnergy
    return np.sqrt(numPhotons)

def SnrBySummingFlux(energyKeV,spectrum,sampleMaterial,sampleDiameterMm,coneAngleDeg=60.):
    '''
    similar to the estimateMeasuredScatterAsPercentOfFlux function but use
    SNR = sum(spectrum_transmitted * detector efficiency)/
          sqrt(sum(spectrum_scattered * detector efficiency))
    '''
    # get material properties
    thicknessAvgCm = 0.075*sampleDiameterMm
    materialWeights,materialSymbols,dens = getMaterialProperties(sampleMaterial)
    # calculate signal
    sample_trans = xs.calcTransmission(energyKeV, materialWeights, materialSymbols, dens, 0.1*sampleDiameterMm)
    spectrum_trans = spectrum*sample_trans
    signal = np.sum(xs.detectedSpectrum(energyKeV,spectrum_trans))

    # estiamte amount of total scatter
    # Start with Compton ('absorption' component)
    sampIncohScat = xs.calcIncohScatter(energyKeV,materialWeights,materialSymbols,dens,thicknessAvgCm)
    spectrumScat = spectrum*sampIncohScat
    # Plus Raleigh ('absorption' component after transmission from Compton)
    sampIncohTrans = xs.calcIncohTransmission(energyKeV,materialWeights,materialSymbols,dens,thicknessAvgCm)
    sampCohScat = xs.calcCohScatter(energyKeV,materialWeights,materialSymbols,dens,thicknessAvgCm)
    spectrumScat += spectrum*(sampIncohTrans)*sampCohScat
    # attenuate total scatter by photo-electric half(thicknessAvg)
    sampPhotoTrans = xs.calcPhotoTransmission(energyKeV,materialWeights,materialSymbols,dens,0.5*thicknessAvgCm)
    spectrumScat *= sampPhotoTrans
    # total scatter that detectable is:
    noise_total = np.sum(xs.detectedSpectrum(energyKeV, spectrumScat))
    # assume scatter in all directions, so fraction detected is:
    scatFracDet = (coneAngleDeg**2)/(360.0**2)
    noise_detected = np.sqrt(noise_total*scatFracDet+signal)

    return signal/(noise_detected)
    


def estimateSnrThroughSample(energyKeV, spectrumIn, filterMaterial="Al", filterThicknessMm=2.0, 
                             sampleMaterial="sandstone",sampleDiameterMm=15.):
    # estimate SNR in darkest region of radiograph as #photons / sqrt(#photons) = sqrt(#photons)
    # get filtered spectrum
    spectrumFilt = getFilteredSpectrum(energyKeV, spectrumIn, filterMaterial, filterThicknessMm)
    # estimate sample transmission/attenuation
    materialWeights,materialSymbols,dens = getMaterialProperties(sampleMaterial)
    tSampCm = 0.1*sampleDiameterMm
    sampTrans = xs.calcTransmission(energyKeV, materialWeights, materialSymbols, dens, tSampCm)
    spectrumOut = spectrumFilt*sampTrans
    snr = estimateSnr(energyKeV,spectrumOut)
    return snr


def getRotFilterImageStatsPerkVp(filterMaterial,sampleMaterial="sandstone",sampleDiameterMm=25.,kVPeakMin=160.,kVPeakMax=300.,coneAngleDeg=50.):
    #ROT = rule-of-thumb
    energy = []
    tFiltMm = []
    flux = []
    snr = []
    sampT = []
    sampA = []
    bhcFactor = []
    scatContrib = []
    for kVpeak in np.arange(kVPeakMin,kVPeakMax,10):
        energy.append(kVpeak)
        t = getRuleOfThumbFilterThickness(kVpeak,filterMaterial,sampleMaterial,sampleDiameterMm)
        tFiltMm.append(t)
        energyKeV,spectrumIn = xs.generateEmittedSpectrum(kVpeak)
        imgStat = getImagingStatistics(energyKeV,spectrumIn,filterMaterial,t,sampleMaterial,sampleDiameterMm,coneAngleDeg)
        flux.append(imgStat[0])
        snr.append(imgStat[1])
        sampT.append(imgStat[2])
        sampA.append(imgStat[3])
        bhcFactor.append(imgStat[4])
        scatContrib.append(imgStat[5])
        print("kVp: %3d | t (mm): %4.1f | flux (%%): %4.1f | SNR: %4.1f | sample Trans (%%): %4.1f | sample Att: %4.1f | BHC factor: %4.2f | scat contrib. (%%): %8.6f"%(energy[-1],tFiltMm[-1],flux[-1],snr[-1],sampT[-1],sampA[-1],bhcFactor[-1],scatContrib[-1]))

    return


def EquivFilterThicknessTable(sample_material = "sandstone", sample_diameter_mm = 10):
    kVp_min = 60.0
    kVp_max = 310.0
    max_tFiltMm=1000.0
    for kVp in np.arange(kVp_min, kVp_max, 10.0 ):
        al_thickness = getRuleOfThumbFilterThickness(kVp, "Al", sample_material, sample_diameter_mm, max_tFiltMm)
        cu_thickness = getRuleOfThumbFilterThickness(kVp, "Cu", sample_material, sample_diameter_mm, max_tFiltMm)
        al_cu_ratio = al_thickness/cu_thickness
        print(f"kVp:{kVp} | Al thickness: {al_thickness:.2f} mm | Cu thickness: {cu_thickness:.2f} mm | Al/Cu ratio: {al_cu_ratio:.2f}")
    
    return


def statsDifFiltThickness(kvp, sampleMaterial='FeO', sampleThicknessMm=12.0,
                          filterMaterial='304sstl', filterThicknessMmMin=0.5,
                          filterThicknessMmMax=4.2, filterThicknessMmStep=0.2, plot=True):
    filterThicknesses = []
    bhcFactors = []
    flux = []
    sampT = []
    for filterThicknessMm in np.arange(filterThicknessMmMin, filterThicknessMmMax, filterThicknessMmStep):
        energyKeV, spectrumIn = xs.generateEmittedSpectrum(kvp, filterThicknessMm=0.0)
        imgStat = getImagingStatistics(energyKeV, spectrumIn, filterMaterial, filterThicknessMm, sampleMaterial,
                                       sampleThicknessMm)
        filterThicknesses.append(filterThicknessMm)
        flux.append(imgStat[0])
        sampT.append(imgStat[2])
        bhcFactors.append(imgStat[4])  # BHC factor is the 5th element in the returned tuple
        print(
            f"Filter Thickness: {filterThicknessMm:.2f} mm | Flux: {imgStat[0]:.2f}% "
            # f"| SNR: {imgStat[1]:.2f} "
            f"| Sample Transmission: {imgStat[2]:.2f}% | Sample Attenuation: {imgStat[3]:.2f} "
            f"| BHC Factor: {imgStat[4]:.2f} | Scatter Contribution: {imgStat[5]:.6f}%")
    if plot:
        plt.figure()
        plt.plot(filterThicknesses, bhcFactors, marker='o', label='BHC Factor')
        plt.xlabel('Filter Thickness (mm)')
        plt.ylabel('BHC Factor')
        plt.title(f'BHC Factor vs Filter Thickness, kVp={kvp}')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(filterThicknesses, flux, marker='x', label='Flux (%)')
        plt.xlabel('Filter Thickness (mm)')
        plt.ylabel('Flux (%)')
        plt.title(f'Flux vs Filter Thickness, kVp={kvp}')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(filterThicknesses, sampT, marker='s', label='Sample Transmission (%)')
        plt.xlabel('Filter Thickness (mm)')
        plt.ylabel('Sample Transmission (%)')
        plt.title(f'Sample Transmission vs Filter Thickness, kVp={kvp}')
        plt.legend()
        plt.grid(True)
        plt.show()


# if __name__ == "__main__":

#     sampleMaterial="iron ore"
#     sampleDiameterMm = 10.

#     kVpeak = 140.
#     # given kVpeak determine ROT filter thickness for Al and Fe
#     # Aluminium filter
#     t = getRuleOfThumbFilterThickness(kVpeak,"Al",sampleMaterial,sampleDiameterMm)
#     print("ruleOfThumb %s filter thickness: %s"%("Al",t))
#     # Steel (iron) filter
#     t = getRuleOfThumbFilterThickness(kVpeak,"Cu",sampleMaterial,sampleDiameterMm)
#     print("ruleOfThumb %s filter thickness: %s"%("Cu",t))


#     # get ROT thickness per kVpeak for Fe filter
#     getRotFilterImageStatsPerkVp("Fe",sampleMaterial,sampleDiameterMm)

#     exit()
