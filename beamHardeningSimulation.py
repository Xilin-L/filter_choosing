import numpy as np
import matplotlib.pyplot as plt

import skimage as ski

import spekpy
import larch
from larch import xray
from skimage.transform import radon

import xraySimulation as xs
import xrayImagingPerformance as xip


def generateCylinder(imgShape, vxSzMm=1.0, diameterMm=None, attPerCm=1.0):
    # actually a circle
    Ny, Nx = imgShape
    DyMm = Ny * vxSzMm
    DxMm = Nx * vxSzMm
    if diameterMm is None:
        diameterMm = (8 * np.minimum(DyMm, DxMm)) // 10
    radiusVx = 0.5 * diameterMm / vxSzMm
    y = (np.arange(Ny) - Ny // 2).reshape((Ny, 1))
    x = (np.arange(Nx) - Nx // 2).reshape((1, Nx))
    r = np.sqrt(x * x + y * y)
    img = (r < radiusVx).astype(np.float32)
    return attPerCm * img


def project(img, angles=None, vxSzMm=1.0):
    Ny, Nx = img.shape
    if angles is None:
        Na = int(0.5 * np.pi * np.maximum(Ny, Nx))
        angles = np.arange(Na) * 180.0 / Na
    # Assume img is in att/cm
    vxSzCm = 0.1 * vxSzMm
    attPerVx = img * vxSzCm
    sinogram = ski.transform.radon(attPerVx, angles)
    return sinogram


def enforceBinaryImage(imgIn):
    # Assum that img is a binary image where "0" is air and "1" is the material
    # enforce img to be in [0.,1.]
    imgBin = imgIn - np.min(imgIn)
    imgBin /= np.max(imgBin)
    return imgBin


def getProjectedThicknessCm(img, angles=None, vxSzMm=1.0):
    Ny, Nx = img.shape
    if angles is None:
        Na = int(0.5 * np.pi * np.maximum(Ny, Nx))
        angles = np.arange(Na) * 180.0 / Na
    # Assume that img is a binary image where "0" is air and "1" is the material
    imgBin = enforceBinaryImage(img)
    vxSzCm = 0.1 * vxSzMm
    pathlengthPerVx = imgBin * vxSzCm
    projectedThicknessCm = ski.transform.radon(pathlengthPerVx, angles)
    return projectedThicknessCm


def reconstruct(sinogram, angles=None, vxSzMm=1.0):
    Nw, Na = sinogram.shape
    if angles is None:
        angles = np.arange(Na) * 180.0 / Na
    attPerVx = ski.transform.iradon(sinogram, angles, filter_name="ramp")
    # Output img in att/cm
    vxSzCm = 0.1 * vxSzMm
    img = attPerVx / vxSzCm
    return img


def projectSingleMaterialWithSpectrum(spectrum, attenuationPerCm, imgIn, angles=None, vxSzMm=1.0):
    # return projected attenuation = -log( sum_E spectrum(E) exp[-mu(E).pathlength] )
    # where mu = attenuationPerCm

    # Assume that img is a binary image where "0" is air and "1" is the material
    imgBin = enforceBinaryImage(imgIn)

    # get projected thickness (or pathlength)
    pathlengthCm = getProjectedThicknessCm(imgBin, vxSzMm=vxSzMm)

    # normalise sepctrum
    s = spectrum / np.sum(spectrum)

    # accumulate transmission over each energy in spectrum
    trans = np.zeros_like(pathlengthCm)
    for ec in range(len(spectrum)):
        trans += s[ec] * np.exp(-attenuationPerCm[ec] * pathlengthCm)

    # convert to measured attenuation
    sinogram = -np.log(trans)
    return sinogram


def applyBeamHardeningCorrection(attenuation, A, n):
    # using BH model: mu_poly = A.mu_mono^n
    # so mu_mono = (mu_poly/A)^(1/n)
    return np.power((attenuation / A), 1. / n)


def applyBeamHardening(pathlengthCm, A, n):
    # using BH model: mu_poly = A.mu_mono^n
    return A * np.power(pathlengthCm, n)


def projectWithBeamHardeningModel(imgIn, A, n, angles=None, vxSzMm=1.0):
    # using BH model: mu_poly = A.mu_mono^n

    # Assume that img is a binary image where "0" is air and "1" is the material
    imgBin = enforceBinaryImage(imgIn)

    # get projected thickness (or pathlength)
    pathlengthCm = getProjectedThicknessCm(imgBin, angles=angles, vxSzMm=vxSzMm)
    pathlengthMm = 10.0 * pathlengthCm
    polyAtt = applyBeamHardening(pathlengthMm, A, n)
    return polyAtt


def simulateBH(sampleDiameterMm=4., sampleDiameterVx=100, kvp=60, filterMaterial='Al',
                  filterThicknessMm=0.5, materialName='marble', plotIdeal=False,plotBH=False, plotCurve=True,verbose=True):
    '''
    similate the beam hardening effect for a disk.
    sampleDiameterVx: diameter of the sample in voxels
    use the ratio of sampleDiameterMm/sampleDiameterVx to determine the resolution of the image
    '''
    Ny = int(sampleDiameterVx*1.2)
    Nx = int(sampleDiameterVx*1.2)
    vxSzMm = sampleDiameterMm/sampleDiameterVx
    imgShape = (Ny, Nx)
    img = generateCylinder(imgShape, diameterMm=sampleDiameterMm, vxSzMm=vxSzMm)

    ### SIMPLE TEST ###
    sinogramIdeal = project(img, vxSzMm=vxSzMm)

    if plotIdeal:
        plt.imshow(img, "gray") # plot the truth at given resolution
        plt.show()
        plt.imshow(sinogramIdeal, "gray") # plot the ideal sinogram (ideal -> defect-free)
        plt.show()

    ### ADD SPECTRUM INFO ###
    energyKeV, spectrum = xs.setSpectrum(kvp=kvp, filterMaterial=filterMaterial, filterThicknessMm=filterThicknessMm)
    # energyKeV, spectrum = xs.generateEmittedSpectrum(kvp=kvp, filterMaterial=filterMaterial, filterThicknessMm=filterThicknessMm)

    materialWeights, materialSymbols, dens = xip.getMaterialProperties(materialName)
    sampleAttPerCm = xs.getMaterialAttenuation(energyKeV, materialWeights, materialSymbols, dens)
    A, n = xip.estimateBeamHardening(spectrum, sampleAttPerCm, sampleDiameterMm, plot=plotCurve)
    if verbose:
        print("BHC parms [A, n] = [%s, %s], bhcFactor = %s" % (A, n, 1. / n))

    sinogramFP = projectSingleMaterialWithSpectrum(spectrum, sampleAttPerCm, img, angles=None, vxSzMm=vxSzMm)
    # full physics sinogram
    meanAttenuation = np.mean(sinogramFP) / np.mean(sinogramIdeal)
    sinogramIdeal *= meanAttenuation

    sinogramBH = projectWithBeamHardeningModel(img, A, n, vxSzMm=vxSzMm)
    # just applying the simple rule, not full physics

    tomo = reconstruct(sinogramFP)
    if plotBH: # plot the beam-hardened sinogram and reconstruction
        plt.imshow(sinogramFP, "gray")
        plt.show()
        plt.imshow(tomo, "gray")
        plt.show()

    if plotCurve:
        plt.plot(sinogramIdeal[:, 0], label="monochrom.")
        plt.plot(sinogramBH[:, 0], label="beam hard.")
        plt.plot(sinogramFP[:, 0], label="polychrom.")
        plt.grid()
        plt.legend()
        plt.show()
        plt.plot(tomo[Ny // 2, :])
        plt.show()

    return 1./n, vxSzMm, tomo, sinogramFP



def idealiseTomo(tomo, threshold=0.4):
    # idealise the binarised tomogram
    return (tomo > threshold).astype(np.float32)


def estimateBhcFromTomo(tomo, vxSzMm=0.1, plot=False, verbose=True):
    # Estimate the beam hardening coefficient from a tomogram
    tomo[tomo < 0] = 0

    projIdeal = np.sum(idealiseTomo(enforceBinaryImage(tomo)), axis=0)
    projIdeal *= vxSzMm # pathlength, normalise to mm

    projBH = np.sum(tomo, axis=0)

    stackedData = np.stack((projIdeal, np.exp(-projBH)), axis=1)

    A, n = xip.fitPowerLawToBhcData(stackedData)

    if plot:
        x = np.arange(100) * np.max(projIdeal) / 100 # in 0.1mm
        bhc=1/n
        plt.plot(x, A * x ** n, label="fit bhc = %s" % bhc)
        plt.plot(stackedData[:,0], -np.log(stackedData[:, 1]), "x", label="BH curve tomo")
        plt.xlabel("pathlength (mm)")
        plt.legend()
        plt.grid()
        plt.show()
    if verbose:
        bhc=1./n
        print("bhc tomo = %s" % bhc)
    return 1./n




if __name__ == "__main__":

    '''
    modified the simulateBH function so the Nx and Ny are determined by sampleDiameter in terms of voxels.
    difference between points with same depth get smaller as sampleDiameterVx increases.

    discardFirst: not very accurate compared to the original
    '''

    kvp = 180
    materialName = 'peek'


    ### test the accuracy for a single case ###

    bhc, vxSzMm, tomo,_= simulateBH(sampleDiameterMm=2, sampleDiameterVx=200, kvp=kvp, materialName=materialName,
                                    plotIdeal=False, plotBH=False, plotCurve=False)
    bhsEsti=estimateBhcFromTomo(tomo, vxSzMm=vxSzMm, plot=True, verbose=True)


    ### generate the estimation curves ###

    # bhcList=[]
    # bhcEstiList=[]
    # sampleDiaList = np.arange(3, 16, 1)
    # sampleVxList = np.arange(10, 200, 20)
    # for sampleDia in sampleDiaList:
    #     for sampleVx in sampleVxList:
    #         bhc, vxSzMm, tomo, _ = simulateBH(sampleDiameterMm=sampleDia, sampleDiameterVx=sampleVx, kvp=kvp,
    #                                           materialName=materialName,
    #                                           plotIdeal=False, plotBH=False, plotCurve=False)
    #         bhsEsti = estimateBhcFromTomo(tomo, vxSzMm=vxSzMm, plot=False, verbose=False)
    #         bhcList.append(bhc)
    #         bhcEstiList.append(bhsEsti)
    #
    # plt.plot(sampleDiaList, bhcList[sampleVxList.tolist().index(10)::len(sampleVxList)],
    #              label=f"simulated")
    # for sampleVx in sampleVxList:
    #     plt.plot(sampleDiaList, bhcEstiList[sampleVxList.tolist().index(sampleVx)::len(sampleVxList)],
    #              label=f"estimated (Vx={sampleVx})")
    #
    # plt.grid()
    # plt.legend()
    # plt.title("BHC estimation, kVp= %s, material= %s" % (kvp, materialName))
    # plt.show()


    exit()
