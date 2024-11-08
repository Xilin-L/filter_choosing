import numpy as np
import matplotlib.pyplot as plt

import skimage as ski

import spekpy
import larch
from larch import xray

import xraySimulation as xs
import xrayImagingPerformance as xip


def generateCylinder(imgShape, vxSzMm=1.0, diameterMm=None, attPerCm=1.0):
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
    pathlengthCm = getProjectedThicknessCm(imgBin, vxSzMm=vxSzMm)
    pathlengthMm = 10.0 * pathlengthCm
    polyAtt = applyBeamHardening(pathlengthMm, A, n)
    return polyAtt


def setSpectrum(kvp, filterMaterial='Al', filterThicknessMm=0.5, plot=False):
    energyKeV, spectrumIn = xs.generateEmittedSpectrum(kvp, filterMaterial, filterThicknessMm)
    spectrumDet = xs.detectedSpectrum(energyKeV, spectrumIn)
    if plot is True:
        xs.plotSpectrum(energyKeV, spectrumDet, title="Spectrum at the detector")
    return energyKeV, spectrumDet


def setMaterial(materialName):
    if materialName.lower() == "al" or materialName.lower() == "aluminium":
        materialWeights = [1.0]
        materialSymbols = ["Al"]
        dens = 2.7
    elif materialName.lower() == "ti" or materialName.lower() == "titanium":
        materialWeights = [1.0]
        materialSymbols = ["Ti"]
        dens = 4.51
    elif materialName.lower() == "pmma" or materialName.lower() == "acrylic":
        materialWeights = [0.080541, 0.599846, 0.319613]
        materialSymbols = ["H", "C", "O"]
        dens = 1.18
    elif materialName.lower() == "sio2" or materialName.lower() == "glass":
        materialWeights, materialSymbols, dens = xip.getMaterialProperties("sandstone")
    elif materialName.lower() == "caco3" or materialName.lower() == "marble":
        materialWeights, materialSymbols, dens = xip.getMaterialProperties("carbonate")
    else:
        raise Exception(
            "unknown material name %s. Currently have in dictionary: Aluminium, Titanium, Acrylic, Glass, Marble." % (
                materialName))
    return materialWeights, materialSymbols, dens


def getMaterialAttenuation(energyKeV, materialWeights, materialSymbols, dens):
    sampleAttPerCm = dens * xs.calcMassAttenuation(energyKeV, materialWeights, materialSymbols)
    return sampleAttPerCm


if __name__ == "__main__":
    Ny = Nx = 128
    imgShape = (Ny, Nx)

    vxSzMm = 0.1
    sampleDiameterMm = 10.

    img = generateCylinder(imgShape, diameterMm=sampleDiameterMm, vxSzMm=vxSzMm)
    plt.imshow(img, "gray")
    plt.show()

    ### SIMPLE TEST ###
    sinogramIdeal = project(img, vxSzMm=vxSzMm)
    plt.imshow(sinogramIdeal, "gray")
    plt.show()

    recon = reconstruct(sinogramIdeal)
    plt.imshow(recon, "gray")
    plt.show()

    ### ADD SPECTRUM INFO ###

    # set spectrum
    energyKeV, spectrum = setSpectrum(kvp=60, filterMaterial='Al', filterThicknessMm=0.5)

    # set material to "sandstone"
    materialWeights, materialSymbols, dens = setMaterial("marble")

    # get material attenuation
    sampleAttPerCm = getMaterialAttenuation(energyKeV, materialWeights, materialSymbols, dens)

    # estimate beam hardening
    A, n = xip.estimateBeamHardening(spectrum, sampleAttPerCm, sampleDiameterMm, plot=True)
    print("BHC parms [A, n] = [%s, %s], bhcFactor = %s" % (A, n, 1. / n))

    # project with spectrum
    sinogram = projectSingleMaterialWithSpectrum(spectrum, sampleAttPerCm, img, angles=None, vxSzMm=vxSzMm)
    plt.imshow(sinogram, "gray")
    plt.show()

    # scale ideal sinogram to match
    meanAttenuation = np.mean(sinogram) / np.mean(sinogramIdeal)
    sinogramIdeal *= meanAttenuation

    # estiamate equivalent sinogram using model
    sinogramBH = projectWithBeamHardeningModel(img, A, n, vxSzMm=vxSzMm)

    plt.plot(sinogramIdeal[:, 0], label="monochrom.")
    plt.plot(sinogramBH[:, 0], label="beam hard.")
    plt.plot(sinogram[:, 0], label="polychrom.")
    plt.legend()
    plt.show()

    recon = reconstruct(sinogram)
    plt.imshow(recon, "gray")
    plt.show()

    plt.plot(recon[Ny // 2, :])
    plt.show()

    exit()