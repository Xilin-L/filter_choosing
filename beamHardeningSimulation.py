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


def simulateBH(Ny=128, Nx=128, vxSzMm=0.1, sampleDiameterMm=4., kvp=60, filterMaterial='Al',
                  filterThicknessMm=0.5, materialName='marble', plotIdeal=False,plotBH=False, plotCurve=True,verbose=True):

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
    materialWeights, materialSymbols, dens = xip.getMaterialProperties(materialName)
    sampleAttPerCm = xs.getMaterialAttenuation(energyKeV, materialWeights, materialSymbols, dens)
    A, n = xip.estimateBeamHardening(spectrum, sampleAttPerCm, sampleDiameterMm, plot=plotCurve)
    if verbose:
        print("BHC parms [A, n] = [%s, %s], bhcFactor = %s" % (A, n, 1. / n))

    sinogram = projectSingleMaterialWithSpectrum(spectrum, sampleAttPerCm, img, angles=None, vxSzMm=vxSzMm)
    meanAttenuation = np.mean(sinogram) / np.mean(sinogramIdeal)
    sinogramIdeal *= meanAttenuation

    sinogramBH = projectWithBeamHardeningModel(img, A, n, vxSzMm=vxSzMm)

    recon = reconstruct(sinogram)
    if plotBH: # plot the beam-hardened sinogram and reconstruction
        plt.imshow(sinogram, "gray")
        plt.show()
        plt.imshow(recon, "gray")
        plt.show()

    if plotCurve:
        plt.plot(sinogramIdeal[:, 0], label="monochrom.")
        plt.plot(sinogramBH[:, 0], label="beam hard.")
        plt.plot(sinogram[:, 0], label="polychrom.")
        plt.grid()
        plt.legend()
        plt.show()
        plt.plot(recon[Ny // 2, :])
        plt.show()

    return 1./n, recon, sinogramBH, sinogramIdeal


def bhcFromProj(vxSzMm=0.1, sampleDiameterMm=4., bhc=1.0,plot=False):
    # a trivial test to estimate the beam hardening coefficient from a given bhc
    imgShape = (int(30 * sampleDiameterMm), int(30 * sampleDiameterMm))
    img = generateCylinder(imgShape, diameterMm=sampleDiameterMm, vxSzMm=vxSzMm)
    projIdeal = projectWithBeamHardeningModel(img, A=1.0 , n=1.0, angles=[0], vxSzMm=vxSzMm)
    bhf=1./bhc
    projBH = projectWithBeamHardeningModel(img, A=1.0 , n=bhf, angles=[0], vxSzMm=vxSzMm)

    A, n = xip.fitPowerLawToBhcData(np.stack((projIdeal[:,0], np.exp(-projBH[:,0])), axis=1))
    if plot:
        x = np.arange(100) * np.max(projIdeal[:, 0]) / 100
        plt.plot(projIdeal[:,0],label="ideal")
        plt.plot(projBH[:,0],label="BH")
        plt.grid()
        plt.legend()
        plt.show()

        plt.plot(x, A * x ** n, label="fit n = %s" % n)
        plt.plot(projIdeal[:, 0], projBH[:, 0], "x", label="BH curve")
        plt.legend()
        plt.show()
    return 1./n


def idealiseTomo(tomo, threshold=0.4):
    # idealise the binarised tomogram
    return (tomo > threshold).astype(np.float32)


def estimateBhcFromTomo(tomo, sino, plot=False, verbose=True, refineData=False):
    # Estimate the beam hardening coefficient from a tomogram and a sinogram
    projIdeal = np.sum(idealiseTomo(enforceBinaryImage(tomo)), axis=0)
    projBH = np.sum(tomo, axis=0)
    # normalise them to make them comparable
    projBH /= np.max(projBH)
    sino[:, 0] /= np.max(sino[:, 0])
    stackedData = np.stack((projIdeal, np.exp(-projBH)), axis=1)
    stackedDataSino = np.stack((projIdeal, np.exp(-sino[:, 0])), axis=1)

    # if refineData:
    #     stackedData = stackedData[stackedData[:, 0].argsort()]  # Sort by the first column
    #     _, idx = np.unique(stackedData[:, 0], return_index=True)  # Get the indices of unique first arguments
    #     stackedData = np.array([stackedData[stackedData[:, 0] == val][-1]
    #                             for val in np.unique(stackedData[:,0])])
    #     # Keep only the rows with the largest second argument for the same first argument

    A, n = xip.fitPowerLawToBhcData(stackedData)
    A2, n2 = xip.fitPowerLawToBhcData(stackedDataSino)

    if plot:
        x = np.arange(100) * np.max(projIdeal) / 100
        plt.plot(x, A * x ** n, label="fit n = %s" % n)
        plt.plot(stackedData[:,0], -np.log(stackedData[:, 1]), "x", label="BH curve tomo")
        plt.plot(x, A2 * x ** n2, label="fit n = %s" % n2)
        plt.plot(stackedDataSino[:,0], -np.log(stackedDataSino[:, 1]), "x", label="BH curve sino")
        plt.legend()
        plt.show()
    if verbose:
        bhc=1./n
        bhc2=1./n2
        print("bhc tomo = %s" % bhc, "bhc sino = %s" % bhc2)
    return 1./n, 1/n2




if __name__ == "__main__":

    bhc, tomo, sinoBH,_= simulateBH(Ny=400, Nx=400, vxSzMm=0.05, kvp=180,materialName='feo',
                                    sampleDiameterMm=15.0, plotIdeal=False,plotBH=False, plotCurve=False)
    estimateBhcFromTomo(tomo,sinoBH, plot=True)




    exit()
