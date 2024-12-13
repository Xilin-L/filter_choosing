import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import netCDF4 as nc


import xraySimulation as xs
import xrayImagingPerformance as xip
import beamHardeningSimulation as bhs

def getRadiusMm(img, voxelSizeMm=0.1, threshold=0.1):
    temp = img-10000 # remove the offsetand the wall
    temp[temp<0]=0
    # look at the actual tomo to change the threshold
    imgIdealised=bhs.idealiseTomo(bhs.enforceBinaryImage(temp),threshold=threshold)
    radius=np.sqrt(np.sum(imgIdealised)/np.pi)*voxelSizeMm

    return radius


def transmissionTheoretical(tomoSlice,kvp=120,filterMaterial='Cu', filterThicknessMm=0.5,
                            materialName='feo', voxelSizeMm=0.1):
    sampleDiameterMm = 2 * getRadiusMm(tomoSlice, voxelSizeMm=voxelSizeMm, threshold=0.1)
    sampleThicknessCm = 0.1 * sampleDiameterMm
    energyKeV, spectrum = xs.setSpectrum(kvp,filterMaterial,filterThicknessMm)
    materialWeights, materialSymbols, dens = xip.getMaterialProperties(materialName)
    sampleAttPerCm = xs.getMaterialAttenuation(energyKeV, materialWeights, materialSymbols, dens)
    specNormalisd = spectrum / np.sum(spectrum)

    sampleTransmission = np.sum(specNormalisd[:, np.newaxis, np.newaxis]
                   * np.exp(-sampleAttPerCm[:, np.newaxis, np.newaxis] * sampleThicknessCm),
                   axis=0)
    # sampleTransmission = np.exp(-sampleAttPerCm*sampleThicknessCm)
    return sampleTransmission.flatten()[0]


def normaliseProj(proj,clearField,darkField):
    projNorm= (proj - darkField) / (clearField - darkField)
    projNorm[projNorm < 0]=0
    projNorm[projNorm > 1]=1
    projSmooth=sp.ndimage.gaussian_filter(projNorm,2)
    return projSmooth


if __name__ == '__main__':




    proj = np.fromfile(
        '/home/xilin/projects/recon_ws/AM/AM_Kingston_BH_38mmCaCO3_SFT/proju16_raw/rawfiles_KF0000/expt_KF001080.raw',
        dtype=np.uint16).reshape((1420, 1436))

    clearField = np.fromfile(
        '/home/xilin/projects/recon_ws/AM/AM_Kingston_BH_38mmCaCO3_SFT/proju16_raw/rawfiles_CF0000/expt_CF000000.raw',
        dtype=np.uint16).reshape((1420, 1436))

    darkField = np.fromfile(
        '/home/xilin/projects/recon_ws/AM/AM_Kingston_BH_38mmCaCO3_SFT/proju16_raw/rawfiles_DF0000/expt_DF000000.raw',
        dtype=np.uint16).reshape((1420, 1436))

    projSmooth=normaliseProj(proj,clearField,darkField)
    midIdx=projSmooth.shape[0]//2
    choppedProj=projSmooth[midIdx-100:midIdx+100,:]

    # plt.imshow(choppedProj)
    # plt.show()

    # assume the projection at the center is the smallest value in the image
    transExpe=np.min(choppedProj)
    print("Experimental transmission = %s" %transExpe)



    tomo = np.array(nc.Dataset('/home/xilin/projects/recon_ws/AM/AM_Kingston_BH_38mmCaCO3_SFT/tomoLoRes.nc'
                               ).variables['tomo'][:], dtype=np.float32, copy=True)

    tomoMidIdx=tomo.shape[0]//2
    tomoSlice=tomo[tomoMidIdx]
    transTheo = transmissionTheoretical(tomoSlice, kvp=100, filterMaterial='Al', filterThicknessMm=0.5,
                                        materialName='marble', voxelSizeMm=0.123)
    print("Theoretical transmission = %s" %transTheo)

    diameterMm= 2 * getRadiusMm(tomoSlice, voxelSizeMm=0.123, threshold=0.1)
    print("Diameter of the sample is %s mm" % diameterMm)


    result=transExpe-transTheo
    print("Scattering contribution = %s" %result)


'''
From tomo.in, the voxel size is 30.98 um and diameter is 38 mm
the tomoLowRes.nc is binning by 4, so the voxel size is 30.98*4=123.92 um = 0.12392 mm

'''



