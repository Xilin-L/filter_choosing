import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import netCDF4 as nc


import xraySimulation as xs
import beamHardeningSimulation as bhs
import materialPropertiesData as mpd
import filterPerformance

from scipy.integrate import quad


def forwardFraction(energyKeV, coneAngleDeg):
    """
    Numerically integrate the Klein–Nishina differential cross-section
    over 0→θ_max and 0→2π in phi to get f_incoh(E).
    """

    θ_max = np.deg2rad(coneAngleDeg)
    mc2   = 511.0  # electron rest mass in keV
    E     = energyKeV

    def kn_diff_cosμ(cosθ):
        θ = np.arccos(cosθ)
        Eprime = E / (1 + (E/mc2)*(1 - cosθ))
        kn = (Eprime/E)**2 * (E/Eprime + Eprime/E - np.sin(θ)**2)
        return kn

    # integrate dσ/dΩ dΩ = ∫φ=0²π ∫θ=0->θmax kn(θ) sinθ dθ dφ
    # since kn_diff_cosμ gives kn vs cosθ, and d(cosθ) = -sinθ dθ,
    # ∫θ=0->θmax kn sinθ dθ = ∫μ=cosθmax->1 kn dμ
    μ_max = np.cos(θ_max)
    num = 2*np.pi * quad(kn_diff_cosμ, μ_max, 1)[0]
    # total over 4π
    den = 2*np.pi * quad(kn_diff_cosμ, -1, 1)[0]
    return num/den


def estimateMeasuredScatterAsPercentOfFlux(energyKeV,spectrum,sampleMaterial,sampleDiameterMm,coneAngleDeg=10.):
    # estimate fraction of spectrum flux is detected as scatter
    # get material properties
    thicknessAvgCm = 0.064*sampleDiameterMm #average chord length
    materialWeights,materialSymbols,dens = mpd.getMaterialProperties(sampleMaterial)
    # estiamte amount of total scatter
    # Start with Compton ('absorption' component)
    sampIncohScat = xs.calcIncohScatter(energyKeV,materialWeights,materialSymbols,dens,thicknessAvgCm)
    # assume incoh scatter in all directions, so fraction detected is:
    scatFracDet = np.array([
        forwardFraction(E, coneAngleDeg)
        for E in energyKeV
    ])
    # scatFracDet = (coneAngleDeg**2)/(360.0**2)
    spectrumScat = spectrum*sampIncohScat
    # Plus Raleigh ('absorption' component after transmission from Compton)
    sampIncohTrans = xs.calcIncohTransmission(energyKeV,materialWeights,materialSymbols,dens,thicknessAvgCm)
    sampCohScat = xs.calcCohScatter(energyKeV,materialWeights,materialSymbols,dens,thicknessAvgCm)
    spectrumScat += spectrum*(sampIncohTrans)*sampCohScat
    # attenuate total scatter by sample half
    sampPhotoTrans = xs.calcTransmission(energyKeV,materialWeights,materialSymbols,dens,0.05*sampleDiameterMm)
    spectrumScat *= sampPhotoTrans*scatFracDet
    # total scatter energy is:
    scatEnergyFluence = np.sum(spectrumScat)

    sampTrans = xs.calcTransmission(energyKeV, materialWeights, materialSymbols, dens, sampleDiameterMm*0.1)
    specTrans = spectrum * sampTrans

    # ### filter by Carbon Fibre (3.7mm)
    # # (65%V, 70%W C p=1.7, 35%V 30%W C21H29O3 p=1.35)
    # # Z=1	: 0.088722
    # # Z=6	: 0.765590
    # # Z=8	: 0.145688
    # materialWeights = [1.0]
    # materialSymbols = ["C"]
    # dens = 1.7
    # thicknessCm = 0.4
    # scat1 = xs.calcIncohScatter(energyKeV, materialWeights, materialSymbols, dens, 0.65 * thicknessCm)
    # dens = 1.35
    # materialWeights = [0.088722, 0.765590, 0.145688]
    # materialSymbols = ["H", "C", "O"]
    # scat2 = xs.calcIncohScatter(energyKeV, materialWeights, materialSymbols, dens, 0.35 * thicknessCm)
    # ### filter by Aluminium (0.1mm)
    # materialWeights = [1.0]
    # materialSymbols = ["Al"]
    # dens = 2.7
    # thicknessCm = 0.01
    # scat3 = xs.calcIncohScatter(energyKeV, materialWeights, materialSymbols, dens, thicknessCm)
    # detectorScat = specTrans* (scat1 + scat2 + scat3)
    # ratio = np.sum(detectorScat) / np.sum(specTrans)/2
    # print("detector scatter / transmission = ", ratio)

    return 100.0*scatEnergyFluence/(np.sum(specTrans)+scatEnergyFluence)
    # return 100.0*scatEnergyFluence/np.sum(spectrum)
#


# def estimateMeasuredScatterAsPercentOfFlux(energyKeV,spectrum,sampleMaterial,sampleDiameterMm,coneAngleDeg=60.):
#     # estimate fraction of spectrum flux is detected as scatter
#     # get material properties
#     thicknessAvgCm = 0.075*sampleDiameterMm
#     materialWeights,materialSymbols,dens = mpd.getMaterialProperties(sampleMaterial)
#     # estiamte amount of total scatter
#     # Start with Compton ('absorption' component)
#     sampIncohScat = xs.calcIncohScatter(energyKeV,materialWeights,materialSymbols,dens,thicknessAvgCm)
#     spectrumScat = spectrum*sampIncohScat
#     # Plus Raleigh ('absorption' component after transmission from Compton)
#     sampIncohTrans = xs.calcIncohTransmission(energyKeV,materialWeights,materialSymbols,dens,thicknessAvgCm)
#     sampCohScat = xs.calcCohScatter(energyKeV,materialWeights,materialSymbols,dens,thicknessAvgCm)
#     spectrumScat += spectrum*(sampIncohTrans)*sampCohScat
#     # attenuate total scatter by photo-electric half(thicknessAvg)
#     sampPhotoTrans = xs.calcPhotoTransmission(energyKeV,materialWeights,materialSymbols,dens,0.5*thicknessAvgCm)
#     spectrumScat *= sampPhotoTrans
#     # total scatter energy is:
#     scatEnergyFluence = np.sum(spectrumScat)
#     # assume scatter in all directions, so fraction detected is:
#     θ = np.deg2rad(coneAngleDeg)
#     scatFracDet = 0.5 * (1 - np.cos(θ))

    # scatFracDet = (coneAngleDeg**2)/(360.0**2)
    # '''
    # change spectrum to spectrum_trans as we are comparing signal measured to noise in return
    # '''
    # sample_trans = xs.calcTransmission(energyKeV, materialWeights, materialSymbols, dens, thicknessAvgCm)
    # spectrum_trans = spectrum*sample_trans
    # return 100.0*scatEnergyFluence*scatFracDet/np.sum(spectrum)



def getRadiusMm(img, voxelSizeMm=0.1, threshold=0.1, offset=10000):
    temp = img-offset # remove the offsetand the wall
    temp[temp<0]=0
    # look at the actual tomo to change the threshold
    imgIdealised=bhs.idealiseTomo(bhs.enforceBinaryImage(temp),threshold=threshold)
    radius=np.sqrt(np.sum(imgIdealised)/np.pi)*voxelSizeMm

    return radius


def transmissionTheoretical(tomoSlice,kvp=120,filterMaterial='Cu', filterThicknessMm=0.5,
                            materialName='feo', voxelSizeMm=0.1,offset=10000):
    sampleDiameterMm = 2 * getRadiusMm(tomoSlice, voxelSizeMm=voxelSizeMm, threshold=0.1, offset=offset)
    sampleThicknessCm = 0.1 * sampleDiameterMm
    energyKeV, spectrum = filterPerformance.setSpectrum(kvp,filterMaterial,filterThicknessMm)
    materialWeights, materialSymbols, dens = mpd.getMaterialProperties(materialName)
    sampleAttPerCm = xs.getMaterialAttenuation(energyKeV, materialWeights, materialSymbols, dens)
    specNormalisd = spectrum / np.sum(spectrum)

    sampleTransmission = np.sum(specNormalisd[:, np.newaxis, np.newaxis]
                   * np.exp(-sampleAttPerCm[:, np.newaxis, np.newaxis] * sampleThicknessCm),
                   axis=0)
    # sampleTransmission = np.exp(-sampleAttPerCm*sampleThicknessCm)
    return sampleTransmission.flatten()[0], sampleDiameterMm


def normaliseProj(proj,clearField,darkField):
    projNorm= (proj - darkField) / (clearField - darkField)
    projNorm[projNorm < 0]=0
    projNorm[projNorm > 1]=1
    projSmooth=sp.ndimage.gaussian_filter(projNorm,2)
    return projSmooth


def calcScatteringFromData(kf, cf, df, tomoSlice, kvp, filterMaterial, filterThicknessMm,
                      sampleMaterial, vxSizeMm,offset=10000):
    projSmooth = normaliseProj(kf, cf, df)
    midIdx = projSmooth.shape[0] // 2
    choppedProj = projSmooth[midIdx - 100:midIdx + 100, :]
    # assume the projection at the center is the smallest value in the image
    transExpe = np.nanmin(choppedProj)

    transTheo, EstimatedSampleDiameterMm = transmissionTheoretical(tomoSlice, kvp=kvp,
                                                                      filterMaterial=filterMaterial,
                                                                      filterThicknessMm=filterThicknessMm,
                                                                      materialName=sampleMaterial,
                                                                      voxelSizeMm=vxSizeMm, offset=offset)
    result = transExpe - transTheo

    return transExpe, transTheo, EstimatedSampleDiameterMm, result


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

    tomo = np.array(nc.Dataset('/home/xilin/projects/recon_ws/AM/AM_Kingston_BH_38mmCaCO3_SFT/tomoLoRes.nc'
                               ).variables['tomo'][:], dtype=np.float32, copy=True)

    tomoMidIdx=tomo.shape[0]//2
    tomoSlice=tomo[tomoMidIdx]

    transExpe, transTheo, EstimatedSampleDiameterMm, result = \
        calcScatteringFromData(proj, clearField, darkField, tomoSlice, 100, 'Al',
                               0.5, 'marble', 0.123, offset=10000)

    print("Experimental transmission = %s" %transExpe)
    print("Theoretical transmission = %s" %transTheo)
    print("Diameter of the sample is %s mm" % EstimatedSampleDiameterMm)
    print("Scattering contribution = %s" %result)


'''
From tomo.in, the voxel size is 30.98 um and diameter is 38 mm
the tomoLowRes.nc is binning by 4, so the voxel size is 30.98*4=123.92 um = 0.12392 mm

'''



