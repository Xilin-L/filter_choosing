

import numpy as np
import netCDF4 as nc
import os
import json
import sys
from contextlib import redirect_stdout

import automation.extractData as extractData

import materialPropertiesData as mpd
import xraySimulation
import filterPerformance
import beamHardeningSimulation as bhs
import scatteringSimulation as ss
import resolutionEstimation as re
import snrTest



class Tee:
    """ Class to redirect standard output to multiple files """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()


class QualityMeasuresAnalyzer:
    """
    A class to analyse the quality measures of a tomographic dataset
    Input:
    - directoryPath: the path to the directory containing the tomographic dataset
    - sampleMaterial: the material of the sample
    - shape: the shape of the projections, the dtype should be np.uint16

    Available methods and corresponding quality measures:
    - computeBhc: Beam Hardening Correction (BHC)
    - computeScattering: Scattering Contribution
    - computeSnr: Signal-to-Noise Ratio (SNR)
    - computeResolution: Resolution based on decorrelation analysis
    - analyseAll: Perform all analyses and save results to a file
    """
    def __init__(self, directoryPath, sampleMaterial, shape):
        self.directoryPath = directoryPath
        self.sampleMaterial = sampleMaterial
        self.shape = shape
        (self.sampleDiameterMm, self.filterThicknessMm, self.vxSizeMm, self.kvp, self.dfAverage,
         self.cfAverage, self.kfMiddle, self.tomoLoRes) = \
            extractData.extractAllData(self.directoryPath, shape=self.shape, offset=10000)
        self.tomoSliceZ = self.tomoLoRes[self.tomoLoRes.shape[1] // 2]
        self.tomoSliceX = self.tomoLoRes[:, self.tomoLoRes.shape[0] // 2]
        self.tomoSliceY = self.tomoLoRes[:, :, self.tomoLoRes.shape[0] // 2]

    def computeBhc(self):
        """compute beam hardening correction coefficient"""
        sampleDiameterVx = self.sampleDiameterMm / self.vxSizeMm
        bhcTheo, _, tomoSimu, _ = bhs.simulateBH(sampleDiameterMm=self.sampleDiameterMm, sampleDiameterVx=sampleDiameterVx, kvp=self.kvp,
                                                 filterMaterial='Fe', filterThicknessMm=self.filterThicknessMm,
                                                 materialName=self.sampleMaterial, plotCurve=False, verbose=False)
        bhcSimu = bhs.estimateBhcFromTomo(tomoSimu, vxSzMm=self.vxSizeMm, plot=False, verbose=False)
        bhc = bhs.estimateBhcFromTomo(self.tomoSliceZ, vxSzMm=2 * self.vxSizeMm, plot=False, verbose=False)

        print("\n#### BHC Result ####")
        print("Experimental BHC = %.4f" % bhc)
        print("Simulated BHC = %.4f" % bhcSimu)
        print("Theoretical BHC = %.4f" % bhcTheo)

        return bhc

    def computeScattering(self):
        """compute the scattering contribution"""
        transExpe, transTheo, estimatedSampleDiameterMm, result = \
            ss.calcScatteringFromData(self.kfMiddle, self.cfAverage, self.dfAverage, self.tomoSliceZ, self.kvp, 'Fe',
                                      self.filterThicknessMm, self.sampleMaterial, 2 * self.vxSizeMm, offset=0)

        energyKeV, spectrum = filterPerformance.setSpectrum(self.kvp, 'Fe', self.filterThicknessMm)
        scatEsti = ss.estimateMeasuredScatterAsPercentOfFlux(energyKeV, spectrum, self.sampleMaterial, self.sampleDiameterMm)
        print("\n#### Scattering Result ####")
        print("Diameter of the sample is %.4f mm" % estimatedSampleDiameterMm)
        print("Experimental transmission = %.4f" % transExpe)
        print("Theoretical transmission = %.4f" % transTheo)
        print("Scattering contribution = %.4f" % result)
        print("Estimated scattering = %.4f" % (scatEsti / 100))

        return result

    def computeSnr(self,radiusFraction=1):
        """
        compute the signal-to-noise ratio
        can mask the snrZ if it is too high, the radiusFraction can be 0.9 to let the first stdv peak
        greater than the left end
        """
        snrX = snrTest.estimateSNR(self.tomoSliceX, kernelRangePx=3, verbose=False)
        snrY = snrTest.estimateSNR(self.tomoSliceY, kernelRangePx=3, verbose=False)
        snrZ = snrTest.estimateSNR(self.tomoSliceZ, kernelRangePx=3, verbose=False,
                                   mask=True, radiusFraction=radiusFraction)

        print("\n#### SNR Result ####")
        print("SNR X = %.4f" % snrX)
        print("SNR Y = %.4f" % snrY)
        print("SNR Z = %.4f" % snrZ)

        return [snrX, snrY, snrZ]

    def computeResolution(self):
        """compute the resolution of the tomographic dataset"""
        resX = re.findImageRes(self.tomoSliceX, pxSzMm=2 * self.vxSizeMm, Ng=8, plot=False)
        resY = re.findImageRes(self.tomoSliceY, pxSzMm=2 * self.vxSizeMm, Ng=8, plot=False)
        resZ = re.findImageRes(self.tomoSliceZ, pxSzMm=2 * self.vxSizeMm, Ng=8, plot=False)
        # resZ is much lower than the other two, similar to the SNR result without masking

        print("\n#### Resolution Result ####")
        print("Resolution X = %.4f mm" % resX)
        print("Resolution Y = %.4f mm" % resY)
        print("Resolution Z = %.4f mm" % resZ)

        return [resX, resY, resZ]

    def analyseAll(self):
        """
        Perform the analysis of the quality measures and save both the results and the log to the parent directory
        :return: bhc, scattering, snr, resolution
        """

        # Get the parent directory of the input directory
        parentDirectory = os.path.dirname(self.directoryPath)

        # Generate the log file name
        logFileName = f"{self.sampleMaterial}_{int(self.sampleDiameterMm)}mm_{int(self.kvp)}kv_log.txt"
        logFilePath = os.path.join(parentDirectory, logFileName)

        # Open the log file
        with open(logFilePath, "w") as logFile:
            # Redirect standard output to both the log file and the console
            with redirect_stdout(Tee(sys.stdout, logFile)):
                # Perform the analysis
                bhc = self.computeBhc()
                scattering = self.computeScattering()
                snr = self.computeSnr()
                resolution = self.computeResolution()

                results = {
                    "BHC": bhc,
                    "Scattering": scattering,
                    "SNR": snr,
                    "Resolution": resolution
                }

                # Generate the output file name
                resultFileName = f"{self.sampleMaterial}_{int(self.sampleDiameterMm)}mm_{int(self.kvp)}kv_result.json"
                resultFilePath = os.path.join(parentDirectory, resultFileName)

                # Save results to a file in the parent directory
                with open(resultFilePath, "w") as resultFile:
                    json.dump(results, resultFile, indent=4)

        return bhc, scattering, snr, resolution



# if __name__ == '__main__':
#     al33mm180kv = QualityMeasuresAnalyzer('/home/xilin/projects/recon_ws/EfficientScans/'
#                                        'AL_33mm__180kV-80uA_bin2_450_CD1150mm', "Al", shape=(938, 938))
#     al33mm180kv.analyseAll()


if __name__ == '__main__':
    pmma38mm60kv = QualityMeasuresAnalyzer('/home/xilin/projects/recon_ws/EfficientScans/'
                                          'PMMA_38mm__60kV-150uA_bin2_450_CD1150mm', "pmma", shape=(938, 938))
    pmma38mm60kv.analyseAll()