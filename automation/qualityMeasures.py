

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
import resolutionEstimation as resEst
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
    def __init__(self, directoryPath: str, sampleMaterial: str, shape: tuple[int, int]):
        self.directoryPath = directoryPath
        self.sampleMaterial = sampleMaterial
        self.shape = shape
        (self.sampleDiameterMm, self.filterThicknessMm, self.vxSizeMm, self.kvp, self.binning, self.dfAverage,
         self.cfAverage, self.kfMiddle, self.tomoSliceX, self.tomoSliceY, self.tomoSliceZ) = \
            extractData.extractAllData(self.directoryPath, shape=self.shape, offset=10000)
        (_,_,_,_,_,_,_,_, self.tomoSliceXraw, self.tomoSliceYraw, self.tomoSliceZraw) = \
            extractData.extractAllData(self.directoryPath, shape=self.shape, offset=0)
        # self.tomoSliceZ = self.tomoLoRes[self.tomoLoRes.shape[1] // 2]
        # self.tomoSliceX = self.tomoLoRes[:, self.tomoLoRes.shape[0] // 2]
        # self.tomoSliceY = self.tomoLoRes[:, :, self.tomoLoRes.shape[0] // 2]

        # self.tomoLoResFactor = np.floor(self.kfMiddle.shape[0] / self.tomoSliceZ.shape[0])

    def computeBhc(self):
        """compute beam hardening correction coefficient"""
        sampleDiameterVx = self.sampleDiameterMm / self.vxSizeMm
        bhcTheo, _, tomoSimu, _ = bhs.simulateBH(sampleDiameterMm=self.sampleDiameterMm,
                                                 sampleDiameterVx=sampleDiameterVx, kvp=self.kvp,
                                                 filterMaterial='Fe', filterThicknessMm=self.filterThicknessMm,
                                                 materialName=self.sampleMaterial, plotCurve=False, verbose=False)
        bhcSimu = bhs.estimateBhcFromTomo(tomoSimu, vxSzMm=self.vxSizeMm, plot=False, verbose=False)
        bhc = bhs.estimateBhcFromTomo(self.tomoSliceZ, vxSzMm=self.vxSizeMm, plot=False, verbose=False)

        print("\n#### BHC Result ####")
        print("Experimental BHC = %.4f" % bhc)
        print("Simulated BHC = %.4f" % bhcSimu)
        print("Theoretical BHC = %.4f" % bhcTheo)

        return bhc, bhcSimu, bhcTheo

    def computeScattering(self):
        """compute the scattering contribution"""
        transExpe, transTheo, estimatedSampleDiameterMm, result = \
            ss.calcScatteringFromData(self.kfMiddle, self.cfAverage, self.dfAverage, self.tomoSliceZ, self.kvp, 'Fe',
                                      self.filterThicknessMm, self.sampleMaterial, self.vxSizeMm, offset=0)

        energyKeV, spectrum = filterPerformance.setSpectrum(self.kvp, 'Fe', self.filterThicknessMm)
        scatEsti = ss.estimateMeasuredScatterAsPercentOfFlux(energyKeV, spectrum, self.sampleMaterial, self.sampleDiameterMm)
        scatEsti /= 100
        print("\n#### Scattering Result ####")
        print("Diameter of the sample is %.4f mm" % estimatedSampleDiameterMm)
        print("Experimental transmission = %.4f" % transExpe)
        print("Theoretical transmission = %.4f" % transTheo)
        print("Scattering contribution = %.4f" % result)
        print("Estimated scattering = %.4f" % scatEsti)

        return result, scatEsti

    def computeSnr(self,kernelRangePx=9):
        """
        compute the signal-to-noise ratio
        can mask the snrZ if it is too high, the radiusFraction can be 0.9 to let the first stdv peak
        greater than the left end
        """
        # snrX, rangeX = snrTest.estimateSNR(self.tomoSliceX, kernelRangePx=9, verbose=False, cropFactor=0.32)
        # snrY, rangeY = snrTest.estimateSNR(self.tomoSliceY, kernelRangePx=9, verbose=False, cropFactor=0.32)
        # snrZ, rangeZ = snrTest.estimateSNR(self.tomoSliceZ, kernelRangePx=9, verbose=False, cropFactor=0.32)

        snrX, rangeX = snrTest.estimateSNR(self.tomoSliceXraw, kernelRangePx=9, verbose=False)
        snrY, rangeY = snrTest.estimateSNR(self.tomoSliceYraw, kernelRangePx=9, verbose=False)
        snrZ, rangeZ = snrTest.estimateSNR(self.tomoSliceZraw, kernelRangePx=9, verbose=False, circularMask=True)

        print("\n#### SNR Result ####")
        print("SNR X = %.4f" % snrX)
        print("SNR Y = %.4f" % snrY)
        print("SNR Z = %.4f" % snrZ)

        return [snrX, snrY, snrZ] , [rangeX, rangeY, rangeZ]

    def computeResolution(self):
        """compute the resolution of the tomographic dataset"""
        resX, resPxX = resEst.findImageRes(self.tomoSliceX, pxSzMm=self.vxSizeMm, Ng=10, plot=False, cropFactor=0)
        resY, resPxY = resEst.findImageRes(self.tomoSliceY, pxSzMm=self.vxSizeMm, Ng=10, plot=False, cropFactor=0)
        resZ, resPxZ = resEst.findImageRes(self.tomoSliceZ, pxSzMm=self.vxSizeMm, Ng=10, plot=False, cropFactor=0.4)
        # resZ is much lower than the other two, similar to the SNR result without masking

        print("\n#### Resolution Result ####")
        print("Resolution X = %.4f mm (%.2f pixels)" % (resX, resPxX))
        print("Resolution Y = %.4f mm (%.2f pixels)" % (resY, resPxY))
        print("Resolution Z = %.4f mm (%.2f pixels)" % (resZ, resPxZ))

        return [resX, resY, resZ], [resPxX, resPxY, resPxZ]


    def analyseAll(self):
        """
        Perform the analysis of the quality measures and save both the results and the log to the parent/results directory
        :return: bhc, scattering, snr, resolution
        """

        # Get the parent directory of the input directory
        parentDirectory = os.path.dirname(self.directoryPath)

        # Create the results directory if it does not exist
        resultsDirectory = os.path.join(parentDirectory, 'results')
        os.makedirs(resultsDirectory, exist_ok=True)

        # Generate the result file name
        resultFileName = f"{self.sampleMaterial}_{int(self.sampleDiameterMm)}mm_{int(self.kvp)}kv_bin{int(self.binning)}_result.json"
        resultFilePath = os.path.join(resultsDirectory, resultFileName)

        # Check if the result file already exists
        if os.path.exists(resultFilePath):
            print(f"Result file already exists: {resultFilePath}")
            return

        # Generate the log file name
        logFileName = f"{self.sampleMaterial}_{int(self.sampleDiameterMm)}mm_{int(self.kvp)}kv_bin{int(self.binning)}_log.txt"
        logFilePath = os.path.join(resultsDirectory, logFileName)

        # Open the log file
        with open(logFilePath, "w") as logFile:
            # Redirect standard output to both the log file and the console
            with redirect_stdout(Tee(sys.stdout, logFile)):
                # Perform the analysis
                # bhc, bhcSimu, bhcTheo = self.computeBhc()
                # scattering, scatEsti = self.computeScattering()
                snr, snrRange = self.computeSnr()
                resolution, resolutionPx = self.computeResolution()

                results = {
                    # "BHC": [bhc, bhcSimu, bhcTheo],
                    # "Scattering": [scattering, scatEsti],
                    "SNR": snr,
                    "SNRRange": snrRange,
                    "Resolution": resolution,
                    "ResolutionPx": resolutionPx
                }

                # Save results to a file in the results directory
                with open(resultFilePath, "w") as resultFile:
                    json.dump(results, resultFile, indent=4)

        # return bhc, scattering, snr, resolution, resolutionPx



if __name__ == '__main__':

    import os
    import re

    base_path = '/home/xilin/projects/recon_ws/EfficientScans/'

    # Regular expression to match the folder name pattern
    folder_pattern = re.compile(
        r'(?P<sampleMaterial>[^_]+)_(?P<sampleDiameter>[^_]+)__(?P<kvp>[^-]+)-(?P<sourceCurrent>[^_]+)_bin(?P<binning>\d+)_.*')

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            match = folder_pattern.match(folder_name)
            if match:
                sampleMaterial = match.group('sampleMaterial')
                binning = int(match.group('binning'))
                # Corrected shape assignment based on binning value
                if binning == 1:
                    shape = (1876, 1876)
                elif binning == 2:
                    shape = (938, 938)
                else:
                    raise ValueError(f"Unsupported binning value: {binning}")

                print(f"\nProcessing folder: {folder_name}")
                print(f"Sample Material: {sampleMaterial}, Binning: {binning}, Shape: {shape}")

                analyzer = QualityMeasuresAnalyzer(folder_path, sampleMaterial, shape)
                analyzer.computeScattering()
            else:
                print(f"\nFolder name does not match pattern: {folder_name}")



