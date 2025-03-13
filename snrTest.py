import numpy as np

import scipy as sp

from scipy import ndimage

from scipy import signal

import matplotlib.pyplot as plt



from resolutionEstimation import generateTestImage

'''
define the SNR/CNR as follows:
dataRange/noiseStdv

where:

dataRange = highestHistPeakPos - lowestHistPeakPos
       OR = 5*stdv of noise free image (if only one peak in histogram)
noise free image created by median filtering data

noiseStdv = stdv of noise only image
lowest peak in the local stdv histogram
'''

def getCumulativeHistProb(hist):

    prob = hist/np.sum(hist)

    cumHistProb = np.cumsum(prob) # CDF of the histogram

    return cumHistProb



def getHistPercentilePos(hist,percentile=10):

    Nb = len(hist)

    if percentile < 0. or percentile > 100.:
        raise Exception("Invalid percentile specified: %s\\% (should be in [0,100])"%(percentile))
    else:

        prob = percentile/100.

    cumHistProb = getCumulativeHistProb(hist)

    pos = np.searchsorted(cumHistProb, prob) # faster sorting

    # pos = 0

    # while cumHistProb[pos] < prob and pos < Nb-1:

    #     pos += 1

    return pos



def identifySignificantPeaks(hist):

    Nb = len(hist)

    probmin = 100.0/float(Nb)

    pmin = getHistPercentilePos(hist,percentile=probmin)

    probmax = 100.0 - probmin

    pmax = getHistPercentilePos(hist,percentile=probmax)

    peaks,_ = sp.signal.find_peaks(hist[pmin:pmax])

    if not peaks.size:

        raise Exception("no peaks found in median filtered image histogram.")

    peaks += pmin

    peakCounts = hist[peaks]

    T = sp.ndimage.median(hist)

    if T < 0.:

        return peaks

    else:

        peaksOut = [peaks[pc] for pc in range(len(peaks)) if peakCounts[pc] > T]

        # peaksOut = []

        # for pc in range(len(peaks)):

        #     if peakCounts[pc] > T:

        #         peaksOut.append(peaks[pc])

        return np.array(peaksOut)



def optimiseHistogramRange(img,numBins=256):

    img = img[~np.isnan(img)]  # Remove NaN values

    pmin = 100.0/float(numBins)

    pmax = 100.0-pmin

    return np.percentile(img,(pmin,pmax))



def histogram(data,bins=256):

    return np.histogram(data[~np.isnan(data)],bins=bins,range=optimiseHistogramRange(data))



def calcDataRange(img,medFiltRangePx=1,plotHistChange=False):

    medFiltSizePx = 2*medFiltRangePx + 1

    imgMed = sp.ndimage.median_filter(img,medFiltSizePx)
    imgMed = imgMed[~np.isnan(imgMed)]  # Remove NaN values
    hist,edges = histogram(img)

    vals = 0.5*(edges[1:] + edges[:-1]) # bin centers

    histMed,edgesMed = histogram(imgMed)

    valsMed = 0.5*(edgesMed[1:] + edgesMed[:-1]) # bin centers after median filter

    sigma = len(histMed)/256.

    histMedSmth = sp.ndimage.gaussian_filter(histMed,sigma)

    peaks = identifySignificantPeaks(histMedSmth)

    if not peaks.size:

        raise Exception("no peaks found in median filtered image histogram.")

    elif peaks.size<2: # for single peak

        if peaks[0] > 0 and histMedSmth[0] > histMedSmth[peaks[0]]:

            # if the left end is a peak

            dmin = valsMed[0]

            dmax = valsMed[peaks[0]]

            dataRange = dmax - dmin

        elif peaks[0] < len(histMedSmth) - 1 and histMedSmth[-1] > histMedSmth[peaks[-1]]:

            # if the right end is a peak

            dmin = valsMed[peaks[0]]

            dmax = valsMed[-1]

            dataRange = dmax - dmin

        else:

            print("Only a single peak found in the histgram, determining data range as 5 x sigma")

            dataRangeSigma = np.std(imgMed)

            dataRange = 5.0*dataRangeSigma

    elif peaks[0] > 0 and histMedSmth[0] > histMedSmth[peaks[0]]:

        # if the left end is a peak

        dmin = valsMed[0]

        dmax = valsMed[peaks[-1]]

        dataRange = dmax - dmin

    else:

        dmin = valsMed[peaks[0]]

        dmax = valsMed[peaks[-1]]

        dataRange = dmax - dmin

    if plotHistChange is True:

        plt.plot(vals,hist,label="hist.")

        plt.plot(valsMed,histMed,label="med hist.")

        plt.plot(valsMed,histMedSmth,label="smth med hist.")

        plt.plot(valsMed[peaks],histMedSmth[peaks],"x", markersize=10, markeredgewidth=2, label="peaks")

        plt.xlabel("img. gray values")

        plt.ylabel("counts")

        plt.legend()

        plt.show()

    return dataRange



def calcStdvHistogram(img,stdFiltRangePx=1,plotStdvImg=False):

    stdFiltSizePx = 2*stdFiltRangePx + 1

    imgStdv = sp.ndimage.generic_filter(img, np.std, size=stdFiltSizePx) # local std. dev. within given filter size

    hist,edges = histogram(imgStdv)

    vals = 0.5*(edges[1:] + edges[:-1])

    if plotStdvImg is True:
        imgStdv = np.clip(imgStdv, edges[0], edges[-1]) # shifted this line from if statement below

        plt.imshow(imgStdv,"gray")

        plt.title("Local std. dev. (kernel size = %s px)"%(stdFiltSizePx))
        plt.colorbar(label='Value')
        plt.show()


    return hist,vals



def calcNoiseStdv(img,stdFiltRangePx=1,plotData=False):

    hist,vals = calcStdvHistogram(img,stdFiltRangePx,plotData)

    sigma = len(hist)/100.

    histSmth = sp.ndimage.gaussian_filter(hist,sigma)

    peaks,_ = sp.signal.find_peaks(histSmth)

    if plotData is True:

        plt.plot(vals,hist,label="hist.")

        plt.plot(vals,histSmth,label="smth hist.")

        plt.plot(vals[peaks],histSmth[peaks],"x", markersize=10, markeredgewidth=2,label="peaks")

        plt.xlabel("local std. dev.")

        plt.ylabel("counts")
        plt.semilogy()
        plt.legend()

        plt.show()

    if not peaks.size:

        if np.max(histSmth) > histSmth[0]:

            peaks = np.array([0])

        else:

            raise Exception("No peaks found in the stdv histogram. Unable to estimate noise levels.")

    if histSmth[0] > histSmth[peaks[0]]:

        peak = 0

    else:

        peak = peaks[0] # select the lowest peak as the noise stdv

    stdv = vals[peak]

    return stdv



def estimateSNR(img, kernelRangePx=1, plotData=False, verbose=False, circularMask=False, radiusFraction=1, cropFactor=0):
    """
    Estimate the signal-to-noise ratio (SNR) of an image.
    :param img: image as a 2D numpy array
    :param kernelRangePx: the range of the median/std filter kernel in pixels
    :param plotData: if True, plot the data
    :param verbose: if True, print the SNR information
    :param circularMask: if True, only contains the image within a circle of radiusFraction
    :param radiusFraction: radius of the circular mask as a fraction of the image size, default is 1
    :param cropFactor:  if 0, no cropping, if between 0 and 0.5, crop the image by that fraction, default is 0
    :return: snr, dataRange
    """
    if circularMask:
        # only contains the sample and a small ring around it
        imgFiltered = np.copy(img)
        center = np.array(imgFiltered.shape) // 2
        distance_from_center = np.sqrt(
            ((np.indices(imgFiltered.shape) - center[:, None, None]) ** 2).sum(axis=0))
        imgFiltered[(imgFiltered < 0) | (distance_from_center > center[0] * radiusFraction)] = np.nan  # Set negative values and values away from center to NaN
        img = imgFiltered
    elif 0.5 > cropFactor > 0:
        crop_size = int(cropFactor * np.min(img.shape))
        img = img[crop_size:-crop_size, crop_size:-crop_size]
    elif cropFactor == 0:
        img = img
    else:
        raise Exception("cropFactor should between 0 and 0.5")

    dataRange = calcDataRange(img, kernelRangePx, plotData)
    noiseStdv = calcNoiseStdv(img, kernelRangePx, plotData)
    snr = dataRange / noiseStdv

    if verbose:
        print("Using size = %s for a median filter and local std. dev. estimates:" % (2 * kernelRangePx + 1))
        print("data range = %s" % (dataRange))
        print("noise stdv = %s" % (noiseStdv))
        print("SNR = %s" % (snr))

    return snr, dataRange



# analyse .nc file
# import netCDF4 as nc
#
# def tomoSliceSNR(nc_file, plot=False, kernelRangePx=1):
#     tomoSlice = nc.Dataset(nc_file)
#     tomoData = np.array(tomoSlice.variables['tomo'][:], dtype=np.float32, copy=True)
#
#     # Determine the dimension with length 1
#     data_dim = np.argmin(tomoData.shape)
#
#     # Reshape the data to have the data dimension last
#     tomoData = np.moveaxis(tomoData, data_dim, -1)
#
#     if plot:
#         # Plot the grayscale image
#         tomoDataPlot = np.copy(tomoData) - 10000  # shift does not matter in the SNR calculation though
#         tomoDataPlot[tomoDataPlot < 0] = 0  # Set negative values to zero
#         plt.imshow(tomoDataPlot, cmap='gray')
#         plt.colorbar(label='Value')
#         plt.title('Grayscale Plot of Data Array')
#         plt.show()
#
#     # Estimate the SNR
#     if 'tomoSliceZ' in tomoSlice.filepath():
#         # to remove the container walls for the Z data
#         tomoDataFiltered = np.copy(tomoData)
#         center = np.array(tomoDataFiltered.shape) // 2
#         distance_from_center = np.sqrt(
#             ((np.indices(tomoDataFiltered.shape) - center[:, None, None, None]) ** 2).sum(axis=0))
#         tomoDataFiltered[(tomoDataFiltered < 0) | (
#                 distance_from_center > center[0])] = np.nan  # Set negative values and values away from center to zero
#
#         snr = estimateSNR(tomoDataFiltered, kernelRangePx, plot)
#     else:
#         snr = estimateSNR(tomoData, kernelRangePx, plot)
#
#     return snr





# if __name__ == "__main__":

    # N = 512
    # img = generateTestImage(N,pxSzMm=1.0,resMm=3.5,numPhotonsPerPx=1000.)
    # plt.imshow(img)
    # plt.show()
    #
    # snr1 = estimateSNR(img,1,True)
    # snr2 = estimateSNR(img,2,True)
    # snr3 = estimateSNR(img,3,True)



    # from scipy import misc
    # img = sp.misc.ascent().astype(np.float32)
    # plt.imshow(img)
    # plt.show()
    #
    # snr1 = estimateSNR(img,1,True)
    #


    # img = np.random.normal(100.,20.,size=(512,512))
    # plt.imshow(img)
    # plt.show()
    #
    # snr1 = estimateSNR(img,1,True)


    # img = np.random.normal(1000.,100.,size=(512,512))
    # img = sp.ndimage.gaussian_filter(img,5.0) # apply gaussian filter
    # img -= np.min(img) # shift the minimum to zero
    # # poisson-like distribution
    # img = np.random.normal(img,np.sqrt(img)) # add poisson noise
    # '''
    #  larger sigma in producing img gives larger SNR due to the shift in the minimum,
    #  the growth of poisson noise is slower than the growth of the data range
    # '''
    # plt.imshow(img)
    # plt.colorbar()
    # plt.show()
    #
    # snr1 = estimateSNR(img,1,True)

    # from PIL import Image
    #
    # img = Image.open('/home/xilin/Downloads/pic5.jpg').convert('L')
    # img = np.array(img, dtype=np.float32)
    # plt.figaspect(1)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    #
    # snr1 = estimateSNR(img, 1, True)
    # snr2 = estimateSNR(img, 2, True)
    # snr3 = estimateSNR(img, 3, True)
    #





    exit()