import numpy as np

import scipy as sp

from scipy import ndimage

from scipy import signal

import matplotlib.pyplot as plt



from image_resolution import generateTestImage



# define the SNR/CNR as follows:

# dataRange/noiseStdv

#

# where:

#

# dataRange = highestHistPeakPos - lowestHistPeakPos

#        OR = 5*stdv of noise free image (if only one peak in histogram)

# noise free image created by median filtering data

#

# noiseStdv = stdv of noise only image

# lowest peak in the local stdv histogram



def getCumulativeHistProb(hist):

    prob = hist/np.sum(hist)

    cumHistProb = np.cumsum(prob) # CDF of the histogram

    return cumHistProb



def getHistPercentilePos(hist,percentile=10):

    Nb = len(hist)

    if percentile < 0. or percentile > 100.:

        raise Exception("Invalid percentile specified: %s%% (should be in [0,100])"%(percentile))

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

    pmin = 100.0/float(numBins)

    pmax = 100.0-pmin

    return np.percentile(img,(pmin,pmax))



def histogram(data,bins=256):

    return np.histogram(data,bins=bins,range=optimiseHistogramRange(data))



def calcDataRange(img,medFiltRangePx=1,plotHistChange=False):

    medFiltSizePx = 2*medFiltRangePx + 1

    imgMed = sp.ndimage.median_filter(img,medFiltSizePx)

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

    if plotStdvImg is True:

        imgStdv = np.clip(imgStdv,edges[0],edges[-1])

        plt.imshow(imgStdv,"gray")

        plt.title("Local std. dev. (kernel size = %s px)"%(stdFiltSizePx))

        plt.show()

    vals = 0.5*(edges[1:] + edges[:-1])

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



def estimateSNR(img,kernelRangePx=1,plotData=False):

    print("Using size = %s for a median filter and local std. dev. estimates:"%(2*kernelRangePx+1))

    dataRange = calcDataRange(img,kernelRangePx,plotData)

    print("data range = %s"%(dataRange))

    noiseStdv = calcNoiseStdv(img,kernelRangePx,plotData)

    print("noise stdv = %s"%(noiseStdv))

    snr = dataRange/noiseStdv

    print("SNR = %s"%(snr))

    return snr