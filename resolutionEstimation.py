import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import signal
import matplotlib.pyplot as plt
import concurrent.futures

def getRadialGrid(Ny,Nx):
    y = (np.arange(Ny) - Ny // 2).reshape((Ny, 1))
    x = (np.arange(Nx) - Nx // 2).reshape((1, Nx))
    return np.sqrt(y*y+x*x)

def getMaxRadius(Ny,Nx):
    return np.minimum(Ny,Nx)//2

def apodiseImageGauss(img): # apply Gaussian apodisation window function
    Ny,Nx = img.shape
    sigma = 0.4*getMaxRadius(Ny,Nx)
    rad = getRadialGrid(Ny,Nx)
    apodisationWindow = np.exp((-rad*rad)/(2.0*sigma*sigma))
    return img*apodisationWindow

def apodiseImage(img): # apply cosine apodisation window function
    Ny,Nx = img.shape
    y = 0.5 - 0.5*np.cos(10.0*np.pi*np.arange(Ny)/float(Ny))
    y[Ny//10:Ny-Ny//10] = 1.0
    x = 0.5 - 0.5*np.cos(10.0*np.pi*np.arange(Nx)/float(Nx))
    x[Nx//10:Nx-Nx//10] = 1.0
    apodisationWindow = y.reshape((Ny,1))*x.reshape((1,Nx))
    return img*apodisationWindow

def pearsonCrossCorr(imga,imgb): # Pearson cross correlation (equation 1 without mask)
    pccNum = np.real(np.sum(imga*np.conjugate(imgb)))
    denoma = np.real(np.sum(imga*np.conjugate(imga)))
    denomb = np.real(np.sum(imgb * np.conjugate(imgb)))
    pccDenom = np.sqrt(denoma*denomb)
    return pccNum/pccDenom

def pearsonCrossCorrMasked(imga,imgb,maskRadNorm=None): # Pearson cross correlation with mask (equation 1)
    if np.abs(maskRadNorm)<0.00001:
        return 0.
    Ny,Nx = imga.shape
    if maskRadNorm is None:
        mask = np.ones((Ny,Nx),dtype=float)
    elif maskRadNorm>=0. and maskRadNorm<=1.0:
        rPx = float(getMaxRadius(Ny,Nx))*maskRadNorm
        mask = getRadialGrid(Ny,Nx)<rPx
    else:
        raise Exception("radius specified %s is in pixels? radius specified should be normalised, "
                        "i.e., a fraction of the image pixel radius, i.e., in [0,1]"% maskRadNorm)
    return pearsonCrossCorr(imga,imgb*mask)

def decorr(img):
    return decorrMasked(img)

def decorrMasked(img,maskRadNorm=None):
    apod = apodiseImage(img-np.mean(img)) # this is not used in this function
    ft = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img)))
    reg = 0.01*np.mean(np.abs(ft))
    phase = ft / (np.abs(ft) + reg)
    pcc = pearsonCrossCorrMasked(ft,phase,maskRadNorm)
    return pcc

def getDecorrCurve(img,rMinPx=7.5):
    Ny,Nx = img.shape
    rMaxPx = getMaxRadius(Ny,Nx)
    d = np.zeros(rMaxPx)
    for rc in range(rMaxPx):
        rNorm = rc/float(rMaxPx)
        scale = np.minimum(rc/float(rMinPx),1.0)
        d[rc] = scale*decorrMasked(img,rNorm)
    return d # decorrelation with radius as index

def getSigmaList(sMinPx,sMaxPx,Ng=10):
    # want list from sMin to sMax pixels
    sExpMax = np.log(sMaxPx)
    sExpMin = sOffset = np.log(sMinPx)
    sExpStride = (sExpMax-sExpMin)/float(Ng)
    sExpVals = np.arange(Ng)*sExpStride + sOffset
    return np.exp(sExpVals)

def getDecorrMax(d):
    ri = np.argmax(d)
    Ai = d[ri]
    return ri,Ai

def getDecorrLocalMax(d,prominence=0.01,width=10, distance=10):
    rPeaks,_ = sp.signal.find_peaks(d, prominence=prominence, width=width, distance=distance)
    if rPeaks.size:
        dPeaks = d[rPeaks]
        p = np.argmax(dPeaks)
        ri = rPeaks[p]
        Ai = dPeaks[p] # peak decorrelation
    else:
        ri = 1
        Ai = 0.
    return ri,Ai

def unsharpMask(img,sigma):
    return img - sp.ndimage.gaussian_filter(img,sigma)

def performDecorrScan(img,sList,maxVal,maxPos,maxSig,geometricMax=False,plot=False,title=None,
                      prominence=0.01,width=10, distance=10, verbose=False):
    # max = np.max(d)
    # OR
    # geoMax = np.max(r*d)
    Ny,Nx = img.shape
    rMaxPx = getMaxRadius(Ny,Nx)
    Ng = len(sList)

    def process_sigma(sc):
        sigma = sList[sc]
        imgUnsharp = unsharpMask(img, sigma)
        di = getDecorrCurve(imgUnsharp)
        rPxi, Ai = getDecorrLocalMax(di, prominence=prominence, width=width, distance=distance)
        currVal = rPxi
        if geometricMax:
            currVal *= Ai
        return currVal, rPxi, Ai, sigma, di

    with concurrent.futures.ThreadPoolExecutor() as executor: # parallel processing to speed up
        results = list(executor.map(process_sigma, range(Ng)))

    for currVal, rPxi, Ai, sigma, di in results:
        if verbose:
            print("r: %3d | A: %10f | Ar: %10f | sigmaPx: %0.2f" % (rPxi, Ai, Ai * rPxi, sigma))
        if currVal > maxVal:
            maxVal = currVal
            maxPos = rPxi
            maxSig = np.where(sList == sigma)[0][0]
        if plot:
            rNorm = np.arange(len(di)) / rMaxPx
            plt.plot(rNorm, di, label="decorr sigPx=%0.2f" % (sigma))
            if Ai > 0.:
                plt.plot(rPxi / rMaxPx, Ai, "x")
    if plot:
        if title is not None:
            plt.title(title)
        plt.xlabel("Normalised frequency")
        plt.ylabel("Pearson cross corr.")
        plt.legend(loc='upper right')
        plt.show()
    return maxVal, maxPos, maxSig



def findImageRes(img, pxSzMm=1.0, Ng=10, geometricMax=False, plot=True,prominence=0.01,width=10, distance=10,
                 verbose=False, cropFactor=0):
    # cropFactor: fraction of the smallest dimension of the image size to crop off the edges.
    # a non-square image will be cropped to a non-square image

    # max = np.max(d)
    # OR
    # geoMax = np.max(r*d)
    if 0.5 > cropFactor > 0:
        crop_size = int(cropFactor * np.min(img.shape))
        img = img[crop_size:-crop_size, crop_size:-crop_size]
    elif cropFactor == 0:
        img = img
    else:
        raise Exception("cropFactor should between 0 and 0.5")

    Ny,Nx = img.shape
    rMaxPx = getMaxRadius(Ny,Nx)

    #first pass no sharpening
    d0 = getDecorrCurve(img)
    rPx0,A0 = getDecorrLocalMax(d0, prominence=prominence, width=width, distance=distance)
    if verbose:
        print("r: %3d | A: %10f | Ar: %10f"%(rPx0,A0,A0*rPx0))
    maxVal = rPx0
    if geometricMax:
        maxVal *= A0
    maxPos = rPx0
    maxSig = 0.

    rNorm = np.arange(len(d0))/rMaxPx

    if plot:
        plt.plot(rNorm,d0,label="decorr init.")
        plt.plot(rPx0/rMaxPx,A0,"x")
        plt.xlabel("Normalised frequency")
        plt.ylabel("Pearson cross corr.")
        plt.show()

    # repeat Ng times with sigma in sList
    sMaxPx = 2.*rMaxPx/rPx0
    sMinPx = 1.0/2.355
    sList = getSigmaList(sMinPx,sMaxPx,Ng)
    maxVal,maxPos,maxSig = performDecorrScan(img, sList, maxVal, maxPos, maxSig, geometricMax, plot=plot,
                                             title="Coarse Scan",prominence=prominence,width=width, distance=distance,
                                             verbose=verbose)

    # refine for sigma around maxSig
    if maxSig > 0:
        sMinPx = sList[maxSig-1]
        sMaxPx = sList[maxSig]
    else:
        sMinPx = 0.15
        sMaxPx = sList[0]
    sList = getSigmaList(sMinPx,sMaxPx,Ng)
    maxVal,maxPos,maxSig = performDecorrScan(img, sList, maxVal, maxPos, maxSig, geometricMax, plot=plot,
                                             title="Refinement Scan", prominence=prominence,width=width,
                                             distance=distance, verbose=verbose)

    # return image resolution
    resPx = 2.0*rMaxPx/maxPos
    return pxSzMm*resPx, resPx

def generateTestImage(N,pxSzMm=1.0,resMm=3.5,numPhotonsPerPx=1000.):
    img = np.ones((N,N),dtype=np.float32)
    loc = N//20
    lic = N//9
    ric = N - lic
    roc = N - loc
    # add contanier walls
    img[:,loc:lic] = 0.5
    img[:,ric:roc] = 0.5
    # add (N/10)*(N/10) random spheres
    nSpheres = (N//30)**2
    for sc in range(nSpheres):
        r = np.random.uniform(N/40,N/20)
        cy = np.random.uniform(N/20,N-N/20)
        cx = np.random.uniform(lic+N/20,ric-N/20)
        trans = np.random.normal(0.2,0.04)
        y = (np.arange(N)-cy).reshape((N,1))
        x = (np.arange(N)-cx).reshape((1,N))
        rad = np.sqrt(x*x + y*y)
        circ = 1.0-trans*(rad<=r)
        img *= circ
    # define resPx as FWHM = 2.355*sigmaPx, therefore:
    resPx = resMm/pxSzMm # resolution in pixels
    sigmaPx = resPx/2.355
    # input image already has resPx = 2, so adjust sigmaPx:
    sigmaPx = np.sqrt(sigmaPx**2 - (2.0/2.355)**2) # why in this form?
    # perform Gaussian blurring
    img = sp.ndimage.gaussian_filter(img,sigmaPx)
    # add Poisson noise
    img = np.random.poisson(img*numPhotonsPerPx)/numPhotonsPerPx
    ## linearise image
    img = -np.log(img)
    return img


#
# import netCDF4 as nc
#
# def tomoSliceRes(nc_file, pxSzMm=1.0, Ng=8, geometricMax=False, plot=True, crop=False):
#     tomoSlice = nc.Dataset(nc_file)
#     tomoData = np.array(tomoSlice.variables['tomo'][:], dtype=np.float32, copy=True)
#
#     # Determine the dimension with length 1
#     data_dim = np.argmin(tomoData.shape)
#
#     # Reshape the data to have the data dimension last
#     tomoData = np.moveaxis(tomoData, data_dim, -1)
#
#     # have problem with space frequency calculation
#     if crop:
#         # Crop the data to the center 0.7*edge length square
#         edge_length = np.min(tomoData.shape[:2])
#         crop_size = int(0.7 * edge_length)
#         start_y = (tomoData.shape[0] - crop_size) // 2
#         start_x = (tomoData.shape[1] - crop_size) // 2
#         tomoDataFiltered = tomoData[start_y:start_y + crop_size, start_x:start_x + crop_size]
#
#         res = findImageRes(np.squeeze(tomoDataFiltered), pxSzMm, Ng, geometricMax, plot)
#     else:
#         res = findImageRes(np.squeeze(tomoData), pxSzMm, Ng, geometricMax, plot)
#
#     return res


if __name__ == "__main__":
    #img = sp.misc.ascent().astype(np.float32)
    img = generateTestImage(512)
    Ny,Nx = img.shape
    plt.imshow(img,"gray")
    plt.colorbar()
    plt.show()
    res, resPx = findImageRes(img)
    print(res)

    exit()