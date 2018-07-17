import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from astropy.io import fits
from scipy import ndimage, interpolate

# Pipeline for cleaning high-res spectroscopic data 
'''
   Pipeline Functions
'''

# Raw Data 
def collectData(dataFile):
  with open(dataFile,'rb') as f:
    data = pickle.load(f)

  return data

def plotOrder(flux, wave, order=None, orderLabels=None):
  xlocs = [0, int(np.shape(flux)[1]/2), np.shape(flux)[1]-1]
  xlabs = wave[xlocs]
  xlabs = ['%.3f' % (np.round(x,3)) for x in xlabs]

  plt.figure()
  plt.imshow(flux)
  plt.ylabel('Frame')
  plt.xlabel('Wavelength')
  if order == None:
    plt.title('')
  else:
    if orderLabels == None:
      plt.title('Order: '+str(order))
    else:
      plt.title('Order: '+str(orderLabels[order]))
  plt.clim(np.percentile(flux,1),np.percentile(flux,99))
  plt.xticks(xlocs,xlabs)

  plt.show()



# Step 1: Discard Bad Data:
# #########################################
def getBounds(flux, sigma = 10, neighborhood_size=20, edge_discard=10, zeroTol=0.1):
  med_data = np.median(flux,0)
  filt = ndimage.gaussian_filter(med_data, sigma)
  grad = np.gradient(filt)

  # find maxima on right - before ramp down
  grad_max = ndimage.maximum_filter(grad,neighborhood_size)
  maxima = (grad == grad_max)
  maxima[np.isclose(grad_max,0,atol=zeroTol)] = 0
  maxima[:edge_discard]  = 0
  maxima[-edge_discard:] = 0
  last_maxima = np.where(maxima)[0][-1]

  # find minima on left - after ramp up
  grad_min = ndimage.minimum_filter(grad,neighborhood_size)
  minima = (grad == grad_min)
  minima[np.isclose(grad_min,0,atol=zeroTol)] = 0
  minima[-edge_discard:] = 0
  minima[:edge_discard]  = 0
  first_minima = np.where(minima)[0][0]

  return [first_minima,last_maxima]

def trimData(flux, wave, error, sigma = 10, neighborhood_size=20, edge_discard=10, zeroTol=0.1):
  bounds = getBounds(flux, sigma=sigma, neighborhood_size=neighborhood_size, edge_discard=edge_discard, zeroTol=zeroTol)

  bound_flux = flux[:,bounds[0]:bounds[1]]
  bound_wave = wave[bounds[0]:bounds[1]]
  bound_error = error[:,bounds[0]:bounds[1]]

  return bound_flux, bound_wave, bound_error


# Step 2: Alignment w/ highest SNR
# #########################################
def getHighestSNR(flux, error):
  snrs = np.median(flux/error,1)
  return np.argmax(snrs)


# Gives alignment fixes from crosscorrelations with highSNR spectrum
# Shift value is center(highSNR) - center(this_spec)
#  I.e. negative shift value indicates this spectra is moved right, needs to be moved left
def findPixelShifts(flux, error, interpolation_half_width = 2, 
                    peak_half_width = 1.2, 
                    upSampleFactor = 2000,
                    verbose = False,
                    xcorMode = 'full'
):
  highSNR = getHighestSNR(flux, error)

  ref_spec = flux[highSNR] - np.mean(flux[highSNR])
  auto_cor = np.correlate(ref_spec, ref_spec, xcorMode)
  zero_point = np.argmax(auto_cor)

  centers = []

  for i,spec in enumerate(flux):
    if verbose:
      if i % 500 == 0:
        print('On Spectra '+str(i)+'/'+str(len(flux)))

    xcor = np.correlate(spec-np.mean(spec), ref_spec, xcorMode)
    mid_point = np.argmax(xcor)

    #upsample the Cross Correlation Peak
    xcor_lb = mid_point - interpolation_half_width
    xcor_rb = mid_point + interpolation_half_width + 1

    peak_x = range(xcor_lb,xcor_rb)
    peak   = xcor[xcor_lb:xcor_rb]

    upSamp, upSampPeak = upSampleData(peak_x, peak, upSampleFactor)

    upSampPeakHW = int(peak_half_width / (len(peak_x) - 1) * upSampleFactor)

    center = findCenterOfPeak(upSamp, upSampPeak, upSampPeakHW)

    centers.append(center)

  return  zero_point - np.array(centers)

def upSampleData(x, y, upSampleFactor = 10):
  upSampX = np.linspace(x[0], x[-1], len(x)*upSampleFactor)
  interpolation = interpolate.splrep(x, y)
  upSampY = interpolate.splev(upSampX, interpolation)

  return upSampX, upSampY

def findCenterOfPeak(x,y, peak_half_width = 10):
  mid_point = np.argmax(y)

  left_bound  = mid_point - peak_half_width 
  right_bound = mid_point + peak_half_width + 1

  quad_fit = np.polyfit(x[left_bound:right_bound], y[left_bound:right_bound] ,2)

  center = (-quad_fit[1] / (2*quad_fit[0]))

  return center

def getSpacing(arr):
  return (arr[-1]-arr[0])/(len(arr)-1)

# Assumes shift is in pixel values
# Positive shift moves data to the right
def shiftData(x, y, shift):
  ip = interpolate.splrep(x, y)

  dx = getSpacing(x)
  shift_x = x - (dx * shift)

  interpolated = interpolate.splev(shift_x, ip)

  return interpolated

# ######################################### 

# Identifies regions of high telluric absorbtion by their local SNR
# TODO add custom windowing
def identifyBadRegions(allData, sigma=3, window_size = 100, snr_cutoff=5):
  

  masks = []
  for i in range(len(allData)):
    medSpec = np.median(allData[i],0)
    smoothed = normalize(ndimage.gaussian_filter(medSpec,sigma))
    snrs = ndimage.generic_filter(smoothed, snr, size=window_size)
    masks.append(snrs >= snr_cutoff)

  return masks

def maskData(allData, allWaves, masks=None, sigma=3, window_size=100, snr_cutoff=5):
  if masks == None:
    masks = identifyBadRegions(allData, sigma=sigma, window_size=window_size, snr_cutoff=snr_cutoff)
  

# #########################################
def sysrem(data, error, ncycles=1, aVec=None, maxIterations=200, maxErr=0.001, verbose=False):
  rawData = np.array(data,copy=True)
  data = data-np.mean(data,0)

  data = np.transpose(data)
  error = np.transpose(error)

  models = []
  aVecs  = []
  cVecs  = []

  n,m = np.shape(data)


  cycle = 0
  # Begin Cycle Loop
  while cycle < ncycles:
    if verbose:
      print('Performing Cycle '+str(cycle+1)+'.')
    iteration=0
    relChange_aVec = 10
    relChange_cVec = 10

    if aVec is None:
      aVec = np.linspace(0,1,m)

    cVec = np.zeros(n)

    while (relChange_aVec > maxErr or relChange_cVec > maxErr) and iteration < maxIterations:
      last_aVec = aVec
      last_cVec = cVec

      cVec_num = np.sum(np.nan_to_num( (data * aVec)/(error**2)),1)
      cVec_den = np.sum(np.nan_to_num( (aVec**2 / error**2)    ),1)
      cVec     = np.nan_to_num(cVec_num/cVec_den)

      aVec_num = np.sum(np.nan_to_num((data*cVec[:,np.newaxis])/(error**2)),0)
      aVec_den = np.sum(np.nan_to_num((cVec[:,np.newaxis]**2)/(error**2))  ,0)
      aVec     = np.nan_to_num(aVec_num/aVec_den)

      if np.any(np.isnan(aVec)):
        print('nans in avec')

      if np.any(np.isnan(cVec)):
        print('nans in cvec')

      relChange_aVec = np.median(np.nan_to_num(np.abs(last_aVec / aVec -1 )))
      relChange_cVec = np.median(np.nan_to_num(np.abs(last_cVec / cVec -1 )))

      iteration += 1
      if iteration == maxIterations:
        print("Warning, reached maxIteration ("+str(maxIterations)+") in cycle "+str(cycle))    

    model = np.outer(cVec,aVec)
    fullModel = np.zeros((n,m))
    if cycle == 0:
      fullModel = np.array(rawData, copy=True)

    fullModel = model

    models.append(fullModel)
    cVecs.append(cVec)
    aVecs.append(aVec)

    data = data-model
    cycle += 1
  if verbose:
    print('Done with sysrem.')

  ret =  {
    'sysrem'    : np.transpose(data),
    'models'    : np.transpose(np.array(models),(0,2,1)),
    'model'     : np.transpose(np.sum(models,0)),
    'aVecs'     : np.array(aVecs),
    'cVecs'     : np.array(cVecs)
  } 
  return ret



# simple divide by columnwise median
def crudeTelluricFixing(allData):
  scaling = np.median(np.median(allData,0),1)
  ret = [order/scaling[:,None] for order in allData]
  return ret


'''
  Misc Helper Functions
'''
def normalize(d, outRange=[0,1]):
  num = d-np.min(d)
  den = (np.max(d)-np.min(d))/(outRange[1]-outRange[0])
  return (num/den) +outRange[0]

def snr(data,ddof=1):
  return np.mean(data)/np.std(data,ddof=ddof)

''' 
  Plotting Functions
'''

def showOrders(allData, allWave, saveName=None, dataShape = None, orderLabels=None, aspect=None, figsize = (8,8), cRange=None, cmap='viridis', waveUnit = 'microns'):
  orderNum = len(allData)

  if dataShape == None:
    # Find nearest square
    sq = np.ceil(np.sqrt(orderNum))
    rows = sq
    cols = sq
  else:
    rows,cols = dataShape

  if orderLabels == None:
    orderLabels = np.arange(orderNum)+1

  plt.figure(figsize=figsize)
  for i in range(orderNum):
    data = np.array(allData[i])

    if aspect == None:
      aspect = np.ceil(np.shape(data)[1] / np.shape(data)[0])


    plt.subplot(rows,cols,i+1)
    plt.imshow(data,aspect=aspect,cmap=cmap)

    if(cRange!=None):
      plt.clim(cRange)

    plt.title('Order '+str(orderLabels[i]))

    xlocs = [0, int(np.shape(data)[1]/2), np.shape(data)[1]-1]
    xlabs = allWave[i][xlocs]
    xlabs = ['%.3f' % (np.round(x,3)) for x in xlabs]
    plt.xticks(xlocs,xlabs)
    plt.xlabel('Wavelength ('+waveUnit+')')
  plt.tight_layout()
  if saveName != None:
    plt.savefig(saveName)
  plt.show()

# Plots 1d bounds found using getBounds
def plotBounds(allData, sigma = 10, neighborhood_size=20, edge_discard = 10,zeroTol = 0.1):
  bounds = getBounds(allData, sigma=sigma, neighborhood_size=neighborhood_size, edge_discard=edge_discard, zeroTol=zeroTol)
  plt.figure(figsize=(8,(len(allData)*4)))
  for i in range(len(allData)):
      plt.subplot(len(allData),1,i+1)
      plt.plot(normalize(np.median(allData[i],0)))

      plt.plot([bounds[i][0],bounds[i][0]],[0,1],'r')
      plt.plot([bounds[i][1],bounds[i][1]],[0,1],'r')
      plt.title('Order '+ str(orderLabels[i]))
  plt.tight_layout()
  plt.show()

# Overlays image plots of data, masks
# Masks should be taken from identify bad regions
# (Masks are boolean arrays same shape as the data, true's are areas that are kept)
def plotMasks(allData, allWave, masks, saveName=None, dataShape = None, orderLabels=None, aspect=None, figsize = (8,8), cRange=None, cmap='viridis', waveUnit = 'microns', alpha=0.3, mask_color = 'r'):
  n_orders = len(allData)
  rows     = len(allData[0])

  # tile masks
  plotMasks = [np.tile(mask.astype(float), (rows,1)) for mask in masks]

  if dataShape == None:
    # Find nearest square
    sq = np.ceil(np.sqrt(n_orders))
    rows = sq
    cols = sq
  else:
    rows,cols = dataShape

  if orderLabels == None:
    orderLabels = np.arange(n_orders)+1

  plt.figure(figsize=figsize)    
  for i in range(n_orders):
    data = np.array(allData[i])
    mask = plotMasks[i]
    # Replace 0 with nan
    # have to loop b/c irregular sized array
    mask[mask==1] = np.nan

    if aspect == None:
      aspect = np.ceil(np.shape(data)[1] / np.shape(data)[0])


    plt.subplot(rows,cols,i+1)
    plt.imshow(data,aspect=aspect,cmap=cmap)

    if(cRange!=None):
      plt.clim(cRange)

    # This has to be after clim for mysterious reasons
    mask_cmap = matplotlib.colors.ListedColormap([mask_color])
    plt.imshow(mask, alpha=alpha, cmap=mask_cmap, aspect=aspect)

    plt.title('Order '+str(orderLabels[i]))

    xlocs = [0, int(np.shape(data)[1]/2), np.shape(data)[1]-1]
    xlabs = allWave[i][xlocs]
    xlabs = ['%.3f' % (np.round(x,3)) for x in xlabs]
    plt.xticks(xlocs,xlabs)
    plt.xlabel('Wavelength ('+waveUnit+')')
  plt.tight_layout()
  if saveName != None:
    plt.savefig(saveName)
  plt.show()


