import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from astropy.io import fits
from scipy import ndimage
from numpy.polynomial.polynomial import polyval


dataRange = list(range(105,115+1)) + list(range(525,535+1)) +  list(range(885,895+1)) +  list(range(1230,1240+1)) +  list(range(1580,1590)) + list(range(2125,2130)) + list(range(2131,2135+1)) +  list(range(2845,2855+1)) +  list(range(3200,3210+1)) +  list(range(3610,3620+1)) + list(range(3814,3824+1))

file_dir = '/Users/nicholasmehrle/documents/science/spectroscopy/2016oct16/proc/'
file_pre = 'spec_'
file_pos = 'sint.fits'

apNum = 19
apRange = range(apNum-1,-1,-1)
orderLabels = list(np.arange(1,10,1)) + list(np.arange(13,23,1))

# Pipeline for cleaning high-res spectroscopic data 


'''
   Pipeline Functions
'''
# Reads data in from file
def collectData():
  allData = []
  allWave = []
  allSig  = []

  apRange = range(apNum)
  for i in apRange:
      allData.append([])
      allSig.append([])
      
  for i,dataFile in enumerate(dataRange):
      filename = file_dir+file_pre+str(dataFile).zfill(4)+file_pos
      data = fits.getdata(filename)
      for ap in apRange:
          if i==0:
              allWave.append(data[4][ap]/10000)
          spec = data[0][ap]
          spec[spec<=0] = 0
          allData[ap].append(spec)

          sigmas = data[3][ap]
          allSig[ap].append(sigmas)

  return np.array(allData), np.array(allWave), np.array(allSig)

# Step 1: Alignment w/ highest SNR
# #########################################
def getHighestSNR(allData, allSig):
  point_snrs = np.nan_to_num(allData/allSig)
  spec_snrs  = np.median(point_snrs,2)
  time_snrs  = np.median(spec_snrs,0)

  return np.argmax(time_snrs)


# Gives alignment fixes from crosscorrelations with highSNR spectrum
def findShifts(allData, order, highSNR, neighborhood_size=20, peak_width_frac = 1/2, peak_half_width=None):
  n = np.shape(allData)[2]

  # Select this order, de mean
  data = allData[order]
  data = [spec-np.mean(spec) for spec in data]

  ref_spec = data[highSNR]
  shifts = []

  for this_data in data:
    xcor = np.correlate(this_data,ref_spec,'full')

    rough_center = np.argmax(xcor)
    xcor_min     = ndimage.minimum_filter(xcor,neighborhood_size)


    if peak_half_width==None:
      m1 = np.where(xcor[:rough_center] == xcor_min[:rough_center])[0][-1]
      m2 = np.where(xcor[rough_center:] == xcor_min[rough_center:])[0][0] + rough_center

      baseline = (xcor[m1] + xcor[m2])/ 2

      peak_min = baseline + (np.max(xcor) - baseline) * (peak_width_frac)

      left_bound = rough_center
      for i in np.arange(rough_center,m1,-1):
        if (xcor[i] <= peak_min):
          left_bound = i+1
          break

      right_bound = rough_center
      for i in np.arange(rough_center,m2,1):
        if (xcor[i] <= peak_min):
          right_bound = i
          break
    else:
      left_bound  = rough_center - peak_half_width
      right_bound = rough_center + peak_half_width + 1

    peak = xcor[left_bound:right_bound]
    peak_x = np.arange(left_bound,right_bound)

    quad_p = np.polyfit(peak_x, peak, 2)

    this_shift = (-quad_p[1] / (2*quad_p[0])) - (n-1)
    shifts.append(this_shift)

  return np.array(shifts)
# ######################################### 

# Step 2: Extracting Bad Regions
# ########################################
# Finds Bounds of data ->
# Bounds off sections of increase/decrease on either end
def getBounds(allData, sigma = 3, neighborhood_size=20, edge_discard = 1,zeroTol = 1e-8):
  # higher sigma = tighter bounds
  bounds = []

  for i in range(len(allData)):
    data = np.median(allData[i],0)

    filt = ndimage.gaussian_filter(data, sigma)
    grad = np.gradient(filt)

    # find maxima on left - ramp up point
    grad_max = ndimage.maximum_filter(grad,neighborhood_size)
    maxima = (grad == grad_max)
    maxima[np.isclose(grad_max,0,atol=zeroTol)] = 0
    maxima[:edge_discard] = 0
    first_maxima = np.where(maxima)[0][0]

    # find minima on right - ramp down point
    grad_min = ndimage.minimum_filter(grad,neighborhood_size)
    minima = (grad == grad_min)
    minima[np.isclose(grad_min,0,atol=zeroTol)] = 0
    minima[-edge_discard:] = 0
    last_minima = np.where(minima)[0][-1]
    bounds.append([first_maxima,last_minima])
  return bounds

# Trims data to fit bounds
def trimOrders(allData, allWave, sigma = 3, neighborhood_size=20, edge_discard = 1,zeroTol = 1e-8):
  bounds = getBounds(allData, sigma=sigma, neighborhood_size=neighborhood_size, edge_discard=edge_discard, zeroTol=zeroTol)
  n_orders = len(allData)

  boundData = []
  boundWave = []
  for i in range(n_orders):
      this_order   = np.array(allData[i])
      wave_order   = np.array(allWave[i])
      these_bounds = bounds[i]
      data = this_order[:,these_bounds[0]:these_bounds[1]]
      wave = wave_order[these_bounds[0]:these_bounds[1]]
      
      boundData.append(data)
      boundWave.append(wave)

  return boundData, boundWave


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


# def sysrem(data, error, ncycles=1, aVec=None, maxIterations=100, maxErr=0.001):
  # print np.shape(data)




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
def plotBounds(allData, sigma=20, whichSpec = None ):
  bounds = getBounds(allData,sigma=sigma)
  plt.figure(figsize=(8,(len(allData)*4)))
  for i in range(len(allData)):
      plt.subplot(len(allData),1,i+1)
      if whichSpec==None:
        plt.plot(normalize(np.median(allData[i],0)))
      else:
        plt.plot(normalize(allData[i][whichSpec]))
      plt.plot([bounds[i][0],bounds[i][0]],[0,1],'r')
      plt.plot([bounds[i][1],bounds[i][1]],[0,1],'r')
      plt.title('Order '+ str(orderLabels[i]))
  plt.tight_layout()
  plt.show()

# Overlays image plots of data, masks
# Masks should be taken from identify bad regions
# (Masks are boolean arrays same shape as the data, true's are areas that are kept)
# TODO Cmap
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


