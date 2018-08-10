import numpy as np
import pickle, json
from scipy import ndimage as ndi
from scipy import constants, signal, stats, interpolate, optimize
from astropy.io import fits
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from barycorr import barycorr

def type_of_script():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'

if type_of_script() == 'jupyter':
  from tqdm import tqdm_notebook as tqdm
else:
  from tqdm import tqdm

# Sections:
  # Composite Functions - Wraps everything Together
  # Plotting Functions
  # Step 0: Raw Data
  # Step 1: Removing Bad Data
  # Step 2: Aligns Data
  # Step 3: Remove Coherent Structure
  # Step 4: Comparing to Template
  # Template Functions - Getting Template, Injecting fake signal
  # Math - Generic Math Functions
  # Physics - Generic Physics functions

#-- Composite Functions
def collectData(date, order, dataFileParams,
                #Bad Data Kws
                discard_rows = [],
                discard_cols = [],
                doAutoTrimCols = True,
                trim_neighborhood_size = 20,
                doAlign = True,
                alignmentIterations = 3,
                padLen = 50, peak_half_width = 3,
                upSampleFactor = 1000,
                plotCuts = False,
                obsname = None, raunits = None,
                verbose = False,  **kwargs
):
  """ #Performs Steps 0-2 all at once
  """
  #Load Raw Data
  if verbose:
    print('Collecting Data')
  data, headers, templateData = collectRawData(date, order,
    **dataFileParams)
  
  flux = data['fluxes']
  wave = data['waves']
  error = data['errors']
  del data

  times = np.array(headers['JD'])
  ras   = np.array(headers['RA'])
  decs  = np.array(headers['DEC'])
  del headers

  if verbose:
    print('Trimming Data')

  applyRowCuts  = [times, ras, decs]
  applyColCuts  = [wave]
  applyBothCuts = [error]

  flux, applyRowCuts, applyColCuts, applyBothCuts = \
      applyDataCuts(flux, rowCuts = discard_rows, colCuts = discard_cols,
        doColEdgeFind = doAutoTrimCols, applyRowCuts=applyRowCuts,
        applyColCuts=applyColCuts, applyBothCuts=applyBothCuts,
        neighborhood_size=trim_neighborhood_size, showPlots=plotCuts,
        figTitle = 'Date: '+date+', Order: '+str(order))

  times, ras, decs = applyRowCuts
  wave  = applyColCuts[0]
  error = applyBothCuts[0]

  rv_params = {
    'doBarycentricCorrect':True,
    'ra'      : ras,
    'dec'     : decs,
    'times'   : times,
    'raunits' : raunits,
    'obsname' : obsname
  }  

  if doAlign:
    if verbose:
      print('Aligning Data')
    highSNR = getHighestSNR(flux,error)
    ref = flux[highSNR]

    flux, error = alignment(flux, ref, iterations=alignmentIterations,
      error = error, padLen = padLen,
      peak_half_width = peak_half_width,
      upSampleFactor = upSampleFactor, verbose = verbose>1)
    
  template = interpolateTemplate(templateData, wave)

  if verbose:
    print('Done')
  return flux, error, wave, template, rv_params

def prepareData(flux,
              #Contunuum Params:
                continuum_order = 4,
              #Masking Params:
                use_time_mask = True, time_mask_cutoffs = [3,0],
                use_wave_mask = False, wave_mask_window = 100,
                wave_mask_cutoffs = [3,0],
                mask_smoothing_factor = 40,
              #Sysrem Params:
                sysremIterations = 0, error = None,
                returnAllSysrem = False,
              #Variance Weighting Params:
                doVarianceWeight = True,
              #Plotting Options:
                plotTimeMask = False, plotWaveMask = False,
                plotMask = False,
                verbose=False, **kwargs
):
  superVerbose = verbose>1

  # Calculate Mask for flux (To be done before continuum subtracting)
  if use_time_mask or use_wave_mask:
    if verbose:
      print("Creating Mask")

    time_mask = np.ones(np.shape(flux)[1])
    wave_mask = np.ones(np.shape(flux)[1])

    if use_time_mask:
      time_mask = getTimeMask(flux, *time_mask_cutoffs,
          smoothingFactor=0, showPlots = plotTimeMask)
    
    if use_wave_mask:
      wave_mask = getWaveMask(flux, wave_mask_window, *wave_mask_cutoffs, smoothingFactor=0, showPlots = plotWaveMask)

    mask = combineMasks(time_mask, wave_mask, 
        smoothingFactor=mask_smoothing_factor)

    if plotMask:
      plt.figure()
      plt.title('Full Mask')
      plt.plot(normalize(np.median(flux,0)))
      plt.plot(mask)
      plt.ylim(-0.2,1.2)
      plt.show()

  # Continuum Subtract Flux
  if continuum_order != 0:
    if verbose:
      print("Subtracting Continuum")

    flux = continuumSubtract(flux, continuum_order, verbose=superVerbose)

  # Apply mask to flux
  if use_time_mask or use_wave_mask:
    flux = applyMask(flux, mask)

  if sysremIterations != 0:
    if verbose:
      print('Doing Sysrem')
    flux = sysrem(flux, error, sysremIterations, verbose=superVerbose,
      retAll = returnAllSysrem)

  if doVarianceWeight:
    if verbose:
      print('Variance Weighting Columns')
    flux = varianceWeighting(flux)

  return flux

def calcSysremIterations(date, order, dataFileParams, orb_params,
                        fake_signal_strengths=[1/1000],
                        maxIterations=10,
                        verbose=False, **kwargs
):
  """ Computes detection strength vs number of sysrem iterations
    Params: See collectData(), prepareData(), generateSmudgePlot()
    for additional optional parameters
  """

  # Collect Data
  flux, error, wave, template, rv_params = collectData(date, order,
              dataFileParams, verbose=verbose, **kwargs)

  # Injected KpValue
  kpRange = np.array([orb_params['Kp']])
  all_detection_strengths = []

  for signal_strength in fake_signal_strengths:
    if verbose:
      print('-----------------------------')
      print('Working on signal: '+str(signal_strength))
      print('-----------------------------')
    this_ds = []

    fake_signal = injectFakeSignal(flux, wave, rv_params, orb_params, signal_strength, dataFileParams=dataFileParams, verbose=verbose-1)

    # Calculate maxIterations sysrem iterations
    sysremData = prepareData(fake_signal,
                    sysremIterations=maxIterations, error=error,
                    returnAllSysrem=True, verbose=verbose, **kwargs)

    if verbose:
        print('Calculating Detection Strengths')
    # Calculate the detection strength for each iteration
    for residuals in sysremData:
      smudges, vsys_axis, kp_axis = generateSmudgePlot(residuals, wave,
                                      rv_params, template, kpRange,
                                      orb_params, retAxes=True,
                                      verbose=(verbose>1)*2,
                                      **kwargs)

      x_pos = np.argmin(np.abs(vsys_axis - orb_params['v_sys']))
      detection_strength = smudges[0,x_pos]
      this_ds.append(detection_strength)

    all_detection_strengths.append(np.array(this_ds))

  if verbose:
    print('Done!')
  return np.array(all_detection_strengths)

def pipeline(date, order, dataFileParams, orb_params,
             kpRange, verbose=False, **kwargs
):
  """ Completely generates a smudge plot for the given order, data
    See collectData(), prepareData(), generateSmudgePlot() for 
    optional parameter inputs
  """

  # Collect Raw Data
  flux, error, wave, template, rv_params = collectData(date, order,
              dataFileParams, verbose=verbose, **kwargs)

  data = prepareData(flux, wave=wave, rv_params=rv_params,
          orb_params=orb_params,
          error=error, verbose=verbose,
          **kwargs)

  smudges, vsys_axis, kp_axis = generateSmudgePlot(data, wave, rv_params,
                                  template, kpRange, orb_params,
                                  verbose=verbose, retAxes=True,
                                  **kwargs)

  if verbose:
    print('Done!')
  return smudges, vsys_axis, kp_axis 

def pipelineDate(date, orders, planet,
        dataFileParams,
        kpRange=None, full_vsys=None, vsys_range=None,
        normalizeCombined = True, sysremIterations=None,
        obsname='mmto', raunits='hours', verbose=False, **user_kwargs
):
  orb_params = readOrbParams(planet)
  smudges = []
  x_axes  = []
  y_axes  = []
  
  seq = orders
  if verbose:
    seq = tqdm(orders, desc='Pipelining:')

  for order in seq:
    kwargs = getKwargs(date, order)
    if sysremIterations is not None:
      kwargs['sysremIterations'] = sysremIterations

    kwargs.update(user_kwargs)

    sm,rx,ry = pipeline(date, order, dataFileParams, orb_params, kpRange,
                   obsname=obsname, raunits=raunits,
                   vsys_range=vsys_range, stdDivide=False,
                   verbose=(verbose-1), **kwargs)
    smudges.append(sm)
    x_axes.append(rx)
    y_axes.append(ry)

  combinedSmudge = combineSmudges(smudges, x_axes, full_vsys,
                    normalize=normalizeCombined)
  return combinedSmudge
###

#-- Plotting Functions
def plotOrder(flux, wave, order=None, orderLabels=None, cmap='viridis'):
  xlocs = [0, int(np.shape(flux)[1]/2), np.shape(flux)[1]-1]
  xlabs = wave[xlocs]
  xlabs = ['%.3f' % (np.round(x,3)) for x in xlabs]

  plt.figure()
  plt.imshow(flux,aspect='auto',cmap=cmap)
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

def plotSmudge(smudges, vsys_axis, kp_axis,
              orb_params=None, titleStr="",
              saveName = None, cmap='viridis',
              xlim=None, ylim=None
):
  """ Plots a smudge Plot
  """

  # Use units: km/s
  xs = vsys_axis/1000
  ys = kp_axis/1000

  # Offset xs,ys by 1/2 spacing for pcolormesh
  pltXs = xs+getSpacing(xs)/2
  pltYs = ys+getSpacing(ys)/2

  # Get Max of SmudgePlot
  ptmax = np.unravel_index(smudges.argmax(), smudges.shape)
  ptx = xs[ptmax[1]]
  pty = ys[ptmax[0]]

  plt.figure()
  # Plot smudges
  plt.pcolormesh(pltXs,pltYs,smudges,cmap=cmap)
  cbar = plt.colorbar()

  # Plot Peak
  plt.scatter(ptx,pty, color='k')

  # Plot 'true' values if they exist
  true_val_str = ""
  if orb_params is not None:
    trueKp   = orb_params['Kp']/1000
    trueVsys = orb_params['v_sys']/1000
    plt.plot((trueVsys,trueVsys),(pltYs[0],pltYs[-1]),'r')
    plt.plot((pltXs[0],pltXs[-1]),(trueKp,trueKp),'r')

    markYval = np.argmin(np.abs(ys - trueKp))
    markXval = np.argmin(np.abs(xs - trueVsys))

    true_val_str = "\nValue under cross: " + str(np.round(smudges[markYval,markXval],2))

  plt.ylabel("Kp (km/s)")
  plt.xlabel("V_sys (km/s)")

  if titleStr!="":
    titleStr += '\n'
  plt.title(titleStr+'Max Value: '+ 
      str(np.round(smudges[ptmax],2)) + ': (' + str(np.round(ptx,1)) + ',' + str(int(np.round(pty,0))) + ')' +
      true_val_str)

  cbar.set_label('Sigma')

  if xlim is not None:
    plt.xlim(*xlim)
  if ylim is not None:
    plt.ylim(*ylim)

  plt.tight_layout()
  if saveName is not None:
    plt.savefig(saveName)

  plt.show()
###

#-- Step 0: Raw Data
def collectRawData(date, order, 
                   dirPre, dirPos, dataPre, dataPos,
                   header, templateDir, templateName):
  dataDir      = dirPre+date+dirPos
  dataFile     = dataDir+dataPre+str(order)+dataPos
  headerFile   = dataDir+header
  templateFile = templateDir+templateName

  data = readFile(dataFile)
  header = readFile(headerFile)
  templateData = readFile(templateFile)

  return data, header, templateData

def readOrbParams(planet, orbParamsDir='./', 
                  orbParamsFile='orb_params.json'
):
  """ Reads In orbital parameters from the database
  """
  fileStr = orbParamsDir + orbParamsFile
  with open(fileStr) as f:
    data = json.load(f)

  try:
    return data[planet]
  except KeyError:
    print('Planet "'+str(planet)+'" not found')
    print('Valid Planets Are: ')
    print(list(data.keys()))
    raise

def getKwargs(date, order):
  if date == '2016oct15b':
    date_kwargs = {
      'default': {  
        'time_mask_cutoffs' : [2.5,0],
        'sysremIterations'  : 5
      },
    }
  elif date == '2016oct16':
    date_kwargs = {
      'default': {
        'discard_rows' : [-1],
        'sysremIterations': 5,
      },
    }
  elif date == '2016oct19':
    date_kwargs = {
      'default': {
        'discard_rows' : [-1],
        'sysremIterations': 6,
      },
    }
  elif date == '2016oct20b':
    date_kwargs = {
      'default': {
        'discard_rows':[61],
        'sysremIterations': 5,
      },
    }
  else:
    date_kwargs = {}

  try:
    kwargs = date_kwargs[order]
  except KeyError:
    kwargs = {}
  try:
    kwargs.update(date_kwargs['default'])
  except KeyError:
    pass

  return kwargs
###

#-- Step 1: Delete bad data
def getEdgeCuts(flux, neighborhood_size=30,
                showPlots=False, ax = None
):
  col_snr = np.nan_to_num(np.apply_along_axis(snr,0,flux))
  col_snr = col_snr - np.mean(col_snr)
  smooth = ndi.minimum_filter(col_snr,neighborhood_size)

  n = len(smooth)
  step = np.concatenate((np.ones(n),-1*np.ones(n)))

  xcor = np.correlate(smooth, step, 'valid')
  
  # Want maxima on right -> Step Down
  # Want minima on left  -> Step Up
  xcorMinima = getLocalMinima(xcor, neighborhood_size)
  xcorMaxima = getLocalMaxima(xcor, neighborhood_size)
  
  left_bound  = xcorMinima[0]
  right_bound = xcorMaxima[-1]
      
  if showPlots:
    if ax is None:
      ax = plt.gca()
    norm_snr    = normalize(col_snr)
    norm_smooth = normalize(smooth)
    ax.plot(norm_snr-np.median(norm_snr),label='Column SNR')
    ax.plot(norm_smooth - np.median(norm_smooth),label='Minimum Filter')
    ax.plot(normalize(xcor, (-0.5,0)),label='Cross Correlation and Extrema')
    ax.plot(normalize(np.median(flux,0),(-1,-0.5)), label='Median Flux')
    ax.plot((left_bound,left_bound),(-1.0,0), color='C2')
    ax.plot((right_bound,right_bound),(-1.0,0), color='C2')
    ax.legend()
    ax.set_title('Edge Trimming\nLeft: '+str(left_bound)+', Right: '+str(right_bound))
    ax.set_xlabel('Column Number')
    ax.set_ylabel('Normalized SNR')
    # ax.set_ylim(-0.2,0.2)
      
  return left_bound, right_bound


# TODO: Consider
def getEdgeCuts2(flux, gaussian_blur = 15, neighborhood_size=30,
                showPlots=False, ax = None
):
  sig = np.median(flux,0)
  smooth = ndi.gaussian_filter(sig,gaussian_blur)

  grad = np.gradient(smooth)
  # Maxima on left  -> Left Edge
  # Minima on Right -> Right Edge
  maxima = getLocalMaxima(grad,neighborhood_size)
  minima = getLocalMinima(grad, neighborhood_size)

  minima_store = minima
  minima = np.setdiff1d(minima,maxima)
  maxima = np.setdiff1d(maxima,minima_store)

  minima = minima[np.logical_not(np.isclose(grad[minima],0))]
  maxima = maxima[np.logical_not(np.isclose(grad[maxima],0))]

  leftEdge  = maxima[0]
  rightEdge = minima[-1]

  grad2 = np.gradient(grad)
  minima2 = getLocalMinima(grad2, neighborhood_size)

  rightDelta  = rightEdge - minima2
  rightCorner = rightEdge - np.min(rightDelta[rightDelta>0])

  leftDelta  = minima2 - leftEdge
  leftCorner = np.min(leftDelta[leftDelta>0]) + leftEdge
  
  left_bound  = leftCorner
  right_bound = rightCorner
      
  if showPlots:
    if ax is None:
      ax = plt.gca()
    norm_snr    = normalize(sig)
    norm_smooth = normalize(smooth)
    ax.plot(norm_snr-np.median(norm_snr),label='Column SNR')
    ax.plot(norm_smooth - np.median(norm_smooth),label='Minimum Filter')
    ax.plot((left_bound,left_bound),(-0.5,0), color='C2')
    ax.plot((right_bound,right_bound),(-0.5,0), color='C2')
    ax.legend()
    ax.set_title('Edge Trimming\nLeft: '+str(left_bound)+', Right: '+str(right_bound))
    ax.set_xlabel('Column Number')
    ax.set_ylabel('Normalized SNR')
    # ax.set_ylim(-0.2,0.2)
      
  return left_bound, right_bound

def applyDataCuts(flux, rowCuts=None, colCuts=None, doColEdgeFind=True,
                  applyRowCuts=None, applyColCuts=None,
                  applyBothCuts=None, neighborhood_size=30,
                  showPlots=False, figsize=(8,8), figTitle=""
):
  if rowCuts is not None:
    nRows, nCols = flux.shape
    rowMask = np.ones(nRows)
    rowMask[rowCuts] = 0
    rowMask = rowMask.astype(bool)

    flux = flux[rowMask,...]
    if applyRowCuts is not None:
      for i in range(len(applyRowCuts)):
        applyRowCuts[i] = applyRowCuts[i][rowMask,...]
    if applyBothCuts is not None:
      for i in range(len(applyBothCuts)):
        applyBothCuts[i] = applyBothCuts[i][rowMask,...]

  if colCuts is not None:
    colMask = np.ones(nCols)
    colMask[colCuts] = 0
    colMask = colMask.astype(bool)

    flux = flux[...,colMask]
    if applyColCuts is not None:
      for i in range(len(applyColCuts)):
        applyColCuts[i] = applyColCuts[i][...,colMask]
    if applyBothCuts is not None:
      for i in range(len(applyBothCuts)):
        applyBothCuts[i] = applyBothCuts[i][...,colMask]

  if showPlots:
    fig, axs = plt.subplots(2,1,figsize=figsize)

    fig.suptitle(figTitle, size=16)
    axs[0].set_title('Row wise SNR (mean/std) \n After hard cuts, before bounds')
    axs[0].plot(np.apply_along_axis(snr,1,flux))
    axs[0].set_xlabel('Row Number')
    axs[0].set_ylabel('SNR')
  else:
    axs = [None,None]

  if doColEdgeFind:
    leftEdge, rightEdge = getEdgeCuts(flux, neighborhood_size,
      showPlots, ax=axs[1])
    flux = flux[...,leftEdge:rightEdge]
    if applyColCuts is not None:
      for i in range(len(applyColCuts)):
        applyColCuts[i] = applyColCuts[i][...,leftEdge:rightEdge]
    if applyBothCuts is not None:
      for i in range(len(applyBothCuts)):
        applyBothCuts[i] = applyBothCuts[i][...,leftEdge:rightEdge]

  if showPlots:
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

  return flux,applyRowCuts, applyColCuts,applyBothCuts
###

#-- Step 2: Align Data
def getHighestSNR(flux, error):
  snrs = np.median(flux/error,1)
  return np.argmax(snrs)

def calcCorrelationOffset(corr, auto_corr,
              peak_half_width = 3,
              upSampleFactor  = 1000, 
              fourier_domain = False,
              verbose = False
):
  '''
    Gives alignment fixes from crosscorrelations with highSNR spectrum
    Shift value is center(highSNR) - center(this_spec)
    I.e. negative shift value indicates this spectra is moved right, needs to be moved left
  '''
  if fourier_domain:
    corr = ifftCorrelation(corr, len(auto_corr))

  zero_point = np.argmax(auto_corr)

  seq = range(len(corr))
  if verbose:
    seq = tqdm(seq, desc="Calculating Offsets")

  centers = []
  for i in seq:
    xcor = corr[i]
    mid_point = np.argmax(xcor)
    #upsample the Cross Correlation Peak
    xcor_lb = mid_point - peak_half_width
    xcor_rb = mid_point + peak_half_width + 1

    peak_x = range(xcor_lb,xcor_rb)
    peak   = xcor[xcor_lb:xcor_rb]
    # return xcor, mid_point


    upSamp, upSampPeak = upSampleData(peak_x, peak, upSampleFactor=upSampleFactor)

    center = upSamp[np.argmax(upSampPeak)]

    centers.append(center)

  return  zero_point - np.array(centers)

def alignment(flux, ref, iterations = 1, 
             error=None, returnOffset = False,
             padLen = 50,
             peak_half_width = 3, upSampleFactor = 1000,
             verbose = False
):
  if iterations <= 0:
    if error is not None:
      return flux, error
    return flux

  if verbose and not returnOffset:
    print(str(iterations) + ' alignment iterations remaining')

  m,n = np.shape(flux)
  
  row_means = np.mean(flux, 1, keepdims = True)
  flux = flux - row_means
  ref  = ref  - np.mean(ref)

  ref  = np.pad(ref,padLen,'constant')
  flux = np.pad(flux, ((0,0),(padLen,padLen)), 'constant')

  ref_autoCorr = signal.correlate(ref, ref, 'same')

  fft_ref = rfft(ref)
  fft_flux, fft_n = rfft(flux, returnPadLen=True)

  fft_corr = correlate(fft_flux, fft_ref,fourier_domain=True)
  offsets = calcCorrelationOffset(fft_corr, ref_autoCorr,
               fourier_domain = True, peak_half_width = peak_half_width,
              upSampleFactor = upSampleFactor, verbose=verbose)

  if returnOffset:
    return offsets

  fft_shifted = fourierShift2D(fft_flux, offsets, n=fft_n, 
                                fourier_domain=True)
  if error is not None:
    error = fourierShift2D(error, offsets, fourier_domain=False)

  # The truncation makes it so I have to irfft, then rfft each round
  flux = np.fft.irfft(fft_shifted)[:,padLen:n+padLen] + row_means

  return alignment(flux, ref[padLen:n+padLen], 
    iterations = iterations-1, error = error,
    padLen = padLen, peak_half_width = peak_half_width,
    upSampleFactor = upSampleFactor, verbose = verbose)
###

#-- Step 3: Remove Coherent Structure
def continuumSubtract(data, order, error=None, verbose=False):
  single_spec = (np.ndim(data) == 1)

  result = []
  x = np.arange(np.shape(data)[-1])

  seq = data
  if single_spec:
    seq = [data]
  if verbose:
    seq = tqdm(seq, desc='Subtracting Continuums')

  weights = None
  if error is not None:
    weights  = 1/error

  for spec in seq:
    spec_polyfit = np.polyfit(x, spec, order, w=weights)
    continuum = np.polyval(spec_polyfit, x)
    result.append(spec-continuum)

  if single_spec:
    result = result[0]

  return np.array(result)

def sysrem(data, error, 
          ncyles = 1,
          initialGuess = None,
          maxIterations = 200,
          maxError = 0.001,
          verbose = False,
          retAll  = True
):
  M,N = np.shape(data)
  residuals = data - np.mean(data,0)

  allResiduals = [residuals]

  if initialGuess == None:
      initialGuess = np.ones(M)

  invErrorSq = 1/(error**2)

  aVec = initialGuess
  cVec = np.ones(N)

  if verbose:
    print('Starting Sysrem')

  for cycle in range(ncyles):
    if verbose:
      print('Starting Cycle '+str(cycle+1),flush=True)
      pbar = tqdm(total=100, desc='Cycle '+str(cycle+1))

    aVecError = maxError*10
    cVecError = maxError*10

    iterations = 0
    while iterations <= maxIterations and (aVecError >= maxError or cVecError >= maxError):
      last_aVec = aVec
      last_cVec = cVec

      cVecNum = np.sum( (residuals * aVec[:,np.newaxis]) * invErrorSq , 0)
      cVecDen = np.sum( ((aVec**2)[:,np.newaxis])        * invErrorSq , 0)

      cVec = cVecNum/cVecDen

      aVecNum = np.sum( residuals * cVec * invErrorSq ,1)
      aVecDen = np.sum( cVec**2          * invErrorSq ,1)

      aVec = aVecNum/aVecDen

      aVecError = np.median(np.nan_to_num(np.abs(last_aVec / aVec - 1 )))
      cVecError = np.median(np.nan_to_num(np.abs(last_cVec / cVec - 1 )))

      if verbose:
        largestError = np.max((aVecError, cVecError))
        convergence = 1/(np.log(largestError/maxError)+1)
        if largestError <= maxError:
          convergence = 1

        pbarVal = int(convergence*100)
        pbar.update(max(pbarVal-pbar.n,0))
        

      iterations += 1

    thisModel = np.outer(aVec,cVec)
    residuals = residuals - thisModel

    allResiduals.append(residuals)

  if retAll:
    return np.array(allResiduals)
  else:
    return allResiduals[-1]
      
def varianceWeighting(data):
  # For data.ndim = 2: gets column variance
  # For data.ndim = 3: gets column variance of each image
  return np.nan_to_num(data/np.var(data,-2, keepdims=1))

def getTimeMask(flux, relativeCutoff = 3, absoluteCutoff = 0,
                smoothingFactor = 20,
                showPlots = False
):
  weights  = np.nan_to_num(np.apply_along_axis(snr, 0, flux))

  if np.any(weights < 0):
    print('Warning, some weights (SNRs) are less than zero, consider using non zero-mean values for generating mask')

  weightMean = np.mean(weights)
  weightStd  = np.std(weights)

  lowerMask = weights < weightMean - relativeCutoff*weightStd
  absoluteMask = weights < absoluteCutoff

  mask = 1-np.logical_or(lowerMask,absoluteMask)
  mask = ndi.minimum_filter(mask, smoothingFactor)

  if showPlots:
    plt.figure(figsize=(6,8))
    plt.subplot(211)
    plt.title('Column wise SNRs')
    plt.plot(weights)
    n = len(weights)
    colors = ['C1','C2','C3','C4']
    labels = ['Mean','1 sigma','2 sigma','3 sigma']
    for i in range(-3,4):
      ls = '-'
      lab = labels[np.abs(i)]
      if i < 0:
        lab = ""
      plt.plot((0,n),(weightMean+i*weightStd,weightMean+i*weightStd),
        label=lab, linestyle=ls, color=colors[np.abs(i)])
    plt.legend(frameon=True,loc='best')
    
    plt.subplot(212)
    plt.title('Time Mask')
    plt.plot(normalize(np.median(flux,0)))
    plt.plot(mask)
    plt.ylim(-0.2,1.2)
    plt.show()

  return mask

def getWaveMask(flux, window_size=100, relativeCutoff = 3,
                  absoluteCutoff = 0, smoothingFactor = 20,
                  showPlots=False,
):
  medSpec = np.median(flux,0)
  weights = ndi.generic_filter(medSpec, snr, size=window_size)

  weightMean = np.mean(weights)
  weightStd  = np.std(weights)

  lowerMask = weights < weightMean - relativeCutoff*weightStd
  absoluteMask = weights < absoluteCutoff

  mask = 1-np.logical_or(lowerMask,absoluteMask)
  mask = ndi.minimum_filter(mask, smoothingFactor)

  if showPlots:
    plt.figure(figsize=(6,8))
    plt.subplot(211)
    plt.title('Windowed SNR along row')
    plt.plot(weights)
    n = len(weights)
    colors = ['C1','C2','C3','C4']
    labels = ['Mean','1 sigma','2 sigma','3 sigma']
    for i in range(-3,4):
      ls = '-'
      lab = labels[np.abs(i)]
      if i < 0:
        lab = ""
      plt.plot((0,n),(weightMean+i*weightStd,weightMean+i*weightStd),
        label=lab, linestyle=ls, color=colors[np.abs(i)])
    plt.legend(frameon=True)
    
    plt.subplot(212)
    plt.title('Wave Mask')
    plt.plot(normalize(np.median(flux,0)))
    plt.plot(mask)
    plt.ylim(-0.2,1.2)
    plt.show()

  return mask

def combineMasks(*masks, smoothingFactor=20):
  mask = np.prod(masks,0)
  return ndi.minimum_filter(mask, smoothingFactor)

def applyMask(data, mask):
  # Number of 'good' points remaining per row
  num_unmasked = np.sum(mask)

  # mean of row after masking (excluding 'bad' (to be masked) vals)
  new_mean = np.sum(data*mask,-1,keepdims=1)/num_unmasked

  y = data - new_mean

  return y*mask
###

#-- Step 4: Compare To Template
def generateXCorMatrix(data, wave, template, 
                        normalize=True,
                        xcorMode='same', verbose=False
):
  template = template-np.mean(template)

  seq = data
  if verbose:
    seq = tqdm(seq, desc='Cross Correlating')

  xcors = []
  for spec in seq:
    spec = spec-np.mean(spec)
    xcor = signal.correlate(spec, template, xcorMode)
    
    if normalize:
      n = len(spec)
      xcor = xcor / (n * np.std(spec)*np.std(template))

    xcors.append(xcor)

  return np.array(xcors)

def convertPixelToVelocity(pixels, wave):
  dw = pixels*getSpacing(wave)
  vels = inverseDoppler(wave, dw)
  return vels

def convertVelocityToPixel(velocity, wave):
  cw = np.median(wave)
  dp = (cw - doppler(cw, velocity))/getSpacing(wave)
  return dp

def getXcorVelocities(wave, xcorMode = 'same'):
  n = len(wave)
  if xcorMode == 'same':
    pixel_offset = np.arange(-int(n/2),int((n+1)/2))
  elif xcorMode == 'full':
    pixel_offset = np.arange(-(n-1), n)

  vel_offset = pixel_offset * inverseDoppler(wave, getSpacing(wave))
  return vel_offset

def alignXcorMatrix(xcor_interps, vsys, rvs, ext=1):
  aligned_xcm = []
  for i in range(len(rvs)):
      vels = vsys+rvs[i]     
      axc = interpolate.splev(vels, xcor_interps[i], ext=ext)
      aligned_xcm.append(axc)
  return np.array(aligned_xcm)

def initializeSmudgePlot(data, wave, rv_params, template,
                        kpRange, orb_params,
                        vsys_range = None,
                        normalizeXCors = True,
                        xcorMode = 'same',
                        verbose = False
):
  if verbose:
    print('Generating Cross Correlations')
  # xcor Interps
  xcm = generateXCorMatrix(data, wave, template, 
                          normalize=normalizeXCors, xcorMode=xcorMode,
                          verbose=(verbose>1))
  vsys = getXcorVelocities(wave, xcorMode)
  times = rv_params['times']

  xcor_interps = []
  for xcor in xcm:
    xcor_interps.append(interpolate.splrep(vsys, xcor))

  # unit RVs
  orb_params = orb_params.copy()
  orb_params['v_sys'] = 0
  orb_params['Kp']    = 1
  if rv_params['doBarycentricCorrect']:
    unitRVs, barycentricCorrection = rv(**rv_params, **orb_params, returnBCSeparately=True)
  else:
    unitRVs = rv(**rv_params, **orb_params, returnBCSeparately=False)
    barycentricCorrection = 0

  # vsys limited
  if vsys_range != None:
    allRVs = np.tile(unitRVs,(len(kpRange),1)) * kpRange[:,np.newaxis]
    min_vels = vsys + np.min((np.min(allRVs),0)) + np.min(barycentricCorrection)
    max_vels = vsys + np.max((np.max(allRVs),0)) + np.max(barycentricCorrection)
    goodCols = np.logical_and(max_vels >= vsys_range[0], min_vels  <= vsys_range[1])
    # print(min_vels,max_vels)
    vsys = vsys[goodCols]

  return xcor_interps, unitRVs, barycentricCorrection, vsys

def generateSmudgePlot(data, wave, rv_params, template,
                        kpRange, orb_params,
                        vsys_range = None,
                        normalizeXCors = True,
                        xcorMode = 'same',
                        verbose = False,
                        retAxes = True,
                        ext = 3,
                        mode=1,
                        stdDivide = True,
                        **kwargs
):
  if verbose:
    print('Initializing')

  # Initialize
  xcor_interps, unitRVs, barycentricCorrection, vsys = \
        initializeSmudgePlot(data, wave, rv_params, 
                             template, kpRange, orb_params,
                             vsys_range=vsys_range,
                             normalizeXCors=normalizeXCors,
                             xcorMode=xcorMode, verbose=verbose)
  if verbose:
    print('Considering Kps')
  #Setting up verbose iterator
  seq = range(len(kpRange))
  if (verbose>1):
    seq = tqdm(seq, desc='Considering Kps')

  # Calculate Smudges
  smudges = []
  for i in seq:
    rvs = kpRange[i] * unitRVs + barycentricCorrection
    aligned_xcm = alignXcorMatrix(xcor_interps, vsys, rvs, ext=ext)

    # return np.array(aligned_xcm)
    m = len(data)
    if mode == 0:
      return aligned_xcm, vsys
    elif mode == 1:
      smudges.append(1/m * np.sum(aligned_xcm,0))
    elif mode == 2:
      smudges.append(np.sqrt(1/m * np.sum(aligned_xcm**2,0)))
    elif mode == 3:
      mlsq = 1 - np.power(np.product(1 - aligned_xcm**2, 0),1/m)
      smudges.append(np.sqrt(mlsq))
    elif mode == 4: 
      col_means = np.mean(aligned_xcm,0)
      col_vars  = np.var(aligned_xcm,0)

      nrows, ncols = aligned_xcm.shape

      full_sum = np.sum(aligned_xcm)
      full_var = np.var(aligned_xcm)*(nrows*ncols)

      pvals = []
      for j in range(ncols):
        lower = j-1
        upper = j+2 

        if j == 0:
          lower = 0
        elif j == ncols-1:
          upper = ncols

        col_width = upper-lower

        n_inCol  = col_width * nrows
        n_outCol = (ncols-col_width) * nrows

        mean_inCol = np.mean(col_means[lower:upper])
        var_inCol  = np.mean(col_vars[lower:upper])

        mean_outCol = (full_sum - n_inCol*mean_inCol)/n_outCol
        var_outCol  = (full_var - n_inCol*var_inCol)/n_outCol

        st, p = stats.ttest_ind_from_stats(
          mean_inCol,  np.sqrt(var_inCol),  n_inCol,
          mean_outCol, np.sqrt(var_outCol), n_outCol,
          equal_var = False)

        pvals.append(p)
      smudges.append(-stats.norm.ppf(pvals))

  if stdDivide:
    smudges = smudges/np.apply_along_axis(percStd,1,smudges)[:,np.newaxis]

  if retAxes:
    return np.array(smudges), vsys, kpRange
  else:
    return np.array(smudges)

def combineSmudges(smudges, x_axes, out_x, normalize=True):
  retSmudge = []
  for i in range(len(smudges)):
    smudge = smudges[i]
    x      = x_axes[i]
    interpolatedSmudge = []
    for ccf in smudge:
      ip = interpolate.splrep(x, ccf)
      it = interpolate.splev(out_x, ip)
      interpolatedSmudge.append(it)
    retSmudge.append(np.array(interpolatedSmudge))

  retSmudge = np.sum(retSmudge,0)
  if normalize:
    retSmudge = retSmudge/np.apply_along_axis(percStd,0,retSmudge)
  return retSmudge
###

#-- Template Functions
def injectFakeSignal(flux, wave, rv_params,
                     orb_params, fake_signal_strength,
                     templateData = None, templateFile = None,
                     dataFileParams = None, verbose=False
):
  # Get Template:
  # Priority of inputs -> dataFileParams -> templatefile ->
  # templateData
  template_interp = None
  if dataFileParams is not None:
    templateFile = dataFileParams['templateDir'] + dataFileParams['templateName']
  if templateFile is not None:
    templateData = readFile(templateFile)
  if templateData is not None:
    t_wave = templateData['wavelengths']/10000
    t_flux = templateData['flux']

    template_interp = interpolate.splrep(t_wave,t_flux)
  if template_interp is None:
    raise ValueError('Must specify a template in some form.')

  if verbose:
    print('Injecting Fake Data')

  fake_signal = []
  rvs = rv(**rv_params, **orb_params)

  seq = rvs
  if verbose>1:
    seq = tqdm(seq, desc='Generating Fake Signal')

  for this_rv in seq:
    sourceWave = doppler(wave, this_rv)
    this_flux  = normalize(interpolate.splev(sourceWave, template_interp))
    fake_signal.append(this_flux)
  fake_signal = np.array(fake_signal) * np.median(flux,1)[:,np.newaxis]*fake_signal_strength
  return flux + fake_signal

def interpolateTemplate(templateData, wave):
  t_wave = templateData['wavelengths']/10000
  t_flux = templateData['flux']

  template_interp = interpolate.splrep(t_wave,t_flux)
  return interpolate.splev(wave, template_interp)
###

'''
   Generic Functions
'''
#-- File I/O
def readFile(dataFile):
  extension = dataFile.split('.')[-1]
  if extension == 'fits':
    data = fits.getdata(dataFile)
  elif extension == 'pickle':
    try:
      with open(dataFile, 'rb') as f:
        data = pickle.load(f)
    except UnicodeDecodeError:
      with open(dataFile, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
  else:
    raise ValueError('Currently only fits and pickle are supported')

  return data
###

#-- Math
def snr(data):
  return np.mean(data)/np.std(data)

def getLocalMinima(data, neighborhood_size=20):
  minima = ndi.minimum_filter(data, neighborhood_size)
  is_minima = (data == minima)
  return np.where(is_minima)[0]

def getLocalMaxima(data, neighborhood_size=20):
  maxima = ndi.maximum_filter(data, neighborhood_size)
  is_maxima = (data == maxima)
  return np.where(is_maxima)[0]

def rfft(a, pad=True, axis=-1, returnPadLen = False):
  """ Wrapper for np.fft.rfft
    Automatically pads to length base-2
  """
  n = np.shape(a)[axis]
  power = int(np.ceil(np.log2(n)))
  if pad:
    if returnPadLen:
      return np.fft.rfft(a, 2**power, axis=axis),2**power
    return np.fft.rfft(a, 2**power, axis=axis)
  else:
    return np.fft.rfft(a, axis=axis)

def correlate(target, reference, fourier_domain = False):
  """ Correlation function with option to pass data already in the 
    Fourier domain.
  """
  if not fourier_domain:
    n = len(reference)
    target = target - np.mean(target,1, keepdims=True)
    target = rfft(target)

    reference = reference - np.mean(reference)
    reference = rfft(reference)

  fft_corr = np.conj(reference) * target
  
  if not fourier_domain:
    return ifftCorrelation(fft_corr, n)

  return fft_corr
  
def ifftCorrelation(fft_corr, n=None):
  """ Inverts the correlation matrix from correlate, 
    Applies the transformation to correct for circular/non-circular ffts
  """
  corr = np.fft.irfft(fft_corr)  
  if n == None:
    m = np.shape(corr)[1]
    mid_point = int(m/2)
    second_half = corr[...,:mid_point]
    first_half  = corr[...,mid_point:]

    corr = np.concatenate((first_half , second_half), axis = -1)
    return corr
  else:
    m = int(n/2)
    return np.concatenate((corr[...,-m:] , corr[...,:m]), axis = -1)

def shiftData(y, shift, error=None, ext=3):
  """ Shifts data considering errors 
  """
  x = np.arange(len(y))

  weights = None
  if error is not None:
    weights = 1/error

  ip = interpolate.splrep(x, y, weights)
  interpolated = interpolate.splev(x - shift, ip, ext=ext)

  return interpolated

def fourierShift1D(y, shift, n=-1, fourier_domain=False):
  """ Shifts data quickly,
    Option to pass data already in fourier domain
  """
  if not fourier_domain:
    m = len(y)
    y, n = rfft(y,returnPadLen=True)

  fft_shift = ndi.fourier_shift(y, shift, n)

  if not fourier_domain:
    return np.fft.irfft(fft_shift)[:m]

  return fft_shift

def fourierShift2D(a, shifts, n=-1, fourier_domain=False):
  if not fourier_domain:
    m = np.shape(a)[1]
    a,n = rfft(a, returnPadLen=True)

  temp = []
  for i in range(len(a)):
    temp.append(fourierShift1D(a[i], shifts[i], n=n, fourier_domain=True))

  if not fourier_domain:
    return np.fft.irfft(temp)[:,:m]

  return np.array(temp)

def upSampleData(x, y, upSampleFactor = 10, error=None, ext=3):
  upSampX = np.linspace(x[0], x[-1], len(x)*upSampleFactor)

  weights = None
  if error is not None:
    weights = 1/error

  interpolation = interpolate.splrep(x, y, weights)
  upSampY = interpolate.splev(upSampX, interpolation, ext = ext)

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

def normalize(d, outRange=[0,1]):
  num = d-np.min(d)
  den = (np.max(d)-np.min(d))/(outRange[1]-outRange[0])
  return (num/den) +outRange[0]

def percStd(data):
    return (np.percentile(data,84) - np.percentile(data,16))/2
###

#-- Physics
def rv(times, t0=0, P=0, w_deg=0, e=0, Kp=0, v_sys=0,
        vectorizeFSolve = False, doBarycentricCorrect=False,
        ra=None, dec=None, raunits='hours', obsname=None,
        returnBCSeparately = False, chunk_size=100, **kwargs
):
  """
  Computes RV from given model, barycentric velocity

  :param t     : Times of Observations
  :param to    : Time of Periastron
  :param P     : Orbital Period
      # t, t0, P must be same units
  :param w_deg : Argument of periastron
      # degrees
  :param Kp     : Planets Orbital Velocity
  :param v_sys : Velocity of System
      # K, v_sys must be same unit
      # Output will be in this unit
  :return: radial velocity
  """
  w = np.deg2rad(w_deg)
  mean_anomaly = ((2*np.pi)/P * (times - t0)) % (2*np.pi)

  if not vectorizeFSolve:
    try:
      E = []
      for m in mean_anomaly:
        kepler_eqn = lambda E: E - e*np.sin(E) - m
        E.append(optimize.fsolve(kepler_eqn, m)[0])
      E = np.array(E)
    except:
      kepler_eqn = lambda E: E - e*np.sin(E) - mean_anomaly
      E = optimize.fsolve(kepler_eqn, mean_anomaly)
  else:
    kepler_eqn = lambda E: E - e*np.sin(E) - mean_anomaly
    E = optimize.fsolve(kepler_eqn, mean_anomaly)

  true_anomaly = np.arctan2(np.sqrt(1-e**2) * np.sin(E), np.cos(E)-e)

  velocity = Kp * (np.cos(true_anomaly+w) + e*np.cos(w)) + v_sys

  if doBarycentricCorrect:
    # Have to split into chunks b/c webserver can only process
    # so much
    bc = np.array([])
    n = len(times)
    all_i = np.arange(n)
    chunks = [all_i[i:i+chunk_size] for i in range(0,n,chunk_size)]
    for chunk in chunks:
      bc_chunk = barycorr.bvc(times[chunk], ra=ra[chunk],
                    dec=dec[chunk], obsname=obsname,
                    raunits=raunits)
      bc = np.concatenate((bc,bc_chunk))

    # Subtract BC to be in target frame
    bc = -1*bc 
    if returnBCSeparately:
      return velocity, bc
    velocity = velocity + bc

  return velocity

def doppler(wave,v, source=False):
  # v in m/s
  # source = False: wave is observed wavelengths
  # source = True: wave is source wavelengths
  beta = v/constants.c
  if source:
    xsq = (1+beta)/(1-beta)
  else:
    xsq = (1-beta)/(1+beta)
  return np.sqrt(xsq)*wave

def inverseDoppler(wave, wave_shift, source=False):
  waveCenter = np.median(wave)
  
  z = wave_shift/ (waveCenter - wave_shift)
  if source:
    z = wave_shift/ (waveCenter)
  A = (1+z)**2
  return (A-1)/(A+1) * constants.c
###