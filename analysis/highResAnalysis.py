import numpy as np
import pickle, json
from scipy import ndimage, interpolate, optimize
from scipy import constants, signal, stats
from astropy.io import fits
import matplotlib.pyplot as plt

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
def collectData(order, data_dir, data_pre, data_pos,
                header_file, templateFile,
                discard_rows = [],
                discard_cols = [],
                doAlign = True,
                alignmentIterations = 3,
                padLen = 50, peak_half_width = 3,
                upSampleFactor = 1000,
                verbose = False, **kwargs
):
  """ #Performs Steps 0-2 all at once
  """
  #Load Raw Data
  dataFileName = data_dir+data_pre+str(order)+data_pos
  if verbose:
    print('Collecting Data')
  data = collectRawData(dataFileName)
  flux = data['fluxes']
  wave = data['waves']
  error = data['errors']
  del data

  headers = collectRawData(data_dir + header_file)
  times = np.array(headers['JD'])
  del headers

  # Discard Bad data
  for row in discard_rows:
    flux  = np.delete(flux, row, 0)
    error = np.delete(error, row, 0)
    times = np.delete(times, row)

  for col in discard_cols:
    flux  = np.delete(flux, col ,1)
    error = np.delete(error, col, 1)
    wave  = np.delete(wave, col)

  # Trim off ends
  flux, wave, error = trimData(flux, wave, error)

  if doAlign:
    if verbose:
      print('Aligning Data')
    highSNR = getHighestSNR(flux,error)
    ref = flux[highSNR]

    flux, error = alignment(flux, ref, iterations=alignmentIterations,
      error = error, padLen = padLen,
      peak_half_width = peak_half_width,
      upSampleFactor = upSampleFactor, verbose = verbose>1)
    
  template = getTemplate(templateFile, wave)

  return flux, error, wave, times, template

def prepareData(flux,
              # Fake Signal Params:
                wave = None, times = None, templateFile = None, 
                orb_params = None, fake_signal_strength = 0, 
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
                verbose=False, **kwargs
):
  superVerbose = verbose>1

  # Add fake signal to Flux
  if fake_signal_strength != 0:
    if verbose:
      print('Injecting Fake Data')

    flux = addTemplateToData(flux, wave, times,  orb_params,
            templateFile, fake_signal_strength, verbose=superVerbose)

  # Calculate Mask for flux (To be done before continuum subtracting)
  if use_time_mask or use_wave_mask:
    if verbose:
      print("Creating Mask")

    time_mask = np.ones(np.shape(flux)[1])
    wave_mask = np.ones(np.shape(flux)[1])

    if use_time_mask:
      time_mask = getTimeMask(flux, *time_mask_cutoffs,
          smoothingFactor=0)
    
    if use_wave_mask:
      wave_mask = getWaveMask(flux, wave_mask_window, *wave_mask_cutoffs, smoothingFactor=0)

    mask = combineMasks(time_mask, wave_mask, 
        smoothingFactor=mask_smoothing_factor)

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

def calcSysremIterations(order, data_dir, data_pre, data_pos,
                        header_file, templateFile, orb_params,
                        fake_signal_strengths=[1/1000],
                        maxIterations=10,
                        verbose=False, **kwargs
):
  """ Computes detection strength vs number of sysrem iterations
    Params: See collectData(), prepareData(), generateSmudgePlot()
    for additional optional parameters
  """

  # Collect Data
  flux, error, wave, times, template = collectData(order,
              data_dir, data_pre, data_pos, header_file,
              templateFile, verbose=verbose, **kwargs)

  # Injected KpValue
  kpRange = np.array([orb_params['Kp']])
  all_detection_strengths = []

  for signal_strength in fake_signal_strengths:
    if verbose:
      print('-----------------------------')
      print('Working on signal: '+str(signal_strength))
      print('-----------------------------')
    this_ds = []

    # Calculate maxIterations sysrem iterations
    sysremData = prepareData(flux,
                    wave=wave, times=times,
                    templateFile=templateFile, orb_params=orb_params,
                    fake_signal_strength= signal_strength,
                    sysremIterations=maxIterations, error=error,
                    returnAllSysrem=True, verbose=verbose, **kwargs)

    if verbose:
        print('Calculating Detection Strengths')
    # Calculate the detection strength for each iteration
    for residuals in sysremData:
      smudges, vsys_axis, kp_axis = generateSmudgePlot(residuals, wave,
                                      times, template, kpRange,
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

def pipeline(order, data_dir, data_pre, data_pos,
             header_file, templateFile, orb_params,
             kpRange, verbose=False, **kwargs
):
  """ Completely generates a smudge plot for the given order, data
    See collectData(), prepareData(), generateSmudgePlot() for 
    optional parameter inputs
  """

  # Collect Raw Data
  flux, error, wave, times, template = collectData(order,
              data_dir, data_pre, data_pos, header_file,
              templateFile, verbose=verbose, **kwargs)

  data = prepareData(flux, wave=wave, times=times,
          templateFile=templateFile, orb_params=orb_params,
          error=error, verbose=verbose,
          **kwargs)

  smudges, vsys_axis, kp_axis = generateSmudgePlot(data, wave, times,
                                  template, kpRange, orb_params,
                                  verbose=verbose, retAxes=True,
                                  **kwargs)

  if verbose:
    print('Done!')
  return smudges, vsys_axis, kp_axis 
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
  plt.pcolormesh(pltXs,pltYs,smudges)
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
      str(np.round(smudges[ptmax],2)) + true_val_str)

  cbar.set_label('Sigma')

  if xlim is not None:
    plt.xlim(*xlim)
  if ylim is not None:
    plt.ylim(*ylim)

  plt.show()
###

#-- Step 0: Raw Data 
def collectRawData(dataFile):
  with open(dataFile,'rb') as f:
    data = pickle.load(f)

  return data

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

###

#-- Step 1: Delete bad data
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
                smoothingFactor = 20
):
  weights  = np.apply_along_axis(snr, 0, flux)
  if np.any(weights < 0):
    print('Warning, some weights (SNRs) are less than zero, consider using non zero-mean values for generating mask')
    absoluteCutoff = -np.inf

  weightMean = np.mean(weights)
  weightStd  = np.std(weights)

  lowerMask = weights < weightMean - relativeCutoff*weightStd
  absoluteMask = weights < absoluteCutoff

  mask = 1-np.logical_or(lowerMask,absoluteMask)


  return ndimage.minimum_filter(mask, smoothingFactor)

def getWaveMask(flux, window_size=100, relativeCutoff = 3,
                  absoluteCutoff = 0, smoothingFactor = 20
):
  medSpec = np.median(flux,0)
  weights = ndimage.generic_filter(medSpec, snr, size=window_size)

  weightMean = np.mean(weights)
  weightStd  = np.std(weights)

  lowerMask = weights < weightMean - relativeCutoff*weightStd
  absoluteMask = weights < absoluteCutoff

  mask = 1-np.logical_or(lowerMask,absoluteMask)

  return ndimage.minimum_filter(mask, smoothingFactor)

def combineMasks(*masks, smoothingFactor=20):
  mask = np.prod(masks,0)
  return ndimage.minimum_filter(mask, smoothingFactor)

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

def initializeSmudgePlot(data, wave, times, template,
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

  xcor_interps = []
  for xcor in xcm:
    xcor_interps.append(interpolate.splrep(vsys, xcor))

  # unit RVs
  orb_params = orb_params.copy()
  orb_params['v_sys'] = 0
  orb_params['Kp']    = 1
  unitRVs = rv(times, **orb_params)

  # vsys limited
  if vsys_range != None:
    allRVs = np.tile(unitRVs,(len(kpRange),1)) * kpRange[:,np.newaxis]
    min_vels = vsys + np.min((np.min(allRVs),0))
    max_vels = vsys + np.max((np.max(allRVs),0))
    goodCols = np.logical_and(max_vels >= vsys_range[0], min_vels  <= vsys_range[1])
    vsys = vsys[goodCols]

  return xcor_interps, unitRVs, vsys

def generateSmudgePlot(data, wave, times, template,
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
  xcor_interps, unitRVs, vsys = initializeSmudgePlot(data, wave, times, template, kpRange, orb_params,
                                 vsys_range=vsys_range, normalizeXCors=normalizeXCors,
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
    rvs = kpRange[i] * unitRVs
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
###

#-- Template Functions
def addTemplateToData(flux, wave, times, orb_params, 
                      templateFile, templateStrength,
                      verbose = False
):
  template_interp = getTemplateInterpolation(templateFile)

  fake_signal = []
  seq = times
  if verbose:
    seq = tqdm(seq, desc='Generating Fake Data')
    
  for time in seq: 
    sourceWave = doppler(wave, rv(time, **orb_params))
    this_flux  = normalize(interpolate.splev(sourceWave, template_interp))
    fake_signal.append(this_flux)
  fake_signal = np.array(fake_signal) * np.median(flux,1)[:,np.newaxis]*templateStrength
  return flux + fake_signal

def getTemplateInterpolation(templateFile):
  extension = templateFile.split('.')[-1]
  if extension == 'fits':
    t_wave, t_flux = np.transpose(fits.getdata(templateFile))
  elif extension == 'pickle':
    try:
      with open(templateFile,'rb') as f:
        data = pickle.load(f)
    except UnicodeDecodeError:
      with open(templateFile,'rb') as f:
        data = pickle.load(f, encoding='latin1')

    t_wave = data['wavelengths']
    t_flux = data['flux']
  else:
    raise ValueError('TemplateFile must be either .fits or .pickle file')

  template_interp = interpolate.splrep(t_wave/10000, t_flux)
  return template_interp

def getTemplate(templateFile, wave):
  template_interp = getTemplateInterpolation(templateFile)
  template = interpolate.splev(wave, template_interp)
  return template
###

'''
   Generic Functions
'''
#-- Math
def snr(data):
  return np.mean(data)/np.std(data)

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

  fft_shift = ndimage.fourier_shift(y, shift, n)

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
def rv(t, t0, P, w, e, Kp, v_sys, vectorizeFSolve = False):
  # t     : Times of Observations
  # to    : Time of Periastron
  # P     : Orbital Period
      # t, t0, P must be same units
  # w     : Argument of periastron
      # Radians
  # Kp     : Planets Orbital Velocity
  # v_sys : Velocity of System
      # K, v_sys must be same unit
      # Output will be in this unit
  # NEEDS BARYCENTRIC VELOCITY

  mean_anomaly = ((2*np.pi)/P * (t - t0)) % (2*np.pi)

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