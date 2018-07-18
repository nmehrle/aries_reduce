import numpy as np
import pickle
from scipy import ndimage, interpolate, optimize, constants, signal, stats
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


'''
   Composite Functions
'''
# Performs Steps 0-2 all at once
def collectData(order, data_dir, data_pre, data_pos,
                header_file, templateFile,
                discard_rows = [],
                discard_cols = [],
                doAlign = True,
                verbose = False
):
  ''' 
    discard_data: rows to discard 
  '''

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
    flux, error = alignData(flux, error, wave, verbose = verbose)

  template = getTemplate(templateFile, wave)

  return flux, error, wave, times, template

def computeSysremIterations(order, orb_params, 
                    data_dir, data_pre, data_pos,
                    header_file, templateFile,
                    signal_strength = 1/1000, 
                    max_sysrem_its = 20,
                    continuum_order = 4,
                    mask_sigma = 2.5, mask_smooth = 40,
                    smudge_mode = 1,
                    vsys_range = None, kpRange = None, 
                    verbose = False
):
  # Collect Data
  print('hi')

def pipeline(order, orb_params, 
             data_dir, data_pre, data_pos, 
             header_file,
             templateFile, 
             signal_strength=1/1000,
             sysrem_its = 3, mask_sigma = 2,
             mask_smoothing_factor = 25,
             continuum_order=4, smudge_mode = 1,
             vsys_range = [-100*1000,100*1000],
             kpRange = np.arange(20,90,1)*1000,
             stdDivide = True,
             verbose=False
):
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

  # Discard Bad Data
  # Last Frame looks bad
  flux = flux[:-1,:]
  error = error[:-1,:]
  times = times[:-1]

  # Trim off ends
  flux, wave, error = trimData(flux, wave, error)

  flux, error = alignData(flux, error, wave, verbose = verbose)

  template = getTemplate(templateFile, wave)

  fake_signal = addTemplateToData(flux, wave, times,
                                orb_params, templateFile,
                                signal_strength, 
                                verbose= verbose)

  cs = continuumSubtract(fake_signal, continuum_order, verbose=True)

  mask = getTimeMask(cs, mask_sigma, smoothingFactor=mask_smoothing_factor)
  masked = applyMask(fake_signal, mask)

  sd = sysrem(masked, error, sysrem_its, verbose=verbose, retAll=False) 

  data = varianceWeighting(sd)

  std_args = [data, wave, times, template, kpRange, orb_params]
  std_kws  = {'normalizeXCors' : True,
              'verbose': verbose,
              'vsys_range': vsys_range,
              'retAxes':True,
              'mode': smudge_mode,
              'ext': 1,
              'stdDivide': stdDivide
             }
  sm,rx,ry = generateSmudgePlot(*std_args, **std_kws)
  return sm, rx, ry

# Misc
# ######################################################
def plotOrder(flux, wave, order=None, orderLabels=None):
  xlocs = [0, int(np.shape(flux)[1]/2), np.shape(flux)[1]-1]
  xlabs = wave[xlocs]
  xlabs = ['%.3f' % (np.round(x,3)) for x in xlabs]

  plt.figure()
  plt.imshow(flux,aspect='auto')
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

# Step 0: 
# Raw Data 
# ######################################################
def collectRawData(dataFile):
  with open(dataFile,'rb') as f:
    data = pickle.load(f)

  return data

# ######################################################

# Step 1: 
# Delete bad data
# ######################################################
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

# ######################################################

# Step 2:
# Align Data
# ######################################################
def getHighestSNR(flux, error):
  snrs = np.median(flux/error,1)
  return np.argmax(snrs)

# Gives alignment fixes from crosscorrelations with highSNR spectrum
# Shift value is center(highSNR) - center(this_spec)
#  I.e. negative shift value indicates this spectra is moved right, needs to be moved left
def findPixelShifts(flux, error,peak_half_width = 3, 
                      upSampleFactor = 100,
                      verbose = False,
                      xcorMode = 'same',
):
  highSNR = getHighestSNR(flux, error)

  ref_spec = flux[highSNR] - np.mean(flux[highSNR])
  auto_cor = signal.correlate(ref_spec, ref_spec, xcorMode)
  zero_point = np.argmax(auto_cor)

  centers = []

  seq = range(len(flux))
  if verbose:
    seq = tqdm(seq, desc="Finding Shifts")

  for i in seq:
    spec = flux[i]
    xcor = signal.correlate(spec-np.mean(spec), ref_spec, xcorMode)
    mid_point = np.argmax(xcor)

    #upsample the Cross Correlation Peak
    xcor_lb = mid_point - peak_half_width
    xcor_rb = mid_point + peak_half_width + 1

    peak_x = range(xcor_lb,xcor_rb)
    peak   = xcor[xcor_lb:xcor_rb]

    upSamp, upSampPeak = upSampleData(peak_x, peak, upSampleFactor=upSampleFactor)

    center = upSamp[np.argmax(upSampPeak)]

    centers.append(center)

  return  zero_point - np.array(centers)

def applyShifts(flux, error, shifts,
                verbose = False
):
  seq = range(len(flux))
  if verbose:
    seq = tqdm(seq, desc='Aligning Spectra')

  aligned_flux  = np.zeros(np.shape(flux))
  aligned_error = np.zeros(np.shape(error))


  for i in seq:
    aligned_flux[i] = fourierShiftData(flux[i],shifts[i])
    aligned_error[i] = fourierShiftData(error[i],shifts[i])
  return aligned_flux, aligned_error

def fourierShifts(flux,error,shifts, verbose=False):
  seq = range(len(flux))
  if verbose:
    seq = tqdm(seq, desc='Aligning Spectra')

  aligned_flux  = np.zeros(np.shape(flux))
  aligned_error = np.zeros(np.shape(error))
  
  for i in seq:
    aligned_flux[i]  = fs(flux[i],shifts[i])
    aligned_error[i]  = fs(error[i],shifts[i])

  return aligned_flux, aligned_error

def alignData(flux, error,
                interpolation_half_width = 2, 
                peak_half_width = 1.2, 
                upSampleFactor = 2000,
                verbose = False,
                xcorMode = 'same'
):
  shifts = findPixelShifts(flux, error,
            interpolation_half_width=interpolation_half_width,
            peak_half_width=peak_half_width,
            upSampleFactor=upSampleFactor,
            verbose=verbose,
            xcorMode=xcorMode)

  aligned_flux, aligned_error = applyShifts(flux,error,shifts,verbose)
  return aligned_flux, aligned_error

# ######################################################

# Step 3: 
# Remove Coherent Structure
# ######################################################
def continuumSubtract(data, order, verbose=False):
  single_spec = (np.ndim(data) == 1)

  result = []
  x = np.arange(np.shape(data)[-1])

  seq = data
  if single_spec:
    seq = [data]
  if verbose:
    seq = tqdm(seq, desc='Subtracting Continuums')

  for spec in seq:
    spec_polyfit = np.polyfit(x, spec, order)
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
  return np.nan_to_num(data/np.var(data,0))


def getTimeMask(flux, cutoff = 3, mode=1, smoothingFactor = 20):
  # Modes:
  # 1: zeroMean, std
  # 2: zeroMean, 1/std
  # 3: std
  # 4: 1/std
  # 5: mean/std
  if mode == 1:
    zeroMean = flux-np.mean(flux,0)
    weights  = np.std(zeroMean,0)
  elif mode == 2:
    zeroMean = flux-np.mean(flux,0)
    weights  = 1/np.std(zeroMean,0)
  elif mode == 3:
    weights  = np.std(flux,0)
  elif mode == 4:
    weights  = 1/np.std(flux,0)
  elif mode == 5:
    weights  = np.mean(flux,0)/np.std(flux,0)

  weightMean = np.mean(weights)
  weightStd  = np.std(weights)

  upperMask = weights > weightMean + cutoff*weightStd
  lowerMask = weights < weightMean - cutoff*weightStd

  mask = 1-np.logical_or(upperMask,lowerMask)
  return ndimage.minimum_filter(mask, smoothingFactor)


def applyMask(data, mask):
  new_sum = np.sum(data) - np.sum(data * (1-mask))
  new_len = np.sum(mask)*len(data)

  meanSub = data - new_sum/new_len
  return meanSub*mask

def snr(data):
  return np.mean(data)/np.std(data)

def getWaveMask(flux, window_size = 100, cutoff = 3):
  medSpec = np.median(flux,0)
  snrs = ndimage.generic_filter(medSpec, snr, size=window_size)
  return snrs

# Step 4:
# Compare To Template
# ######################################################
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
  # xcor Interps
  xcm = generateXCorMatrix(data, wave, template, normalize=normalizeXCors,
                            xcorMode=xcorMode, verbose=verbose)
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
                        retAxes = False,
                        ext = 3,
                        mode=1,
                        stdDivide = True
):
  if verbose:
    print('Initializing')

  # Initialize
  xcor_interps, unitRVs, vsys = initializeSmudgePlot(data, wave, times, template, kpRange, orb_params,
                                 vsys_range=vsys_range, normalizeXCors=normalizeXCors,
                                 xcorMode=xcorMode, verbose=verbose)


  #Setting up verbose iterator
  seq = range(len(kpRange))
  if verbose:
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

'''
   Testing Functions
'''
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
'''
   Generic Functions
'''
# Math
# ######################################################
def upSampleData(x, y, error=None, upSampleFactor = 10, ext=3):
  upSampX = np.linspace(x[0], x[-1], len(x)*upSampleFactor)

  weights = None
  if error is not None:
    weights = 1/error

  interpolation = interpolate.splrep(x, y, weights)
  upSampY = interpolate.splev(upSampX, interpolation, ext = ext)

  return upSampX, upSampY

def shiftData(y, shift, error=None, ext=3):
  x = np.arange(len(y))

  weights = None
  if error is not None:
    weights = 1/error

  ip = interpolate.splrep(x, y, weights)
  interpolated = interpolate.splev(x - shift, ip, ext=ext)

  return interpolated

def fourierShiftData(y, shift):
  fft_shift = ndimage.fourier_shift(np.fft.rfft(y), shift)
  return np.fft.irfft(fft_shift)

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
# ######################################################

# Physics
# ######################################################
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


