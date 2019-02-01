from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import constants, signal, stats, interpolate, optimize
from scipy import ndimage as ndi
import pickle
from functools import partial
import highResAnalysis as hr
import numpy as np
import os
import multiprocessing as mp

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

#TODO GeneralizePlanet
def collectRawIRTF(order, date, dataPaths):
  # Collect Raw Data
  wave  = []
  flux  = []
  error = []
  times = []

  ddir = dataPaths['dirPre']+date+'/'

  for fn in os.listdir(ddir):
    ext = fn.split('.')[1]
    if ext != 'fits':
      continue

    data = fits.getdata(ddir+fn)
    hdr = fits.getheader(ddir+fn)

    
    wave.append(data[order,0])
    flux.append(data[order,1])
    error.append(data[order,2])
    times.append(float(hdr['MJD'])+2400000.5)

  # Try to load Barycentric Correction, write it if not found:
  baryFile = ddir+'barycentricVelocity.pickle'
  try:
    with open(baryFile,'rb') as f:
      barycentricCorrection = pickle.load(f)
  except FileNotFoundError:
    # Calculate barycentric correction
    barycentricCorrection = hr.getBarycentricCorrection(times, 'upsilon andromedae', 'irtf', verbose=1)

    # Write it
    with open(baryFile,'wb') as f:
      pickle.dump(barycentricCorrection,f)

  # Remove Nans from data
  flux = np.array(flux)
  wave = np.array(wave)
  error = np.array(error)

  fnan = np.argwhere(np.isnan(flux[0]))
  try:
    fnan = fnan[0,0]

    flux = flux[:,:fnan]
    wave = wave[:,:fnan]
    error = error[:,:fnan]
  except:
    pass

  return flux, wave[0], error, np.array(times), barycentricCorrection

def getFlux(order,date, dataPaths):
  flux, wave, error, times, baryCor = collectRawIRTF(order, date, dataPaths)

  rv_params = {
    'times'      : times,
    'obsname'    : 'irtf',
    'planetName' : 'upsilon andromedae'
  }

  highSNR = hr.getHighestSNR(flux, error)
  ref = flux[highSNR]
  flux, error = hr.alignment(flux, ref, error=error)

  return flux, wave, error, times, baryCor, rv_params


# TODO: What about correcting error?
def correctBlazeCrude(flux):
  # If flux is 2d, correct each 1d segment individually
  if np.ndim(flux) == 2:
    corrected = np.array([correctBlazeCrude(f) for f in flux])
    return corrected

  elif np.ndim(flux) == 1:
    fourier = hr.rfft(flux)

    fourier[0:2] = 0
    corrected_flux = np.fft.irfft(fourier)
    corrected_flux = corrected_flux[:len(flux)]

    return corrected_flux
  else:
    raise ValueError('Dimensions of flux must be either 1 or 2')


def prepareIRTF(flux,
                # Masking Params:
                  use_time_mask = True, time_mask_cutoffs = [3,0],
                  use_wave_mask = False, wave_mask_window = 100,
                  wave_mask_cutoffs = [3,0],
                  mask_smoothing_factor = 10,
                # Normalizations:
                  correctBlaze = True,
                  normalize = True,
                #Sysrem Params:
                  sysremIterations = 0, error = None,
                  returnAllSysrem = False,
                #Variance Weighting Params:
                  doVarianceWeight = True,
                #Plotting Options:
                  plotTimeMask = False, plotWaveMask = False,
                  plotMask = False,
                  verbose=False, 
                  **kwargs
):

  superVerbose = verbose>1

  # Calculate Mask for flux (To be done before continuum subtracting)
  if use_time_mask or use_wave_mask:
    if verbose:
      print("Creating Mask")

    time_mask = np.ones(np.shape(flux)[1])
    wave_mask = np.ones(np.shape(flux)[1])

    if use_time_mask:
      time_mask = hr.getTimeMask(flux, *time_mask_cutoffs,
          smoothingFactor=0, showPlots = plotTimeMask)
    
    if use_wave_mask:
      wave_mask = hr.getWaveMask(flux, wave_mask_window, *wave_mask_cutoffs, smoothingFactor=0, showPlots = plotWaveMask)

    mask = hr.combineMasks(time_mask, wave_mask, 
        smoothingFactor=mask_smoothing_factor)

    #pbar.update()

    if plotMask:
      plt.figure()
      plt.title('Full Mask')
      plt.plot(hr.normalize(np.median(flux,0)))
      plt.plot(mask)
      plt.ylim(-0.2,1.2)
      plt.show()

  if correctBlaze:
    flux = correctBlazeCrude(flux)

  if normalize:
    flux = flux/np.mean(flux,1)[:,np.newaxis]

  # Apply mask to flux
  if use_time_mask or use_wave_mask:
    flux = hr.applyMask(flux, mask)

  if sysremIterations != 0:
    if verbose:
      print('Doing Sysrem')

    # Send pbar to sysrem
    flux = hr.sysrem(flux, error, sysremIterations, verbose=superVerbose,
      retAll = returnAllSysrem)

  if doVarianceWeight:
    if verbose:
      print('Variance Weighting Columns')
    flux = hr.varianceWeighting(flux)

  return flux

def runIRTF(order, date, dataPaths, templateName, kpRange, orb_params, 
            vsys_range=[-600000,600000],
            verbose=False,
            **kwargs
):
  superVerbose = verbose>1
  flux, wave, error, times, baryCor, rv_params = getFlux(order, date, dataPaths)

  templateData = hr.readFile(dataPaths['templateDir']+templateName)
  template = hr.interpolateTemplate(templateData, wave)

  data = prepareIRTF(flux,
                      error=error,
                      verbose=superVerbose,
                      **kwargs)

  sm, x, y = irtfGenerateSmudgePlot(data, wave, rv_params, template,
                               kpRange, orb_params,
                               vsys_range=vsys_range,
                               verbose=superVerbose,
                               barycentricCorrection=baryCor,
                               **kwargs)

  return sm, x, y 

def singleSys(sysIt, order=0, date='', dataPaths=None,
                     templateName=None, kpRange=None,
                     kpExtent = 4, vsysExtent = 3,
                     plotKp = 40, plotVsys=50,
                     orb_params=None, verbose=False,
                     saveDir = None, doPlot=True,
                     target_Kp=None, target_Vsys=None,
                     **kwargs
):
  smudges, vsys_axis, kp_axis = runIRTF(order, date, dataPaths, templateName,
                                  kpRange, orb_params, sysremIterations=sysIt,
                                  verbose=verbose,
                                  **kwargs)

  # smudges=smudges/np.std(smudges)
  if target_Kp is None:
    target_Kp=orb_params['Kp']
  if target_Vsys is None:
    target_Vsys=orb_params['v_sys']

  # Isolate region in small box around target values
  kpWindowMin = np.where(kp_axis <= target_Kp - kpExtent*1000)[0][-1]
  kpWindowMax = np.where(kp_axis >= target_Kp + kpExtent*1000)[0][0]

  vsys_window_min = np.where(vsys_axis <= target_Vsys - vsysExtent*1000)[0][-1]
  vsys_window_max = np.where(vsys_axis >= target_Vsys + vsysExtent*1000)[0][0]
  
  window = smudges[kpWindowMin:kpWindowMax+1, vsys_window_min:vsys_window_max+1]

  # Get value at target kp, vsys
  targetYCoord = np.argmin(np.abs(kp_axis - target_Kp))
  targetXCoord = np.argmin(np.abs(vsys_axis - target_Vsys))
  
  if doPlot:
    xMin = target_Vsys/1000 - plotVsys
    xMax = target_Vsys/1000 + plotVsys

    yMin = target_Kp/1000 - plotKp
    yMax = target_Kp/1000 + plotKp

    hr.plotSmudge(smudges,vsys_axis,kp_axis,xlim=[xMin,xMax],
              ylim=[yMin,yMax],
              target_Kp=target_Kp/1000,target_Vsys=target_Vsys/1000,
              title=date+', '+templateName.split('.')[0]+'\norder: '+str(order)+', sysit: '+str(sysIt),
              close=False, show=False,figsize=(6,5))

    vsys_window = vsys_axis[vsys_window_min:vsys_window_max+1]
    kp_window = kp_axis[kpWindowMin:kpWindowMax+1]

    box_top = kp_window[-1]/1000
    box_bot = kp_window[0]/1000
    box_left = vsys_window[0]/1000
    box_right = vsys_window[-1]/1000

    box_color='k'
    box_style='--'

    plt.plot( (box_left,box_left), (box_bot, box_top), c=box_color, linestyle=box_style)
    plt.plot( (box_right,box_right), (box_bot, box_top), c=box_color, linestyle=box_style)
    plt.plot( (box_left,box_right), (box_top, box_top), c=box_color, linestyle=box_style)
    plt.plot( (box_left,box_right), (box_bot, box_bot), c=box_color, linestyle=box_style)

    search_max_coords = np.unravel_index(np.argmax(window),np.shape(window))
    search_min_coords = np.unravel_index(np.argmin(window),np.shape(window))

    plt.scatter(vsys_window[search_max_coords[1]]/1000,
        kp_window[search_max_coords[0]]/1000,
        marker='^',s=40)

    plt.scatter(vsys_window[search_min_coords[1]]/1000,
        kp_window[search_min_coords[0]]/1000,
        marker='v',s=40)

    if saveDir is not None:
      datePath = saveDir+date+'/'
      if not os.path.isdir(datePath):
        os.mkdir(datePath)

      temPath = datePath + templateName.split('.')[0]+'/'
      if not os.path.isdir(temPath):
        os.mkdir(temPath)

      orderPath = temPath+'order_'+str(order)+'/'
      if not os.path.isdir(orderPath):
        os.mkdir(orderPath)

      saveName = orderPath+'sysIt_'+str(sysIt)+'.png'

      plt.savefig(saveName)

    plt.show()
    plt.close()

  searchMax = np.max(window)
  searchMin = np.min(window)
  targetVal = smudges[targetYCoord,targetXCoord]

  return sysIt, targetVal, searchMin, searchMax

def sysOpt(orders, maxIt, date, dataPaths,
           kpRange, orb_params, cores=1,
           templateName = None,
           saveDir=None, saveTrend=True,
           verbose=False, **kwargs
):
  orderRange = orders
  if verbose:
    orderRange = tqdm(orders)

  indiciesToPlot = []
  maxValsToPlot = []
  minValsToPlot = []
  targValsToPlot = []


  for i, order in enumerate(orderRange):
    minVals = []
    maxVals = []
    targVals = []
    indicies = []
    pool = mp.Pool(processes = cores)
    if verbose:
      pbar = tqdm(total = maxIt)
    seq = pool.imap_unordered(partial(singleSys,
                                      order=order,
                                      date=date,
                                      dataPaths=dataPaths,
                                      kpRange=kpRange,
                                      templateName=templateName,
                                      orb_params=orb_params,
                                      saveDir=saveDir,
                                      verbose=np.max((0,verbose-1)),
                                      **kwargs),
                              range(maxIt))

    for output in seq:
      if verbose:
        pbar.update()

      sysIt, targVal, minVal, maxVal = output
      minVals.append(minVal)
      maxVals.append(maxVal)
      targVals.append(targVal)
      indicies.append(sysIt)

    if verbose:
      pbar.close()

    minVals = [x for _,x in sorted(zip(indicies,minVals))]
    maxVals = [x for _,x in sorted(zip(indicies,maxVals))]
    targVals = [x for _,x in sorted(zip(indicies,targVals))]
    indicies = sorted(indicies)

    minValsToPlot.append(minVals)
    maxValsToPlot.append(maxVals)
    targValsToPlot.append(targVals)
    indiciesToPlot.append(indicies)
  
  # make master plot
  plt.figure()
  for i,order in enumerate(orders):
    colStr = 'C'+str(i)

    plt.plot(indiciesToPlot[i], targValsToPlot[i], lw=2, label='Order '+str(order),
            color=colStr)
    plt.fill_between(indiciesToPlot[i], maxValsToPlot[i], minValsToPlot[i],
           facecolor=colStr, alpha=0.5)

  plt.legend()
  plt.xlabel('Sysrem Iterations')
  plt.ylabel('SNR')
  plt.title(date+', '+templateName.split('.')[0]+'\nSysrem Optimizations')

  if saveDir is not None and saveTrend:
    datePath = saveDir+date+'/'
    if not os.path.isdir(datePath):
      os.mkdir(datePath)

    temPath = datePath + templateName.split('.')[0]+'/'
    if not os.path.isdir(temPath):
      os.mkdir(temPath)

    orderString = ''
    for order in orders:
      orderString+='-'+str(order)

    saveName = temPath+'sysremTrend'+orderString+'.png'

    plt.savefig(saveName)

  # Make individual plots
  for i,order in enumerate(orders):
    plt.figure()
    colStr = 'C'+str(i)
    plt.plot(indiciesToPlot[i], targValsToPlot[i], lw=2, label='Order '+str(order),
            color=colStr)
    plt.fill_between(indiciesToPlot[i], maxValsToPlot[i], minValsToPlot[i],
           facecolor=colStr, alpha=0.5)

    plt.legend()
    plt.xlabel('Sysrem Iterations')
    plt.ylabel('SNR')
    plt.title(date+', '+templateName.split('.')[0]+'order '+str(order)+'\nSysrem Optimizations')
    if saveDir is not None:
      datePath = saveDir+date+'/'
      temPath = datePath + templateName.split('.')[0]+'/'
      orderPath = temPath+'order_'+str(order)+'/'
      if not os.path.isdir(orderPath):
        os.mkdir(orderPath)

      saveName = orderPath+'sysremTrend.png'
      plt.savefig(saveName)


# Lightly Modified from High Res
def irtfInitializeSmudgePlot(data, wave, rv_params, template,
                        kpRange, orb_params,
                        vsys_range = None,
                        normalizeXCors = True,
                        xcorMode = 'same',
                        verbose = False,
                        barycentricCorrection=None
):
  if verbose:
    print('Generating Cross Correlations')
  # xcor Interps
  xcm = hr.generateXCorMatrix(data, wave, template, 
                          normalize=normalizeXCors, xcorMode=xcorMode,
                          verbose=(verbose>1))
  vsys = hr.getXcorVelocities(wave, xcorMode)
  times = rv_params['times']

  xcor_interps = []
  for xcor in xcm:
    xcor_interps.append(interpolate.splrep(vsys, xcor))

  # unit RVs
  orb_params = orb_params.copy()
  orb_params['v_sys'] = 0
  orb_params['Kp']    = 1
  unitRVs = hr.getRV(**rv_params, **orb_params)
  if barycentricCorrection is None:
    barycentricCorrection = hr.getBarycentricCorrection(**rv_params, verbose=verbose)

  # vsys limited
  if vsys_range != None:
    # Tile together entire range of planet RV's we consider
    allRVs = np.tile(unitRVs,(len(kpRange),1)) * kpRange[:,np.newaxis] 
    # Roll in barycentric correction
    allRVs = allRVs + barycentricCorrection

    # Find which columns will end up having a contribution to a systemic velocity in the given range
    min_vels = vsys + np.min((np.min(allRVs),0))
    max_vels = vsys + np.max((np.max(allRVs),0)) 
    goodCols = np.logical_and(max_vels >= vsys_range[0], min_vels  <= vsys_range[1])
    # print(min_vels,max_vels)
    vsys = vsys[goodCols]

  return xcor_interps, unitRVs, barycentricCorrection, vsys

def irtfGenerateSmudgePlot(data, wave, rv_params, template,
                        kpRange, orb_params,
                        vsys_range = None,
                        normalizeXCors = True,
                        xcorMode = 'same',
                        verbose = False,
                        retAxes = True,
                        ext = 3,
                        mode=1,
                        stdDivide = True,
                        pbar = None,
                        barycentricCorrection=None,
                        **kwargs
):
  if verbose:
    print('Initializing')

  # Initialize
  xcor_interps, unitRVs, barycentricCorrection, vsys = \
        irtfInitializeSmudgePlot(data, wave, rv_params, 
                             template, kpRange, orb_params,
                             vsys_range=vsys_range,
                             normalizeXCors=normalizeXCors,
                             xcorMode=xcorMode, verbose=verbose,
                             barycentricCorrection=barycentricCorrection)

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
    aligned_xcm = hr.alignXcorMatrix(xcor_interps, vsys, rvs, ext=ext)

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
    
    if pbar is not None:
      pbar.update()

  smudges = np.array(smudges)
  if stdDivide:
    smudges = smudges/np.apply_along_axis(hr.percStd,1,smudges)[:,np.newaxis]

  if vsys_range is not None:
    goodCols = np.logical_and(vsys>=vsys_range[0], vsys<=vsys_range[1])
    goodCols = ndi.maximum_filter(goodCols,3)
    smudges = smudges[:, goodCols]
    vsys = vsys[goodCols]

  if retAxes:
    return smudges, vsys, kpRange
  else:
    return smudges