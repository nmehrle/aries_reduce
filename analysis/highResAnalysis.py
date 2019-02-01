import numpy as np
import pickle, json
import multiprocessing as mp
from functools import partial
from scipy import ndimage as ndi
from scipy import constants, signal, stats, interpolate, optimize
from astropy.io import fits
import matplotlib.pyplot as plt
import os, sys

from barycorrpy.utils import get_stellar_data
from astropy.time import Time
import barycorrpy

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
 

#TODO
# all in m/s instead of km/s randomly
# vsysRange -> Common X axes

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
# def collectData(date, order, dataFileParams,
# TODO: Collect barycentric data here instead of many times
def collectOrder(dataPaths,
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
                  obsname = None, planetName = '',
                  verbose = False,  **kwargs
):
  """ Collects, trims and aligns data for 1 order from 1 date
      
      USE:
        flux, error, wave, template, rv_params = collectOrder(dataPaths, **analysis_kws)

        analysis_kws set from setOrder

      VIEW OPTIONS:
        plotCuts - generate plots of where data is trimmed
  """
  #Load Raw Data
  if verbose:
    print('Collecting Data')
  data, headers, templateData = collectRawOrder(**dataPaths)

  #pbar.update()
  
  flux = data['fluxes']
  wave = data['waves']
  error = data['errors']
  del data

  times = np.array(headers['JD'])
  del headers

  if verbose:
    print('Trimming Data')

  applyRowCuts  = [times]
  applyColCuts  = [wave]
  applyBothCuts = [error]

  flux, applyRowCuts, applyColCuts, applyBothCuts = \
      applyDataCuts(flux, rowCuts = discard_rows, colCuts = discard_cols,
        doColEdgeFind = doAutoTrimCols, applyRowCuts=applyRowCuts,
        applyColCuts=applyColCuts, applyBothCuts=applyBothCuts,
        neighborhood_size=trim_neighborhood_size, showPlots=plotCuts,
        figTitle = 'Date: '+dataPaths['date']+', Order: '+str(dataPaths['order']))

  times = applyRowCuts[0]
  wave  = applyColCuts[0]
  error = applyBothCuts[0]

  #pbar.update()

  rv_params = {
    'times'      : times,
    'obsname'    : obsname,
    'planetName' : planetName
  }  

  if doAlign:
    if verbose:
      print('Aligning Data')
    highSNR = getHighestSNR(flux,error)
    ref = flux[highSNR]

    # Send pbar to alignment
    flux, error = alignment(flux, ref, iterations=alignmentIterations,
      error = error, padLen = padLen,
      peak_half_width = peak_half_width,
      upSampleFactor = upSampleFactor, verbose = verbose>1)
    
  template = interpolateTemplate(templateData, wave)

  if verbose:
    print('Done')
  return flux, error, wave, template, rv_params


# TODO Normalize ERROR as well as flux?
def prepareData(flux,
              #Normalization Params:
                normalization = 'divide_row',
                continuum_order = 0,
              #Masking Params:
                use_time_mask = True, time_mask_cutoffs = [3,0],
                use_wave_mask = False, wave_mask_window = 100,
                wave_mask_cutoffs = [3,0],
                mask_smoothing_factor = 10,
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
  """
    Removes Coherent Structure from data
    OPTIONS:
      Continuum Subtract: use continuum_order>0 to subtract

      MASK: set use_time_mask or use_wave_mask = True

      SYSREM: set sysremIterations > 0
        Requires param error to be set

      Variance Weight: set doVarianceWeight = True

      PLOTTING:
        plotTimeMask
        plotWaveMask
        plotMask

    USE:
      data = prepareData(flux, error=error, **analysis_kws)
      analsys_kws set from setOrder()

  """
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

    #pbar.update()

    if plotMask:
      plt.figure()
      plt.title('Full Mask')
      plt.plot(normalize(np.median(flux,0)))
      plt.plot(mask)
      plt.ylim(-0.2,1.2)
      plt.show()

  # Normalize Flux
  if normalization is not None:
    # This is wrong
    if normalization == 'subtract_col':
      flux = flux-np.mean(flux,1)[:,np.newaxis]
    elif normalization == 'subtract_row':
      flux = flux-np.mean(flux,0)
    elif normalization == 'subtract_all':
      flux = flux-np.mean(flux,0)
      flux = flux-np.mean(flux,1)[:,np.newaxis]
    elif normalization == 'divide_col':
      flux = flux / np.mean(flux,1)[:,np.newaxis]
    elif normalization == 'divide_row':
      flux = flux / np.mean(flux,0)
    elif normalization == 'divide_all':
      flux = flux / (np.mean(flux,0) * np.mean(flux,1)[:,np.newaxis])
      # flux = flux / np.mean(flux,1)[:,np.newaxis]
    elif normalization == 'continuum':
      flux = continuumSubtract(flux, continuum_order, verbose=superVerbose)  
    else:
      raise(KeyError('Normalization Keyword '+normalization+' invalid. Valid KWs are a combination of "subtract, divide" and "row, col, all" e.g. "subtract_row". Or "continuum", with a valid Continuum Order'))
    #pbar.update()

  # Apply mask to flux
  if use_time_mask or use_wave_mask:
    flux = applyMask(flux, mask)
    # flux = flux*mask

    # pbar.update()

  if sysremIterations != 0:
    if verbose:
      print('Doing Sysrem')

    # Send pbar to sysrem
    flux = sysrem(flux, error, sysremIterations, verbose=superVerbose,
      retAll = returnAllSysrem)

  if doVarianceWeight:
    if verbose:
      print('Variance Weighting Columns')
    flux = varianceWeighting(flux)

  return flux

def calcSysremIterations(planet, date, order, dataPaths,
                        target_Kp, target_Vsys,
                        fake_signal_strengths=[0],
                        maxIterations=10, templateName=None,
                        kpExtent = 2, vsysExtent = 4,
                        plotKpExtent=40, plotVsysExtent=50,
                        saveDir=None, saveSuffix='',
                        verbose=False, **kwargs
):
  """ Computes detection strength vs number of sysrem iterations
    Params: See collectData(), prepareData(), generateSmudgePlot()
    for additional optional parameters
  """

  # Collect Data
  dataPaths, orb_params, analysis_kws =\
          setObs(planet, date, order, dataPaths, templateName=templateName)
  analysis_kws.update(kwargs)

  flux, error, wave, template, rv_params =\
        collectOrder(dataPaths, **analysis_kws)

  # data = prepareData(flux, error=error,
  #           **analysis_kws)

  # Injected KpValue
  searchKpRange = np.arange(-kpExtent, kpExtent+1)*1000 + target_Kp
  if saveDir is None:
    kpRange = searchKpRange
  else:
    kpRange = np.arange(-plotKpExtent, plotKpExtent+1)*1000 + target_Kp    

  if len(kpRange) == 0:
    kpRange = np.array([target_Kp])
  all_detection_strengths = []

  if 'commonXAxes' in kwargs.keys():
    commonXAxes = kwargs['commonXAxes']
    vsys_range = [commonXAxes[0], commonXAxes[-1]]
    kwargs['vsys_range'] = vsys_range

  for signal_strength in fake_signal_strengths:
    if verbose:
      print('-----------------------------')
      print('Working on signal: '+str(signal_strength))
      print('-----------------------------')
    this_ds = []

    if signal_strength == 0:
      fake_signal = flux
    else:
      fake_signal = injectFakeSignal(flux, wave, rv_params, orb_params,
                    target_Kp, target_Vsys, signal_strength, 
                    dataPaths=dataPaths, verbose=(verbose>1)*2, doAlert=False)

    # Calculate maxIterations sysrem iterations
    sysremData = prepareData(fake_signal,
                    sysremIterations=maxIterations, error=error,
                    returnAllSysrem=True, verbose=(verbose>1)*2, **kwargs)

    # Calculate the detection strength for each iteration
    if verbose>1:
      numKps = len(kpRange)*len(sysremData)
      innerPbar = tqdm(total=numKps, desc='Aligning Xcors')
    else:
      innerPbar=None

    for i, residuals in enumerate(sysremData):
      smudges, vsys_axis, kp_axis = generateSmudgePlot(residuals, wave,
                                      rv_params, template, kpRange,
                                      orb_params, retAxes=True,
                                      verbose=(verbose>2)*2,
                                      pbar=innerPbar,
                                      **kwargs)
      
      vsys_window_min = np.where(vsys_axis <= target_Vsys - vsysExtent*1000)[0][-1]
      vsys_window_max = np.where(vsys_axis >= target_Vsys + vsysExtent*1000)[0][0]

      vsys_window = vsys_axis[vsys_window_min:vsys_window_max+1]
      kpWindow = np.where([ (kp in searchKpRange)  for kp in kpRange])[0]
      window = smudges[kpWindow,vsys_window_min:vsys_window_max+1]

      if saveDir is not None:
        plot_vsys_window_min = np.where(vsys_axis <= target_Vsys - plotVsysExtent*1000)[0][-1]
        plot_vsys_window_max = np.where(vsys_axis >= target_Vsys + plotVsysExtent*1000)[0][0]

        plot_vsysWindow = vsys_axis[plot_vsys_window_min:plot_vsys_window_max+1]
        plotWindow = smudges[:,plot_vsys_window_min:plot_vsys_window_max+1]


        datePath = saveDir+date
        orderPath = datePath+'/order_'+str(order)
        fullPath  = orderPath+'/'+saveSuffix 
        if not os.path.isdir(datePath):
          os.mkdir(datePath)
        if not os.path.isdir(orderPath):
          os.mkdir(orderPath)
        if not saveSuffix == '':
          if not os.path.isdir(fullPath):
            os.mkdir(fullPath)

        plotSmudge(plotWindow, plot_vsysWindow, kpRange,
          xlim=None, title=getTitleStr(planet,date,order)+' :: '+str(i),
          target_Kp=target_Kp/1000, target_Vsys=target_Vsys/1000,
          close=False, show=False)

        # Draw Box around small Search Region
        box_top = searchKpRange[-1]/1000
        box_bot = searchKpRange[0]/1000
        box_left = vsys_window[0]/1000
        box_right = vsys_window[-1]/1000

        box_color='k'

        plt.plot( (box_left,box_left), (box_bot, box_top), c=box_color)
        plt.plot( (box_right,box_right), (box_bot, box_top), c=box_color)
        plt.plot( (box_left,box_right), (box_top, box_top), c=box_color)
        plt.plot( (box_left,box_right), (box_bot, box_bot), c=box_color)

        #mark maximum in search region
        search_max = np.unravel_index(np.argmax(window),np.shape(window))
        plt.scatter(vsys_window[search_max[1]]/1000,
            kpRange[kpWindow[search_max[0]]]/1000)

        plt.savefig(fullPath+'/'+str(i).zfill(2)+'.png')
      
      this_ds.append(np.max(window))

    if verbose>1:
      innerPbar.close()
    all_detection_strengths.append(np.array(this_ds))

  if verbose:
    print('Done!')
  return np.array(all_detection_strengths)

def processSysrem(i, planet, dates, orders, dataPaths,
                  target_Kp, target_Vsys, fake_signal_strength,
                  maxIterations=10, templateName=None,
                  kpExtent = 2, vsysExtent = 4,
                  plotKpExtent=40, plotVsysExtent=50,
                  saveDir=None, saveSuffix='',
                  verbose=False, kwargs=None
):
  n = len(orders)
  date  = dates[int(i/n)]
  order = orders[i%n]

  return calcSysremIterations(planet, date, order, dataPaths,
              target_Kp, target_Vsys,
              fake_signal_strengths = [fake_signal_strength],
              maxIterations = maxIterations, templateName=templateName,
              kpExtent = kpExtent, vsysExtent = vsysExtent,
              plotKpExtent=plotKpExtent, plotVsysExtent=plotVsysExtent,
              saveDir=saveDir, saveSuffix=saveSuffix,
              verbose = verbose, **kwargs)

def optimizeSysrem(planet, orders, dataPaths, target_Kp, target_Vsys,
                   fake_signal_strength, maxIterations = 10,
                   kpExtent = 5, vsysExtent = 8, 
                   dates=None,
                   plotKpExtent = 40, plotVsysExtent=50,
                   saveDir=None, saveSuffix='',
                   cores = 1,
                   verbose = False, write=False, 
                   templateName=None,
                   **kwargs
):
  if type(orders) == int:
    orders = [orders]
  
  #TODO include option for negative
  all_dates = getDates(planet,dataPaths)
  if dates is None:
    dates = all_dates
  elif type(dates) is str:
    dates = [dates]

  n_orders = len(orders) * len(dates)

  detection_strengths = []

  if verbose:
    pbar = tqdm(total=n_orders,desc='Optimizing Sysrem')

  pool = mp.Pool(processes = cores)
  seq = enumerate(pool.map(
                    partial(processSysrem,
                      planet=planet,
                      dates=dates,
                      orders=orders,
                      dataPaths=dataPaths,
                      target_Kp=target_Kp,
                      target_Vsys=target_Vsys,
                      fake_signal_strength=fake_signal_strength,
                      maxIterations=maxIterations,
                      kpExtent=kpExtent,
                      vsysExtent=vsysExtent,
                      plotKpExtent=plotKpExtent,
                      plotVsysExtent=plotVsysExtent,
                      saveDir=saveDir,
                      saveSuffix=saveSuffix,
                      templateName=templateName,
                      verbose=max(0,verbose-1),
                      kwargs=kwargs),
                    range(n_orders)))

  trends = []
  for i, opti in seq:
    # CalcSysremIterations returns list by strength, take 0th
    detection_strengths.append(np.argmax(opti[0]))
    trends.append(opti[0])

    if verbose:
      pbar.update()
  if verbose:
    pbar.close()

  if saveDir is not None:
    if not os.path.isdir(saveDir):
      os.mkdir(saveDir)

    for d in range(len(dates)):
      plt.figure()

      for o in range(len(orders)):
        k = d*len(orders) + o

        plt.plot(trends[k], label='order: '+str(o))

      plt.legend()
      if templateName is None:
        plt.title('SysremIterations: '+planet+', '+str(dates[d]))
        plt.savefig(saveDir+planet+'_'+str(dates[d])+'_'+saveSuffix+'sysIts.png')
      else:
        plt.title('SysremIterations: '+planet+', '+str(dates[d])+', template: '+templateName)
        plt.savefig(saveDir+planet+'_'+str(dates[d])+'_'+templateName+'_'+saveSuffix+'sysIts.png')

  if write:
    writeSysrem(dataPaths['planetData'], planet, dates, orders, detection_strengths, templateName)

  return detection_strengths

def analyzeOrder(planet = None, date = None, order = None,
                  orb_params=None, analysis_kws=None,
                  dataPaths=None, kpRange=None, verbose=False,
                  templateName=None,
                  **kwargs
):
  '''
    Generates Smudge for order.

    Inputs:
      kpRange and 

      Either A) planet, date, order, dataPaths
      or     B) dataPaths, orb_params, analysis_kws
  '''
  
  # If planet is specified, assume input option A
  if planet is not None:
    dataPaths, orb_params, analysis_kws =\
          setObs(planet, date, order, dataPaths, templateName=templateName)

  kwargs['verbose'] = verbose
  analysis_kws.update(kwargs)

  flux, error, wave, template, rv_params =\
        collectOrder(dataPaths, **analysis_kws)

  if 'injectSignal' in kwargs and kwargs['injectSignal'] == True:
    fake_signal = injectFakeSignal(flux, wave, rv_params, orb_params,
                    kwargs['fake_Kp'], kwargs['fake_Vsys'], kwargs['signal_strength'], 
                    dataPaths=dataPaths, verbose=verbose)

    flux = fake_signal

  data = prepareData(flux, error=error,
            **analysis_kws)

  smudge, vsys_axis, kp_axis =\
       generateSmudgePlot(data, wave, rv_params, template,
                          kpRange, orb_params, **analysis_kws)

  return smudge, vsys_axis, kp_axis

# Used for multiprocessign in analyzeData
def processOrder(i, orders, date_kws, order_kws,
          dataPaths, verbose, orb_params, 
          kpRange, templateName, kwargs
):
  analysis_kws, dataPaths = setOrder(orders[i], date_kws, order_kws, dataPaths,templateName=templateName)
  superVerbose = np.max((0, verbose-1))
  smudge, vsys_axis, kp_axis =\
          analyzeOrder(dataPaths = dataPaths, orb_params = orb_params,
            analysis_kws=analysis_kws, kpRange=kpRange,
            verbose=superVerbose, **kwargs)

  return smudge, vsys_axis, kp_axis

def analyzeDate(planet = None, date = None, orders = None,
                 orb_params=None, dates=None,
                 dataPaths=None, kpRange=None, verbose=False,
                 commonXAxes=None,
                 normalizeCombined=True, cores=1,
                 templateName=None,
                 **kwargs
):
  '''
    Generates Smudge plot for date, combining the orders given

    Inputs:
      kpRange and

      Either A) planet, date, orders, dataPaths
      or     B) dataPaths, orb_params, dates, date, orders
  '''
  if planet is not None:
    orb_params, dates, planetName, dataPaths = setPlanet(planet, dataPaths,templateName=templateName)
  date_kws, order_kws, dataPaths = setDate(date, dates, dataPaths)

  kwargs['planetName'] = planetName

  # Cannot normalize each smudge plot, only normalize finished product
  # Set up to use vsys_axis

  if commonXAxes is not None:
    vsys_range = [commonXAxes[0], commonXAxes[-1]]
    kwargs['vsys_range'] = vsys_range

  kwargs['stdDivide'] = False

  smudges = []
  x_axes = []
  y_axes = []

  # Setup Multiprocessing
  pool = mp.Pool(processes = cores)
  seq = enumerate(pool.imap_unordered(partial(processOrder,
                                              orders=orders,
                                              date_kws=date_kws,
                                              order_kws=order_kws,
                                              dataPaths=dataPaths,
                                              verbose=verbose,
                                              orb_params=orb_params,
                                              kpRange=kpRange,
                                              templateName=templateName,
                                              kwargs=kwargs),
                                      range(len(orders))))
  if verbose:
    pbar = tqdm(total=len(orders),desc='Orders')

  for i, order_output in seq:
    smudge, vsys_axis, kp_axis = order_output 
    smudges.append(smudge)
    x_axes.append(vsys_axis)
    y_axes.append(kp_axis)

    if verbose:
      pbar.update()
  if verbose:
    pbar.close()

  # Generate Common Vsys Axis if none specified
  if commonXAxes is None:
    # Set vsys_range so that we don't extrapolate
    minVal = np.max([np.min(x_axis) for x_axis in x_axes])
    maxVal = np.min([np.max(x_axis) for x_axis in x_axes])

    spacing = np.max([getSpacing(x_axis) for x_axis in x_axes])
    commonXAxes = np.arange(minVal, maxVal, spacing)
  
  combined = combineSmudges(smudges, x_axes, commonXAxes, 
                  normalize=normalizeCombined)

  return combined, commonXAxes, kpRange

def processDate(i, dates, orders, dataPaths, verbose, orb_params, 
          kpRange, templateName, kwargs
):
  n = len(orders)
  date  = list(dates.keys())[int(i/n)]
  j = i%n
  date_kws, order_kws, dataPaths = setDate(date, dates, dataPaths)

  return processOrder(j, orders, date_kws, order_kws, dataPaths,
                      verbose, orb_params, kpRange, templateName, kwargs)

def analyzePlanet(planet, orders, dataPaths, kpRange, 
                  cores=1, dates=None, commonXAxes=None, normalizeCombined=True,
                  templateName=None,
                  verbose=False, **kwargs
):
  orb_params, all_dates, planetName, dataPaths = setPlanet(planet, dataPaths,templateName=templateName)

  if dates is None:
    dates = all_dates
  elif type(dates) is str:
    dates = {dates: all_dates[dates]}
  elif type(dates) is list:
    temp = {}
    for date in dates:
      temp[date] = all_dates[date]
    dates = temp

  n_orders = len(orders) * len(dates)
  kwargs['stdDivide'] = False

  if commonXAxes is not None:
    vsys_range = [commonXAxes[0], commonXAxes[-1]]
    kwargs['vsys_range'] = vsys_range

  smudges = []
  x_axes = []
  y_axes = []

  # Setup Multiprocessing
  pool = mp.Pool(processes = cores)
  seq = enumerate(pool.imap_unordered(partial(processDate,
                                              dates=dates,
                                              orders=orders,
                                              dataPaths=dataPaths,
                                              verbose=verbose,
                                              orb_params=orb_params,
                                              kpRange=kpRange,
                                              templateName=templateName,
                                              kwargs=kwargs),
                                      range(n_orders)))
  if verbose:
    pbar = tqdm(total=n_orders,desc='Analyzing')

  for i, order_output in seq:
    smudge, vsys_axis, kp_axis = order_output 
    smudges.append(smudge)
    x_axes.append(vsys_axis)
    y_axes.append(kp_axis)

    if verbose:
      pbar.update()
  if verbose:
    pbar.close()
  # Generate Common Vsys Axis if none specified
  if commonXAxes is None:
    # Set vsys_range so that we don't extrapolate
    minVal = np.max([np.min(x_axis) for x_axis in x_axes])
    maxVal = np.min([np.max(x_axis) for x_axis in x_axes])

    spacing = np.max([getSpacing(x_axis) for x_axis in x_axes])
    commonXAxes = np.arange(minVal, maxVal, spacing)
  
  combined = combineSmudges(smudges, x_axes, commonXAxes, 
                  normalize=normalizeCombined)

  return combined, commonXAxes, kpRange
###

#-- Plotting/Writting Functions
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
              orb_params=None, target_Kp=None,
              target_Vsys=None, title="",
              saveName = None, cmap='viridis',
              xlim=[-100,100], ylim=None, clim=None,
              close=False, show=True, figsize=None
):
  """ Plots a smudge Plot
  """

  # Use units: km/s
  xs = vsys_axis/1000
  ys = kp_axis/1000

  # Offset xs,ys by 1/2 spacing for pcolormesh
  pltXs = xs-getSpacing(xs)/2
  pltYs = ys-getSpacing(ys)/2

  # Get Max of SmudgePlot
  ptmax = np.unravel_index(smudges.argmax(), smudges.shape)

  if xlim is not None:
    left_cut  = np.argmin(np.abs(xs - xlim[0]))
    right_cut = np.argmin(np.abs(xs - xlim[1]))
    cut_smudges = smudges[:,left_cut:right_cut+1]

    ptmax = np.unravel_index(cut_smudges.argmax(), cut_smudges.shape)
    ptmax = (ptmax[0], ptmax[1]+left_cut)
    
  ptx = xs[ptmax[1]]
  pty = ys[ptmax[0]]
  
  if figsize is None:
    plt.figure()
  else:
    plt.figure(figsize=figsize)
  # Plot smudges
  if clim is not None:
    plt.pcolormesh(pltXs,pltYs,smudges,cmap=cmap,
      vmin=clim[0],vmax=clim[1])
  else:
    plt.pcolormesh(pltXs,pltYs,smudges,cmap=cmap)
  cbar = plt.colorbar()

  # Plot Peak
  plt.scatter(ptx,pty, color='k')

  # Plot 'true' values if they exist
  true_val_str = ""
  if orb_params is not None:
    target_Kp   = orb_params['Kp']/1000
    target_Vsys = orb_params['v_sys']/1000

  if target_Kp is not None:
    plt.plot((pltXs[0],pltXs[-1]),(target_Kp,target_Kp),'r--')
  if target_Vsys is not None:
    plt.plot((target_Vsys,target_Vsys),(pltYs[0],pltYs[-1]),'r--')

  if target_Kp is not None and target_Vsys is not None:
    markYval = np.argmin(np.abs(ys - target_Kp))
    markXval = np.argmin(np.abs(xs - target_Vsys))

    true_val_str = "\nValue under cross: " + str(np.round(smudges[markYval,markXval],2))

  plt.ylabel("Kp (km/s)")
  plt.xlabel("V_sys (km/s)")

  if title!="":
    title += '\n'
  plt.title(title+'Max Value: '+ 
      str(np.round(smudges[ptmax],2)) + ': (' + str(np.round(ptx,1)) + ',' +
      str(int(np.round(pty,0))) + ')' + true_val_str)

  cbar.set_label('Sigma')

  if xlim is not None:
    plt.xlim(*xlim)
  else:
    plt.xlim(np.min(vsys_axis)/1000,np.max(vsys_axis)/1000)
  if ylim is not None:
    plt.ylim(*ylim)
  else:
    plt.ylim(np.min(kp_axis)/1000,np.max(kp_axis)/1000)

  # Allow mouseover to display Z value
  def fmt(x, y):
    col = np.argmin(np.abs(xs-x))
    row = np.argmin(np.abs(ys-y))
    z = smudges[row,col]
    return 'x=%1.1f, y=%1.1f, z=%1.2f' % (x, y, z)

  plt.gca().format_coord = fmt

  plt.tight_layout()
  if saveName is not None:
    plt.savefig(saveName)

  if show:
    plt.show()

  if close:
    plt.close()

def writeSysrem(file, planet, dates, orders, sysIts, templateName=None):
  # try to open file
  try:
    with open(file) as f:
      data = json.load(f)
  except (FileNotFoundError, UnicodeDecodeError, ValueError) as e:
    print('Input file must be pre-existing valid json file.')
    raise(e)

  # Check for inconsistent values
  if len(orders)*len(dates) != len(sysIts):
    raise ValueError("Number of input Sysrem Iterations doesn't equal number of dates times number of orders")

  n_orders = len(orders)

  #Enter SysremIterations into Json
  planetData = data[planet]
  for i, date in enumerate(dates):
    dateData = planetData['dates'][date]
    for j, order in enumerate(orders):
      k = i*n_orders + j
      orderData = dateData['order_kws'][str(order)]

      if templateName is None:
        # orders not broken up by templates
        orderData['sysremIterations'] = int(sysIts[k])
      else:
        if templateName not in orderData.keys():
          orderData[templateName] = {'sysremIterations': int(sysIts[k])}
        else:
          orderData[templateName]['sysremIterations'] = int(sysIts[k])

      dateData['order_kws'][str(order)] = orderData
    planetData['dates'][date] = dateData

  data[planet] = planetData
  # return data

  #Write new json to file 
  with open(file,'w') as f:
    json.dump(data, f, indent=2, sort_keys=True)

def getTitleStr(planet,date,order):
  return str(planet)+' '+str(date)+' order-'+str(order)
###

#-- Step 0: Raw Data
def setObs(planet, date, order, dataPaths, templateName=None):
  """
    dataPaths, orb_params, analysis_kws = setObs(planet, date, order, dataPaths)
  """
  orb_params, dates, planetName, dataPaths = setPlanet(planet, dataPaths, templateName=templateName)
  date_kws, order_kws, dataPaths = setDate(date, dates, dataPaths)
  analysis_kws, dataPaths = setOrder(order, date_kws, order_kws, dataPaths, templateName=templateName)

  analysis_kws['planetName'] = planetName
  return dataPaths, orb_params, analysis_kws

def getDates(planet, dataPaths):
  '''
    Used to get the available dates for a planet.

    Redundant but user friendly
  '''
  with open(dataPaths['planetData']) as f:
      data = json.load(f)
  data  = data[planet]
  dates = data['dates']
  return list(dates.keys())

def setPlanet(planet, dataPaths, verbose=False, templateName=None):
  with open(dataPaths['planetData']) as f:
      data = json.load(f)
  data = data[planet]
  orb_params   = data['orb_params']
  dates        = data['dates']
  planetName   = data['planetName']
  tem = data['template']

  if type(tem) == dict:
    if templateName is None:
      raise KeyError("templateName must be specified for planet "+planet+" in setPlanet")
    else:
      print('Using template: '+str(templateName))
      dataPaths['templateName'] = tem[templateName]
  else:
    dataPaths['templateName'] = tem
    
  if verbose:
    print('Possible Dates for '+planet+' are '+str([key for key in dates.keys()])+'.')
    
  return orb_params, dates, planetName, dataPaths

# TODO should set barycorvel, save in orb_params
def setDate(date, dates, dataPaths):
  date_kws = {**dates[date]}
  order_kws = date_kws.pop('order_kws')
  default_kws = date_kws.pop('default_kws')
  date_kws.update(default_kws)

  dataPaths['date'] = date

  return date_kws, order_kws, dataPaths

def setOrder(order, date_kws, order_kws, dataPaths, templateName=None):
  dataPaths['order'] = order

  try:
    order_kws = order_kws[str(order)]

    if dictDepth(order_kws) == 2:
      # KeyWords specified by template
      if templateName == None:
        raise KeyError("templateName must be specified for planet "+planet+" in setOrder")
      else:
        order_kws=order_kws[templateName]

  except KeyError:
    # order does not have specifc params : use default
    return date_kws,dataPaths

  date_kws.update(order_kws)
  return date_kws, dataPaths

def collectRawOrder(date, order, 
                   dirPre, dirPos, dataPre, dataPos,
                   header, templateDir, templateName, **kwargs):
  dataDir      = dirPre+date+dirPos
  dataFile     = dataDir+dataPre+str(order)+dataPos
  headerFile   = dataDir+header
  templateFile = templateDir+templateName

  data = readFile(dataFile)
  header = readFile(headerFile)
  templateData = readFile(templateFile)

  return data, header, templateData
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

  xcor = np.correlate(smooth-np.mean(smooth), step, 'valid')
  
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
    ax.plot(normalize(xcor, (-0.5,0)),label='Cross Correlation')
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
                  showPlots=False, figsize=(12,8), figTitle=""
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
          ncycles = 1,
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

  for cycle in range(ncycles):
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
    plt.figure(figsize=(12,8))
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
  unitRVs = getRV(**rv_params, **orb_params)
  barycentricCorrection = getBarycentricCorrection(**rv_params, verbose=verbose)

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
                        pbar = None,
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
    
    if pbar is not None:
      pbar.update()

  smudges = np.array(smudges)
  if stdDivide:
    smudges = smudges/np.apply_along_axis(percStd,1,smudges)[:,np.newaxis]

  if vsys_range is not None:
    goodCols = np.logical_and(vsys>=vsys_range[0], vsys<=vsys_range[1])
    goodCols = ndi.maximum_filter(goodCols,3)
    smudges = smudges[:, goodCols]
    vsys = vsys[goodCols]

  if retAxes:
    return smudges, vsys, kpRange
  else:
    return smudges

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
    retSmudge = retSmudge/np.apply_along_axis(percStd,1,retSmudge)[:,np.newaxis]
  return retSmudge
###

#-- Template Functions
def injectFakeSignal(flux, wave, rv_params, orb_params, 
                    fake_Kp, fake_Vsys, fake_signal_strength,
                    templateData = None, templateFile = None,
                    dataPaths = None, doAlert = True, verbose = False
):
  if doAlert:
    print('---------------------')
    print('Injecting Fake Signal')
    print('---------------------')

  # Get Template:
  # Priority of inputs -> dataPaths -> templatefile ->
  # templateData
  template_interp = None
  if dataPaths is not None:
    templateFile = dataPaths['templateDir'] + dataPaths['templateName']
  if templateFile is not None:
    templateData = readFile(templateFile)
  if templateData is not None:
    t_wave = templateData['wavelengths']/10000
    t_flux = templateData['flux']

    template_interp = interpolate.splrep(t_wave,t_flux)
  if template_interp is None:
    raise ValueError('Must specify a template in some form.')

  fake_signal = []
  orb_params = orb_params.copy()
  orb_params['Kp'] = fake_Kp
  orb_params['v_sys'] = fake_Vsys
  rvs = getRV(**rv_params, **orb_params)
  bc  = getBarycentricCorrection(**rv_params, verbose=verbose)
  rvs = rvs + bc + fake_Vsys

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

def getTemplate(planet, dataPaths, templateName=None):
  setPlanet(planet,dataPaths, templateName=templateName)
  templateFile = dataPaths['templateDir']+dataPaths['templateName']
  template = readFile(templateFile)
  return template['wavelengths']/10000, template['flux']
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

def dictDepth(d, level=0):
  if d == {}:
    return level+1
  elif not isinstance(d, dict) or not d:
    return level
  return max(dictDepth(d[k], level + 1) for k in d)
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

def rowNorm(data):
  return data/np.apply_along_axis(percStd,1,data)[:,np.newaxis]
###

#-- Physics
#TODO: 
  # phase func
  # w = w-180 equal to rv = -rv
  # switch to rv = -rv
def getRV(times, t0=0, P=0, w_deg=0, e=0, Kp=1, v_sys=0,
        vectorizeFSolve = False, returnPhase=False, **kwargs
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
  w = np.deg2rad(w_deg-180)
  mean_anomaly = ((2*np.pi)/P * (times - t0)) % (2*np.pi)

  if returnPhase:
    return mean_anomaly

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

  # TODO
  # velocity = Kp * (np.cos(true_anomaly+w) + e*np.cos(w)) + v_sys
  velocity = Kp * (np.cos(true_anomaly+w) + e*np.cos(w))

  return velocity

def getBarycentricCorrection(times, planetName, obsname, verbose=False, **kwargs):
  if verbose:
    print('collecting barycentric velocity')

  bc=[]
  for time in times:
    JDUTC = Time(time, format='jd',scale='utc')
    output=  barycorrpy.get_BC_vel(JDUTC, starname=planetName, obsname=obsname)
    bc.append(output[0][0])

  bc = np.array(bc)
  # Subtract BC to be in target frame
  bc = -bc 
  return bc

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