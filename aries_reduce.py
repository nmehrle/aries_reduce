"""A Python script to analyze NIRSPEC data.  Eventually it should be
a function (e.g. FGD's ultimate_automation), but for now it's just a
script.

This routine takes a set of NIRSPEC high-resolution data files and
uses IRAF and homebrew Python to extract meaningful spectral information.
Other routines will be used for manipulation of the data.

This will probably only run on Unix/Linux/Mac OSX platforms.

This to check on a new system:

1) You may want need to edit the default PYFITS.WRITETO function to
   add an 'output_verify' keyword.

2) PyRAF's "apnormalize" routines contain a parameter file called
   "apnorm1.par", which contains references to "apnorm.background" --
   these must all be changed to "apnormalize.background"

Other notes:

1) If the spectral tracing keeps crashing ("Trace of aperture N lost
   at line X"), try fiddling with the minsep/maxsep parameters.



2008-06-10 21:28 IJC: Created.
2008-07-22 15:04 IJC: Split up "procData" into "procCal" and "procTarg"
2008-07-25 16:19 IJC: Finished initial version; renamed ns_reduce
2008-11-25 15:29 IJC: Added fix_quadnoise step and individual frame
                      cosmic ray rejection.
2008-12-05 15:48 IJC: Switched to linear wavelength interpolation,
                      since this will simplify things for LSD
2008-12-16 17:12 IJC: Trying it for a second dataset
2009-04-28 10:03 IJC: Beginning to add the L-band data interface.
                      Updated interface to make better use of
                      nsdata.initobs.  Flat field is now padded on
                      both sides for order-tracing.
2009-07-09 17:32 IJC: Switched pyfits.writeto calls to use 'output_verify=ignore'
2010-09-06 10:43 IJC: Added H-filter option to horizsamp; added
                      cleanec option to preprocess calls.

2012-04-04 15:05 IJMC: E puor si muove!  Added flat_threshold option;
                       subtly changed a few options to new
                       defaults. Set shift=0 in calls to ecreidenfity.

2014-12-17 14:17 IJMC: Added new troubleshooting & alternative
                       flat-normalization approaches, since PyRAF's
                       apnormalize continues to give me trouble.

2016-10-15 02:09 IJMC: Trying it for ARIES. This script has been
                       around for a little while!
2016-10-18 13:50 IJMC: Now apply ARIES quad-detector crosstalk correction
2017-10-11 16:41 IJMC: Handing this off to Nicholas Mehrle. Good luck!
2017-10-25 ????? NM  : Moved version control to Git
"""


import os, sys, shutil
from pyraf import iraf as ir
ir.prcacheOff()
# Might be necessary for some iraf tasks with multiprocessing, unsure
ir.set(writepars=0)


from scipy import interpolate, isnan, isinf

try:
    from astropy.io import fits as pyfits
except:
    import pyfits

import nsdata as ns
import spec
import numpy as ny
from pylab import find
import pdb
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

################################################################
##################### User Input Variables #####################
################################################################

data = '2016oct15' # GX And
data = '2016oct15b' # WASP-33
# data = '2016oct19' # WASP-33
# data = '2016oct20b' # WASP-33
# data = '2016oct16' #Ups And

# Optional change in directory structure for Exobox
local = False

# Determines which subroutines to run
makeDark    = False
makeFlat    = False
makeMask    = False

# Calibration Frames
preProcCal  = True
processCal  = False
calApp      = False

# Target Frames
preProcTarg = False
processTarg = False

# find target aperatures from full data list.


# Should only need to be run once for each dataset
idTargAperatures = False

# SaveAsPickleFiles
# Recommended to use the python 3 routine pickler.py
pickleFiles = False

# WhatToSave
# For PreprocessTarg
saveBadMask        = False

# For ProcessTarg
saveCorrectedImg   = False #(Output of Preprocess)
saveUnInterpolated = False


# Telluric Correction
telluricCorrect = False

#Treats flats as altitude dependent if possible
angledFlats = True

# run IRAF in interactive mode (set true)
interactive = True

# Use iraf.apflatten to flatten blaze fn
# If false uses Ian's custom routine
# Part of Make Flat
irafapflatten = True

verbose = True


dispersion = 0.075  # Resampled dispersion, in angstroms per pixel (approximate)

flat_threshold = 500

# Set number of processors to use for processTarg
#  0 : max number of processors on your machine
# -1 : all but 1 processor on your machine
# -2 : prompt user
num_processors = -2

dir0 = os.getcwd()


# User Specific Directories
dir_data = data
if data[-1].isalpha():
    dir_data = data[:-1]

if local:
    _iraf = ns._home + "/iraf/"
    _raw  = ns._home + "/documents/science/spectroscopy/" + dir_data +"/raw/"
    _proc = ns._home + "/documents/science/spectroscopy/" + dir_data +"/proc/"
    telluric_list = ns._home + '/documents/science/spectroscopy/telluric_lines/hk_band_lines.dat'
    _corquad = ns._home+'/documents/science/codes/corquad/corquad.e'
else:
    _raw  = "/dash/exobox/proj/pcsa/data/raw/"  + dir_data + "/spec/"
    _proc = "/dash/exobox/proj/pcsa/data/proc/" + data + "/"
    _corquad = "/dash/exobox/code/python/nmehrle/corquad/corquad.e"
    _iraf = "/dash/exobox/code/python/nmehrle/iraf"


################################################################
################### END User Input Variables ###################
################################################################

# Initialize routine
# Set in if statement to enable code-folding
if True:
    num_available_cpus = mp.cpu_count()
    if num_processors == -2:
        if processCal or processTarg or preProcCal or preProcTarg:
            print('This machine has '+ str(num_available_cpus) +" CPUs available. \nInput how many you'd like to use:")
            num_processors = int(raw_input())
        else:
            num_processors = 1
    elif num_processors > num_available_cpus or num_processors == -1:
        num_processors = num_available_cpus -1
    elif num_processors == 0:
        num_processors = num_available_cpus

    # Check if _raw exists
    if(not os.path.exists(_raw)):
        raise IOError('No such file or directory '+_raw+'. Update _raw to point to directory containing raw data.')
    # If interactive and _proc doesn't exist, attempt to create it
    if interactive and not os.path.exists(_proc):
        print 'Attempting to create processed data directory at: \n'+_proc
        print 'Input "yes" to allow directory creation.'
        _proc_input = raw_input()
        if _proc_input.lower() == 'yes':
          os.makedirs(_proc)
    # If proc still doesn't exist, abort
    if not os.path.exists(_proc):
        raise IOError('No such file or directory '+_proc+'. Update _proc to point to processed data directory.')


    # Grab observational data from database
    obs = ns.initobs(data, remote=(not local), _raw=_raw, _proc=_proc)

    # reprocess obs dict into vars
    _proc  = obs['_proc']
    _raw   = obs['_raw']
    n_ap   = obs['n_aperture']  #  number of apertures (i.e., echelle orders)
    filter = obs['filter']  # photometric band in which we're operating
    prefn  = str(obs['prefix'])  # filename prefix
    calnod = obs['calnod'] # whether A0V calibrators nod, or not
    db_pre = obs['ap_suffix']


    procData = processCal or processTarg
    preProcData = preProcCal or preProcTarg
    badval = 0
    ir.task(bfixpix = _iraf+"bfixpix.cl")
    ir.task(bfixpix_one = _iraf+"bfixpix_one.cl")
    #ir.load('fitsutil')
    ir.load('noao')
    ir.load('astutil')
    ir.load("imred")
    ir.load('echelle')
    ir.load('twodspec')
    ir.load('apextract')

    if processCal and not os.path.exists(telluric_list):
        raise IOError('No such file or directory '+telluric_list+'. Update telluric_list to point to file with telluric line list for your data.')

    if filter=='K' or filter=='H':
        horizsamp = "10:500 550:995"

    elif filter=='L':
        horizsamp = "10:270 440:500 550:980"

    elif filter=='Karies' or filter=='OPEN5':
        horizsamp = "10:995"

    if filter=='Karies' or filter=='OPEN5':
        observ = 'flwo'
        itime = 'exptime'
        date = 'UTSTART'
        time = None
        dofix = True
        t_width = 15.
        trace_step = 10
        trace_order = 3
        quadcorrect = True # Correct for detector crosstalk
    else:
        observ = 'keck'
        itime = 'itime'
        date = 'date-obs'
        time = 'UTC'
        dofix = True
        t_width = 115.
        trace_step = 50
        trace_order = 7
        quadcorrect = False # Correct for detector crosstalk

    if filter=='K':
        cleanec = True
        cleancr = False
        qfix = True
        csigma=25
        cthreshold=400
        rratio = 5
        rthreshold = 300
    elif filter=='H':
        cleanec = False
        cleancr = True
        csigma=30
        cthreshold=900
        qfix = False
        rratio = 5
        rthreshold = 300
    elif filter=='L':
        cleanec = True
        cleancr = False
        qfix = True
        csigma=25
        cthreshold=400
        rratio = 5
        rthreshold = 300
    elif filter=='Karies' or filter=='OPEN5':
        cleanec = True
        cleancr = False
        qfix = 'aries'
        csigma=25
        cthreshold=400
        rratio = 5
        rthreshold = 300
    else:
        qfix = True



    bsamp   = "-18:-10,10:18"
    bfunc = 'chebyshev'
    bord = 3   # background subtraction function order

    idlexec = os.popen('which idl').read().strip()

    postfn  = ".fits"
    maskfn  = ".pl"

    ##### Sets keywords for filenames
    ################################################################
    ################################################################
    _sflat     = _proc + prefn + "_flat"
    _sflats    = _proc + prefn + "_flat_sig"
    _sflatdc   = _proc + prefn + "_flatd"
    _sflatdcn  = _proc + prefn + "_flatdn"

    _sdark = _proc + prefn + "_dark"
    _sdarks = _proc + prefn + "_dark_sig"
    _sdarkflat = _proc + prefn + "_darkflat"
    _sdarkflats = _proc + prefn + "_darkflats"
    _sdarkcal  = _proc + prefn + "_darkcal"
    _sdarkcals  = _proc + prefn + "_darkcals"

    _mask1  = _proc + prefn + "_badpixelmask1"  + maskfn
    _mask2  = _proc + prefn + "_badpixelmask2"  + maskfn
    _mask3  = _proc + prefn + "_badpixelmask3"  + maskfn
    _mask  = _proc + prefn + "_badpixelmask"  + maskfn
    _fmask  = _proc + prefn + "_flatpixelmask"  + maskfn
    _dmask  = _proc + prefn + "_darkpixelmask"  + postfn
    _wldat = 'ec'

    rawdark  = ns.strl2f(_proc+'rawdark', obs['darkfilelist'], clobber=True)
    rawdarkflat = ns.strl2f(_proc+'rawdarkflat', obs['darkflatlist'], clobber=True)
    rawdarkcal  = ns.strl2f(_proc+'rawdarkcal', obs['darkcallist'], clobber=True)

    # Determines if flats are angle dependent or not
    rawflat_list = []
    rawflat_dict = None
    flats_as_dict = False
    if type(obs['flatfilelist']) == dict:
        rawflat_dict = obs['flatfilelist']
        rawflat_list = [item for sublist in obs['flatfilelist'].values() for item in sublist]
        rawflat_list.sort()
        if angledFlats:
            flats_as_dict = True
    else:
        rawflat_list = obs['flatfilelist']

    procflat_list = [el.replace(_raw, _proc) for el in rawflat_list]
    procflat  = ns.strl2f(_proc+'procflat', procflat_list, clobber=True)

    if flats_as_dict:
        procflat_dict      = {}
        procflatfile_dict  = {}
        _sflat_dict        = {}
        _sflats_dict       = {}
        _sflatdc_dict      = {}
        _sflatdcn_dict     = {}


        for key in rawflat_dict.keys():
            procflat_dict[key] = [el.replace(_raw, _proc) for el in rawflat_dict[key]]
            procflatfile_dict[key] = ns.strl2f(_proc+'procflat_'+key, procflat_dict[key], clobber=True)
            _sflat_dict[key]    = _sflat    + '_' + key
            _sflats_dict[key]   = _sflats   + '_' + key
            _sflatdc_dict[key]  = _sflatdc  + '_' + key
            _sflatdcn_dict[key] = _sflatdcn + '_' + key

    rawcal   = ns.strl2f(_proc+'rawcal',   obs['rawcalfilelist'], clobber=True)
    proccal  = ns.strl2f(_proc+'proccal',  obs['proccalfilelist'], clobber=True)
    rawtarg  = ns.strl2f(_proc+'rawtarg',  obs['rawtargfilelist'], clobber=True)
    proctarg = ns.strl2f(_proc+'proctarg', obs['proctargfilelist'], clobber=True)
    speccal  = ns.strl2f(_proc+'speccal',  obs['speccalfilelist'], clobber=True)
    spectarg = ns.strl2f(_proc+'spectarg', obs['spectargfilelist'], clobber=True)
    fullproctarg = ns.strl2f(_proc+'fullproctarg', obs['fullproctargfilelist'], clobber=True)

    meancal  =  prefn + 'avgcal'

    ################################################################
    ################################################################

    ir.unlearn('imcombine')
    ir.unlearn('echelle')

    # Set parameters for aperture tracing, flat-field normalizing, etc.
    ###################################################################
    ###################################################################
    ir.apextract.dispaxis = 1

    ir.echelle.dispaxis = 1
    ir.echelle.apedit.width = t_width
    ir.echelle.apfind.minsep = 10.
    ir.echelle.apfind.maxsep = 150.
    ir.echelle.apfind.nfind = n_ap
    ir.echelle.apfind.recenter = "Yes"
    ir.echelle.apfind.nsum = -3

    ir.apall.ylevel = "INDEF" #0.05
    ir.apall.bkg = "Yes"
    ir.apall.ulimit = 2
    ir.apall.llimit = -2


    ir.aptrace.order = trace_order
    ir.aptrace.niterate = 3
    ir.aptrace.step = trace_step
    ir.aptrace.naverage = 1
    ir.aptrace.nlost = 999
    ir.aptrace.recenter = "yes"

    # Set detector properties:
    gain = 4.0  # photons (i.e., electrons) per data unit
    readnoise = 10.0   # photons (i.e., electrons)
    ir.imcombine.gain = gain
    ir.imcombine.rdnoise = readnoise
    ir.apall.gain = gain
    ir.apall.readnoise = readnoise
    ir.apnormalize.gain = gain
    ir.apnormalize.readnoise = readnoise


    ir.set(observatory=observ)
###################################################################
###################################################################

# Combine dark frames into a single dark frame:
# See labbook for more details :(
if makeDark:
    ir.imdelete(_sdark)
    ir.imdelete(_sdarks)
    ir.imdelete(_sdarkflat)
    ir.imdelete(_sdarkflats)
    ir.imdelete(_sdarkcal)
    ir.imdelete(_sdarkcals)


    if verbose: print "rawdark file list >>\n" + rawdark
    ir.imcombine("@"+rawdark, output=_sdark, combine="average",reject="avsigclip", sigmas=_sdarks, scale="none", weight="none", bpmasks="")
    ns.write_exptime(_sdark, itime=itime)

    if verbose: print "rawdarkflat file list >>\n" + rawdarkflat
    ir.imcombine("@"+rawdarkflat, output=_sdarkflat, combine="average",reject="avsigclip", sigmas=_sdarkflats, scale="none", weight="none", bpmasks="")
    ns.write_exptime(_sdarkflat, itime=itime)

    if verbose: print "rawdarkcal file list >>\n" + rawdarkcal
    ir.imcombine("@"+rawdarkcal, output=_sdarkcal, combine="average",reject="avsigclip", sigmas=_sdarkcals, scale="none", weight="none", bpmasks="")
    ns.write_exptime(_sdarkcal, itime=itime)

    if verbose: print "Done making dark frames!"
###################################################################
###################################################################

if makeFlat:  # 2008-06-04 09:21 IJC: dark-correct flats; then create super-flat
    if verbose:
        print "Making flat frames"
        print "----------------------------------------"
        print "----------------------------------------"
        print "----------------------------------------"
    ir.imdelete(_sflat)
    # ir.imdelete(_sflats)
    ir.imdelete(_sflatdc)
    ir.imdelete(_sflatdc+'big')

    if flats_as_dict:
        for angle in _sflat_dict.keys():
            ir.imdelete(_sflat_dict[angle])
            # ir.imdelete(_sflatsdict[angle])

            ir.imdelete(_sflatdc_dict[angle])
            ir.imdelete(_sflatdc_dict[angle]+'big')

            ir.imdelete(_sflatdcn_dict[angle])
            ir.imdelete(_sflatdcn_dict[angle]+'big')
    else:
      ir.imdelete(_sflatdcn)
      ir.imdelete(_sflatdcn+'big')

    # Correct for dectector crosstalk
    if verbose:

        print 'Correcting aries crosstalk'
        print "----------------------------------------"


    ns.correct_aries_crosstalk(rawflat_list, output=procflat_list, corquad=_corquad)

    if verbose:
        print "----------------------------------------"
        print 'Done correcting aries crosstalk'

    # 2008-06-04 08:42 IJC: Scale and combine the flats appropriately (as lamp is warming up, flux changes)

    # Makes median flat(s)
    if verbose:
        print "Combining flat fields"
        print "----------------------------------------"
    def combineflats(inflats, outflat, outflatdc, darkflat,flat_sigmas=None):
        ir.imcombine("@"+inflats,output=outflat, combine="average",reject="crreject", scale="median", weight="median", bpmasks="") # sigmas=flat_sigmas
        ns.write_exptime(outflat, itime=itime)
        print(outflat)

        ir.ccdproc(outflat, output=outflatdc, ccdtype="", fixpix="no", overscan="no",trim="no",zerocor="no",darkcor="yes",flatcor="no", dark=darkflat)
    #master flat
    combineflats(procflat, _sflat, _sflatdc, _sdarkflat) #flat_sigmas = _sflats

    #angle dependent flats
    if flats_as_dict:
        for angle, flatlist in procflatfile_dict.items():
            combineflats(flatlist, _sflat_dict[angle], _sflatdc_dict[angle],_sdarkflat) #flat_sigmas = _sflats_dict[angle]

    if verbose:
        print "----------------------------------------"
        print "Done Combining flat frame(s)!"

    # Corrects blaze function (Flattens flat frames)
    if verbose:
        print "----------------------------------------"
        print "Correcting for flat field blaze functions"
        print "----------------------------------------"

    def correctblazefn(inflat, outflat):
        #Create padded file to get aperatures on edges
        flatdat = pyfits.getdata(  inflat+postfn)
        flathdr = pyfits.getheader(inflat+postfn)

        n_big = 1400
        n_base = flatdat.shape[0]
        pad = (n_big-n_base)/2
        bigflat = ny.zeros([n_big,n_base])

        bigflat[pad:(pad+n_base),:] = flatdat
        pyfits.writeto(inflat+'big'+postfn, bigflat, flathdr, overwrite=True, output_verify='warn')

        # Flatten Iraf or otherwise
        if irafapflatten:
            ir.apflatten(inflat+'big', outflat+'big', sample=horizsamp, niterate=1, threshold=flat_threshold, function="spline3", pfit = "fit1d", clean='yes',  recenter='yes', resize='yes', edit='yes', trace='yes', fittrace='yes', interactive=interactive, order=3)
        else:
            mudflat = pyfits.getdata(inflat + 'big.fits')
            mudhdr = pyfits.getheader(inflat + 'big.fits')
            trace = spec.traceorders(inflat + 'big.fits', pord=2, nord=ir.aptrace.order, g=gain, rn=readnoise, fitwidth=100)
            normflat = spec.normalizeSpecFlat(mudflat*gain, nspec=ir.aptrace.order, traces=trace)
            pyfits.writeto(outflat + 'big.fits', normflat, header=mudhdr, output_verify='warn')

        # Remove Padding
        normflatdat = pyfits.getdata(  outflat+'big'+postfn)
        normflathdr = pyfits.getheader(outflat+'big'+postfn)
        smallnormflat = normflatdat[pad:(pad+n_base),:]
        smallnormflat[smallnormflat==0] = 1.
        pyfits.writeto(outflat+postfn, smallnormflat, normflathdr, overwrite=True, output_verify='warn')

    #apply correction to each dc flat
    #if flats come as a dict, don't correct the overall (no angle dependence) flat
    if flats_as_dict:
        for angle,dc_flat in _sflatdc_dict.items():
            correctblazefn(dc_flat, _sflatdcn_dict[angle])
    else:
        correctblazefn(_sflatdc, _sflatdcn)

    if verbose:
        print "----------------------------------------"
        print "Done Correcting for blaze fn"

    if verbose:
        print "----------------------------------------"
        print "----------------------------------------"
        print "----------------------------------------"
        print "Done making flat frame(s)!"
###################################################################
###################################################################

if makeMask:
    if verbose:
        print "Beginning to make bad pixel masks..."
        print "----------------------------------------"
        print "----------------------------------------"


    # iterate through the superflat 3 times to get bad pixels, then
    # construct a super-bad pixel map.
    ir.load('crutil')

    ir.imdelete(_mask)
    ir.imdelete(_fmask)
    ir.imdelete(_dmask)
    ir.imdelete(_mask.replace(maskfn, postfn))
    ir.imdelete(_fmask.replace(maskfn, postfn))
    ir.imdelete(_dmask.replace(postfn, maskfn))

    ir.delete('blah.fits')
    ir.delete('blahneg.fits')

    #ir.cosmicrays(_sflatdc, 'blah', crmasks=_mask1, threshold=750, npasses=7q
    #  , \
    #                  interactive=False) #interactive)
    ns.cleanec(_sflatdc, 'blah', npasses=5, clobber=True, badmask=_mask1.replace(maskfn, postfn), verbose=verbose)
    #ir.imcopy(_mask1, _mask1.replace(maskfn, postfn))
    #pyfits.writeto(_mask1, ny.zeros(pyfits.getdata(_sflatdc+postfn).shape, dtype=int), clobber=True)
    pyfits.writeto(_sflatdc+'neg', 0. - pyfits.getdata(_sflatdc+postfn), clobber=True)
    #ir.cosmicrays(_sflatdc+'neg', 'blahneg', crmasks=_mask2, threshold=750, npasses=7) #, \
    #                      interactive=interactive)

    ns.cleanec(_sflatdc+'neg', 'blahneg', npasses=5, clobber=True, badmask=_mask2.replace(maskfn, postfn))
    #pyfits.writeto(_mask2, ny.zeros(pyfits.getdata(_sflatdc+postfn).shape, dtype=int), clobber=True)

    # create a final binary mask from the 2 masks:
    #ir.imcalc(_mask1+","+_mask2, _fmask, "im1||im2")
    pyfits.writeto(_fmask.replace(maskfn, postfn), ny.logical_or(pyfits.getdata(_mask1.replace(maskfn, postfn)), pyfits.getdata(_mask2.replace(maskfn, postfn))).astype(int), clobber=True)
    #ir.imcopy(_fmask.replace(maskfn, postfn), _fmask)

    # clean up after myself:
    ir.imdelete(_mask1+','+_mask2+','+_sflatdc+'neg,blah,blahneg')

    # Examine the dark frames for highly variable pixels:
    ns.darkbpmap(obs['darkfilelist'], clipsigma=5, sigma=10, writeto=_dmask, clobber=True, verbose=verbose, outtype=float)
    #pyfits.writeto(_dmask, ny.zeros(pyfits.getdata(_sflatdc+postfn).shape, dtype=int), clobber=True)
    try:
        ir.imcopy(_dmask, _dmask.replace(postfn, maskfn))
    except:
        print "couldn't imcopy " + _dmask

    # Combine the flat-field- and dark-frame-derived pixel masks:
    #ir.imcalc(_fmask+","+_dmask, _mask, "im1||im2")
    pyfits.writeto(_mask.replace(maskfn, postfn), ny.logical_or(pyfits.getdata(_fmask.replace(maskfn, postfn)), pyfits.getdata(_dmask)).astype(float), clobber=True)
    ir.imcopy(_mask.replace(maskfn, postfn), _mask)



    if verbose:
        print "Done making bad pixel mask!"
        print "----------------------------------------"
###################################################################
###################################################################

if preProcData:
    flat_for_proc = _sflatdcn
    if flats_as_dict: flat_for_proc = _sflatdcn_dict

    if preProcCal:
        # Add 'exptime' header to all cal, target, and lamp files:
        ns.write_exptime(rawcal, itime=itime)

        # Correct for bad pixels and normalize all the frames by the flat field
        # will edit for multiple flats
        ir.load('crutil')

        ns.preprocess('@'+rawcal, '@'+proccal, qfix=qfix,
            qpref='', flat=flat_for_proc, dark=_sdarkcal,
            mask=_mask.replace(maskfn, postfn),
            cleanec=cleanec, clobber=True, verbose=verbose,
            csigma=csigma, cthreshold=cthreshold,
            cleancr=cleancr, rthreshold=rthreshold, rratio=rratio,
            date=date, time=time, dofix=dofix, corquad=_corquad,
            num_processors=num_processors, saveBadMask=saveBadMask, tryIRccdproc=False, badPixMethod='linear')

    if preProcTarg:
        ns.write_exptime(rawtarg, itime=itime)

        ns.preprocess('@'+rawtarg, '@'+proctarg, qfix=qfix,
            qpref='', flat=flat_for_proc, dark=_sdark,
            mask=_mask.replace(maskfn, postfn),
            cleanec=cleanec, clobber=True, verbose=verbose,
            csigma=csigma, cthreshold=cthreshold,
            cleancr=cleancr, rthreshold=rthreshold, rratio=rratio,
            date=date, time=time, dofix=dofix, corquad=_corquad,
            num_processors=num_processors, saveBadMask=saveBadMask,tryIRccdproc=False, badPixMethod='linear')

    if verbose: print "Done correcting cal frames for bad pixels, dark correcting, and flat-fielding!"


if procData:
    os.chdir(_proc)
    ir.chdir(_proc)

    if processCal:
        if calApp:
        # Extract raw spectral data from the echelle images
          ir.imdelete('@'+speccal)
          ir.apall('@'+proccal, output='@'+speccal, format='echelle', recenter='yes',resize='yes',extras='yes', nfind=n_ap, nsubaps=1, minsep=10, weights='variance', bkg='yes', b_function=bfunc, b_order=bord, b_sample=bsamp, b_naverage=-3, b_niterate=2, t_order=3, t_sample=horizsamp, t_niterate=3, t_naverage=3, background='fit', clean='yes', interactive=interactive, nsum=-10, t_function='chebyshev')

        if verbose:  print "Done extracting spectra from cal stars!"

        ir.imdelete(meancal)
        if calnod:
            shutil.copyfile(obs['speccalfilelist'][0]+postfn, meancal+postfn)
        else:
            ir.imcombine('@'+speccal, meancal, combine='average', reject='avsigclip', weight='median')


        # Construct wavelength solution; apply to all observations.
        print "First identify lines in each of SEVERAL ORDERS using 'm'.  After this, use 'l' to fit dispersion solution.  Maybe then it can find more lines automatically.  Then, use 'f' to fit a dispersion function.  Then use 'o' and set the order offset to 38 (in standard K-band NIRSPEC mode)"
        sys.stdout.flush()
        ir.ecidentify(meancal, database=_wldat, coordlist=telluric_list, ftype='absorption', fwidth='10', niterate=3, low=5, high=5, xorder=3, yorder=3)

        disp_soln = ns.getdisp(_wldat + os.sep + 'ec' + meancal)

        w = ns.dispeval(disp_soln[0], disp_soln[1], disp_soln[2], shift=disp_soln[3])
        w = w[::-1]
        hdr = pyfits.getheader(meancal+postfn)
        pyfits.writeto('wmc'+postfn, w, hdr, clobber=True, output_verify='ignore')

        w_interp   = ns.wl_grid(w, dispersion, method='linear')
        #w_interp = w_interp[ny.argsort(w_interp.mean(1))]
        hdr_interp = pyfits.getheader(meancal+postfn)

        pyfits.writeto('winterp'+postfn, w_interp, hdr_interp, clobber=True, output_verify='ignore')
        ns.interp_spec(meancal, w, w_interp, k=3.0, suffix='int', badval=badval, clobber=True,verbose=True)

        # Sample each aperture so that they all have equal pixel widths
        #   and equal  wavelength coverage:
        ir.ecreidentify('@'+speccal,  meancal, database=_wldat, refit='no', cradius=10., shift=0)

        filelist = open(speccal)
        for line in filelist:
            filename = line.strip()
            disp_new = ns.getdisp(_wldat+'/ec' + filename)
            w_new = ns.dispeval(disp_new[0], disp_new[1], disp_new[2], shift=disp_new[3])
            w_new = w_new[::-1]
            ns.interp_spec(filename, w_new, w_interp, k=3.0, suffix='int', badval=badval, clobber=True)
        filelist.close()

    ##########################################

    if processTarg:

        if idTargAperatures:
            # We take the median dataFrame and identify/trace aperatures on it
            # We then pass this as a reference to apall on all data frames
            _targap  = _proc+prefn+"_targap"
            _targaps = _proc+prefn+"_targaps"

            ir.imdelete(_targap)
            ir.imdelete(_targaps)
            ir.imcombine("@"+fullproctarg, output=_targap, combine="average",reject="avsigclip", sigmas=_targaps, scale="none", weight="median", bpmasks="")

            ir.apfind(_targap, interactive=interactive, nfind=n_ap, minsep=10)
            ir.aptrace(_targap, interactive=interactive, recenter='no', resize='no', function='chebyshev', order=3, sample=horizsamp, naverage=3,niterate=3)
            # ap_ref = db_pre+prefn+"_targap"

            if verbose:
                print "\n\n"
                print "Identified Aperatures"

        ir.imdelete('@'+spectarg)
        list_proctarg = ny.loadtxt(proctarg,str)
        list_spectarg = ny.loadtxt(spectarg,str)

        num_frames = len(list_proctarg)

        apall_kws = {
          'references'  : _targap,
          'format'      : 'echelle',
          'recenter'    : 'yes',
          'resize'      : 'yes',
          'extras'      : 'yes',
          'trace'       : 'no',
          'nfind'       : n_ap,
          'nsubaps'     : 1,
          'minsep'      : 10,
          'bkg'         : 'yes',
          'b_function'  : bfunc,
          'b_order'     : bord,
          'b_sample'    : bsamp,
          'b_naverage'  : -3,
          'b_niterate'  : 2,
          't_order'     : 3,
          't_sample'    : horizsamp,
          't_niterate'  : 3,
          't_naverage'  : 3,
          'background'  : 'fit',
          'clean'       : 'yes',
          'interactive' : False,
          'nsum'        : -10,
          't_function'  : 'chebyshev'
        }

        def processEachTarg(i, input_list, output_list, apall_kws):
            ir.apall(input_list[i], output=output_list[i],**apall_kws)
            if saveCorrectedImg == False:
                ir.imdelete(input_list[i])

        pbar = tqdm(total = num_frames)
        pool = mp.Pool(processes = num_processors)

        for i,_ in tqdm(enumerate(pool.imap_unordered(
                        partial(processEachTarg,
                              input_list  = list_proctarg,
                              output_list = list_spectarg,
                              apall_kws   = apall_kws),
                        xrange(num_frames)))):
            pbar.update()
        pbar.close()
        if verbose:  print "Done extracting spectra from target stars!"

        # Sample each aperture so that they all have equal pixel widths
        #   and equal logarithmic wavelength coverage:
        ir.ecreidentify('@'+spectarg, meancal, database=_wldat, refit='no', shift=0)

        disp_soln = ns.getdisp(_wldat + os.sep + 'ec' + meancal)

        w = ns.dispeval(disp_soln[0], disp_soln[1], disp_soln[2], shift =disp_soln[3])
        w = w[::-1]
        #w_interp = ns.wl_grid(w, dispersion, method='linear')
        w_interp = pyfits.getdata('winterp.fits')
        hdr_interp = pyfits.getheader(meancal+postfn)

        filelist = open(spectarg)
        lines    = filelist.readlines()

        def interp_single_spec(i, lines):
            filename = lines[i].strip()
            disp_new = ns.getdisp(_wldat+'/ec' + filename)
            w_new    = ns.dispeval(disp_new[0], disp_new[1], disp_new[2], shift=disp_new[3])
            w_new = w_new[::-1]
            ns.interp_spec(filename, w_new, w_interp, k=3.0, suffix='int', badval=badval, clobber=True, verbose=False)
            if saveUnInterpolated == False:
                ir.imdelete(filename)

        pbar = tqdm(total = len(lines))
        pool = mp.Pool(processes = num_processors)

        if verbose:
            print('\nUpsampling Spectra\n')

        for i,_ in tqdm(enumerate(pool.imap_unordered(
                        partial(interp_single_spec,
                                lines=lines),
                        xrange(len(lines))))):
            pbar.update()

        pbar.close()
        filelist.close()

if telluricCorrect:
    # Write target and Mean Standard to text files for telluric correction:
    # Should be looped over
    ns.wspectext(filename + 'int', wlsort=True)
    ns.wspectext(meancal  + 'int', wlsort=True)

    print 'Instructions for IDL XTELLCOR:\n'
    print 'Std Spectra is: ' + meancal
    print 'Obj Spectra is: ' + filename
    print 'Units need to be set to Angstroms!  Remove the 2.166 um feature. '
    print 'Make sure to get the velocity shift correction correctly.'
    print 'At the end, make sure you write out both Telluric and A0V files.'
    sys.stdout.flush()
    os.system('cd ' + _proc + '\n' + idlexec + ' -e xtellcor_general')

    # Get telluric filename:
    _telluric = ''
    while (not os.path.isfile(_telluric)) and _telluric!='q':
        temp = os.listdir('.')
        print '\n\nEnter the telluric filename (q to quit); path is unnecessary if\n you saved it in the processed-data directory.  Local possibilities:'
        for element in temp:
            if element.find('tellspec')>-1: print element
        _telluric = raw_input('Filename:   ')

    if _telluric=='q':
        pass
    else:
        # Read telluric file; put in the right format.
        objspec_telcor = ny.loadtxt(_telluric.replace('_tellspec', ''))
        objspec_raw = ny.loadtxt(filename + 'int.dat')


        infile = open(_telluric, 'r')
        data = [map(float,line.split()) for line in infile]
        infile.close()
        n = len(data)
        data = ny.array(data).ravel().reshape(n, 3)
        telluric = data.transpose().reshape(3, n_ap, n/n_ap)
        telluric = telluric[1:3,:,:]
        tl_shape = telluric.shape
        telluric = telluric.ravel()
        nanind = find(isnan(telluric))
        infind = find(isinf(telluric))
        ind = ny.concatenate((nanind, infind))
        telluric[ind] = badval
        telluric = telluric.reshape(tl_shape)

        telluric2 = objspec_raw[:,1] / objspec_telcor[:,1]
        telluric2_err = telluric2 * ny.sqrt((objspec_raw[:,2]/objspec_raw[:,1])**2 + (objspec_telcor[:,2]/objspec_telcor[:,1])**2)
        telluric2_err[np.logical_not(np.isfinite(telluric2))] = badval
        telluric2[np.logical_not(np.isfinite(telluric2))] = badval
        telluric2_err /= np.median(telluric2)
        telluric2 /= np.median(telluric2)
        invtelluric3 = np.vstack((telluric2, telluric2_err)).reshape(tl_shape)

        tel_scalefac = np.median(telluric)
        telluric = telluric / tel_scalefac


        # Divide all target frames by the telluric corrector:
        filelist = open(spectarg)
        for line in filelist:
            filename = line.strip() + 'int'
            hdr  = pyfits.getheader(filename + postfn)
            data = pyfits.getdata(  filename + postfn)
            data = data[ [0,-2], ::-1, :]
            newdata = ny.zeros(data.shape)
            newspec = data[0,:,:] * telluric[0,:,:]
            ns_shape = newspec.shape
            tempdata = newspec.ravel()
            nanind = find(isnan(tempdata))
            infind = find(isinf(tempdata))
            ind = ny.concatenate((nanind, infind))
            tempdata[ind] = badval
            newspec = tempdata.reshape(ns_shape)
            newerr  = newspec * ny.sqrt((data[1,:,:]/data[0,:,:])**2 + (telluric[1,:,:]/telluric[0,:,:])**2)
            newdata[0,:,:] = newspec;
            newdata[1,:,:] = newerr
            hdr.update('TELLURIC', 'Telluric-corrected with file ' + _telluric)
            pyfits.writeto(filename + 'tel' + postfn, newdata[:,::-1], header=hdr, overwrite=True, output_verify='ignore')
        filelist.close()



if pickleFiles:
    import pickle
    import numpy as np
    from astropy.io import fits

    if verbose:
        print "Begining Pickling Process"
        print "Reading Data..."

    # Computer can only handle so many open files at a time
    maxChunkSize = 123
    suffix    = 'int.fits'
    list_spectarg = ny.loadtxt(spectarg,str)

    list_spectarg = list_spectarg

    subset_ranges = np.array_split(list_spectarg, int(len(list_spectarg)/maxChunkSize) +1)

    fluxes = []
    errors = []
    hdr_keys= []
    # hdr_keys = ['JD','refspec1']
    hdr_vals = [[] for _ in range(len(hdr_keys))]

    for i, subset_range in enumerate(subset_ranges):
        if verbose:
            print('On Chunk '+str(i+1)+'/'+str(len(subset_ranges)))

        fluxSubset = []
        errorSubset = []

        for j,data_file in enumerate(subset_range):
            filename = _proc + data_file + suffix
            with fits.open(filename) as hdul:
                hdu = hdul[0]
                data = hdu.data

                fluxSubset.append(data[0])
                errorSubset.append(data[3])
                if i ==0 and j ==0:
                    waves = data[4]/10000

                hdr = hdu.header
                for i,key in enumerate(hdr_keys):
                    hdr_vals[i].append(hdr[key])


            del data
            del hdr
            del hdu.data
            del hdu
            del hdul

        fluxes.append(np.copy(fluxSubset))
        errors.append(np.copy(errorSubset))

        del fluxSubset
        del errorSubset

    fluxes = np.concatenate(fluxes).transpose(1,0,2)
    errors = np.concatenate(errors).transpose(1,0,2)

    hdr_dict = dict(zip(hdr_keys,hdr_vals))

    if verbose:
        print('Writing Data')

    for order in range(len(fluxes)):
        if verbose:
            print('Writing order ' + str(order+1) +'/'+ str(len(fluxes)))
        toPickle = {
            'fluxes': fluxes[order],
            'errors': errors[order],
            'waves' : waves[order]
        }
        toPickle.update(hdr_dict)

        saveName = _proc+'order_'+str(order)+'.pickle'

        with open(saveName,'wb') as pf:
            pickle.dump(toPickle,pf,protocol=pickle.HIGHEST_PROTOCOL)



os.chdir(dir0)
print "\n... and we're done!"
