import os, sys
import numpy as np
from numpy import *
from scipy import interpolate
from warnings import warn
import pdb
import analysis as an
import observing as obs

"""
 IJC's attempt at writing a useful Python library for use with
 astronomical data reduction.  Developed at UC Los Angeles.

 
 2008-06-30 17:13 IJC: Made write_exptime able to recursively use file lists

 2010-10-29 09:18 IJC: Updated documentation for Sphinx.  Removed pad,
 fix_quadnoise_old.

 2011-04-08 11:51 IJC: Moved amedian() to analysis.py.

 :REQUIREMENTS:
     :doc:`analysis`

     :doc:`matplotlib`
"""
 
_home = os.path.expanduser('~')

class aperture:
    "IRAF aperture class."
    def __init__(self):
        self.filename = ''
        self.image = []
        self.nap = 0
        self.center = []
        self.low = []
        self.high = []


def intstr(num, numplaces=4):
    """A simple function to map an input number into a string padded with
    zeros (default 4).  Syntax is: out = intstr(6, numplaces=4) -->
    0006

    2008-05-27 17:12 IJC: Created"""

    formatstr = "%(#)0"+str(numplaces)+"d"

    return formatstr % {"#":int(num)}


def sfilelist(prefix, postfix, numlist, numplaces=4, delim=','):
    """2008-05-27 17:12 IJC: Create a delimited string of filenames based
    on a specified prefix, and a list of numeric values.  You can also
    specify the number of digits in the numeric part filenames;
    default is 4.  Default delimiter is a comma.

    :EXAMPLE: 
       ::
       
         files = sfilelist(prefix, postfix, numlist, numplaces=4, delim=',')

    """
    #2008-05-27 17:13 IJC:
    #2008-07-21 10:24 IJC: Updated to use list instead of start/end num.

    fnlist = ""
    for element in numlist:
        fnlist = fnlist + prefix + intstr(element, numplaces) + postfix + delim

    fnlist = fnlist[0:len(fnlist)-1]

    return fnlist

def filelist(prefix, postfix, numlist, numplaces=4):
    """2008-05-27 17:12 IJC: Create a list of filenames based on a
    specified prefix, and a list of numeric values.  You can also
    specify the number of digits in the filenames; default is 4.

    :EXAMPLE:
      ::
      
         files = filelist(prefix, postfix, numlist, numplaces=4)

    :SEE ALSO:  :func:`wfilelist`, :func:`sfilelist`, :func:`file2list`
    """
    #2008-05-27 17:13 IJC: 
    #2008-07-21 10:24 IJC: Updated to use list instead of start/end num.
    
    fnlist = []
    for element in numlist:
        fnlist = fnlist + [str(prefix + intstr(element, numplaces) + postfix)]

    return fnlist


def wfilelist(prefix, postfix, numlist, numplaces=4, tempname="wfilelist_py.tmp"):
    """2008-05-27 17:12 IJC: Create an ASCII file of a list of filenames
    based on a specified prefix, starting number, and ending number.
    You can also specify the number of digits in the filenames;
    default is 4.  

    If the file already exists, it is overwritten.

    :EXAMPLE:
       ::

           filelist(prefix, postfix, numlist, numplaces=4, tempname='wfilelist_py.tmp')

    :SEE ALSO:  :func:`filelist`, :func:`sfilelist`, :func:`file2list`
    """#
    #2008-05-27 17:13 IJC:
    #2008-07-21 10:25 IJC: Updated to use list instead of start/end num

    f = open(tempname, "w")
    for element in numlist:
        strtowrite = prefix + intstr(element, numplaces) + postfix + "\n"
        f.write(strtowrite)

    f.close()

    return tempname


def strl2f(filename, strl, clobber=True, EOL='\n'):
    """Write a list of strings to the specified filename.

    :INPUTS:
        filename:  string, name of file to write to
        strl: list, to be written to specified file.

    Returns the filename

    :Note: this is only designed for single-depth lists 
       (i.e., no doubly-deep string lists).
    """
    # 2009-04-28 10:46 IJC: Created to convert filelist to wfilelist

    import os

    if clobber==True and os.path.isfile(filename):
        os.remove(filename)
        
    f = open(filename, 'w')
    for el in strl:
        f.write(el+EOL)
    f.close()
    
    return filename
    

def showfits(filename):
    """2008-05-28 13:15 IJC: Routine to load and show a FITS file.

    showfits('blah.fits')"""

    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    
    from pylab import figure, imshow, cm

    im = pyfits.getdata(filename)
    imgsize = (im.shape)[0]
    figure()
    imshow(im, aspect='equal', cmap = cm.gray)

    return

def getval(filename, key, *ext, **kw):
    """Get a keyword's value from a header in a FITS file, or a list of
    files.

    Syntax is the same as pyfits.getval:

    @type filename: string or list

    @param filename: input FITS file name, or list of filenames

    @type key: string

    @param key: keyword name

    @param ext: The rest of the arguments are for extension specification.
       See L{getdata} for explanations/examples.

    @return: keyword value

    @rtype: string, integer, or float

    An extra keyword is path='' -- a path to prepend to the filenames.
    """
    # 2009-02-11 13:54 IJC: Created
    # 

    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    
    
    defaults = dict(path='')

    for keyword in defaults:
        if (not kw.has_key(keyword)):
            kw[keyword] = defaults[keyword]
    
    if (filename.__class__ != str) and (len(filename)>1):
        ret = []
        for file in filename:
            ret.append(pyfits.getval(kw['path']+file, key, *ext))
    else:
        ret = pyfits.getval(kw['path']+filename, key, *ext)

    return ret
    


def write_exptime(filename, coadds='coadds', itime='itime'):
    """Read 'itime' and 'coadds' from a specified file, and write into it
       an "exptime' header keyword (IRAF likes it this way).  If the
       filename does not have a '.fits' extension, write_exptime will
       attempt to add one in order to find the file.

       :EXAMPLE:
          ::

            nsdata.write_exptime('blah.fits')
            """

    # 2008-06-10 15:54 IJC: Updated to accept files with or w/out .fits extension
    # 2008-06-30 17:08 IJC: Make it able to recursively use file lists
    # 2010-01-21 13:15 IJC: Added coadds keyword
    # 2010-11-28 12:26 IJC: Added ignore_missing_end call
    # 2016-10-15 02:17 IJMC: Added test for itime
    
    from pyraf import iraf as ir

    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    
    import pdb

    ir.load('noao')
    ir.load('imred')
    ir.load('ccdred')


    if os.path.isfile(filename):
        
        # Check (crudely) to see if it's a FITS file.  Otherwise, assume it's a list.
        infile = open(filename)
        if infile.read(6)=='SIMPLE':   # It's a FITS file!
            infile.close()
            coa = pyfits.getval(filename,coadds, ignore_missing_end=True)
            exptime =  float(coa) * float(pyfits.getval(filename, itime, \
                                                    ignore_missing_end=True))
                
            ir.ccdhedit(filename, 'exptime', exptime)
        else:
            infile.seek(0, 0)
            for line in infile:
                write_exptime(line.strip(), itime=itime)

    elif os.path.isfile(filename + ".fits"):
        filename = filename + ".fits"
        coa = pyfits.getval(filename, coadds, ignore_missing_end=True)
        exptime = 1.0 * coa * pyfits.getval(filename, itime, ignore_missing_end=True)
        ir.ccdhedit(filename, 'exptime', exptime)
    

    return

def dark_correct(rawpre, procpre, sdark, postfix, startnum, endnum, numplaces=4):
    """Use similar parameters as nsdata.filelist() to correct frames for
    dark current.  Frames will have a 'd' appended to their filenames.

    :EXAMPLE:
       ::

         nsdata.dark_correct('/raw/targ_', '/proc/targ_', '/proc/sflat', '', 10, 20)

    2008-06-11 16:53 IJC: Created
    """

    from pyraf import iraf as ir

    raw  = wfilelist(rawpre, postfix, startnum, endnum, numplaces=numplaces, tempname='tempraw')
    proc = wfilelist(procpre, 'd'+postfix, startnum, endnum, numplaces=numplaces, tempname='tempproc')
    rawlist = wfilelist(rawpre, postfix, startnum, endnum, numplaces=numplaces)

    for ii in range(len(rawlist)):  # write "exptime" header keyword
        write_exptime(rawlist[ii])
        
    ir.ccdproc('@tempraw', output='@tempproc', ccdtype="", fixpix="no", overscan="no",trim="no",zerocor="no",darkcor="yes",flatcor="no", dark=sdark)

    os.remove('tempraw')
    os.remove('tempproc')

    return

def find_features(spec, width=4, absorption=True):
    """Scan through a spectrum and attempt to identify features based
       on the sign of derivatives.  Return the indices of the centers
       of the features.

       :EXAMPLE:
         ::

           ind = find_features(spec, width=4, absorption=True)

       2008-06-20 10:06 IJC: Created
       """

    from pylab import diff, sign, sum, zeros

    if width<2:
        print "Width too small, increasing to 2..."
        width = 2
    elif (int(width)/2.0)!=int(int(width)/2):
        print "Width not an even number, increasing to the next-largest even number."
        width = int(width)+1

    if (absorption):
        s=-1
    else:
        s=1

    w2 = width/2
    ds = diff(spec)
    sigds = sign(ds)

    numiter = len(spec)-width+1
    ind_temp = zeros(len(spec))
    ind_iter = 0

    for ii in range(numiter):
        # If the first w/2 signs are negative and the next w/2 positive, it's a feature.
        if (sum(s*sigds[ii:(ii+w2)]>0)==w2) and (sum(s*sigds[(ii+w2):(ii+width)]<0)==w2):
            ind_temp[ind_iter] = ii+w2
            ind_iter = ind_iter+1
    

    ind = ind_temp[0:ind_iter]
    return ind

def dispeval(c, olim, xlim, shift=0, function='chebyshev', fullxlim=None):
    """Evaluate an IRAF-generated dispersion function (for now, Chebyshev
       only).  Needs the input "C" matrix from ECIDENTIFY, as well as
       the limits of orders and pixels:
          ::

             w = nsdata.dispeval(C, [32,37], [1,1024])

       It can also be used in conjunction with :func:`nsdata.getdisp`:
          ::

             fn = 'ec/ecmar21s0165s'
             d  = nsdata.getdisp(fn)
             w  = nsdata.dispeval(d[0], d[1], d[2], shift=d[3])

       :SEE ALSO: :func:`nsdata.getdisp`, :func:`nsdata.interp_spec`

       :NOTE:
          May not be correct for multi-order (echelle) dispersion solutions!

       2008-06-24 11:48 IJC"""

    # 2008-06-26 10:05 IJC: Now returns w.tranpose() for proper
    #     arrangements.  Also fixed a bug in calculating the Q
    #     polynomial coefficients... was incorrect before.

    # 2012-12-13 15:34 IJMC: Now return precisely the same wavelength
    #      scale as displayed by IRAF; may not yet be correct for
    #      multi-order (echelle) dispersion solutions!

    from pylab import sort, arange, zeros, size, dot

    if function.lower()!='chebyshev':
        print "nsdata.dispeval only works with chebyshev functions for now"
        return -1

    olim = sort(olim[0:2])
    xlim = sort(xlim[0:2])

    o = olim[1] - arange(olim[1]-olim[0]+1)
    x = arange(xlim[1]-xlim[0]+1)+xlim[0]
    on = (2.0*o-(o.max()+o.min())) / (o.max() - o.min()) 
    xn = (2.0*x-(x.max()+x.min())) / (x.max() - x.min())
    if fullxlim is not None:
        if hasattr(fullxlim, '__iter__') and len(fullxlim)>1:
            fullx = arange(fullxlim[1]-fullxlim[0]+1) + fullxlim[0]
        else:
            fullx = arange(fullxlim) +1.5
        fullxn = (2.0*fullx-(x.max()+x.min())) / (x.max() - x.min())
        
    #pdb.set_trace()
    w = zeros([len(x), len(o)], float)

    nord = c.ndim
    np = size(c,0)
    p = zeros(np, float)
    p[0] = 1

    if nord>1:
        nq = size(c,1)
        q = zeros(nq, float)
        q[0] = 1

        for i in range(len(x)):
            p[1] = xn[i]
            for k in (arange(np-2)+2):
                p[k] = 2.0*xn[i]*p[k-1] - p[k-2]

            for j in range(len(o)):
                if nord>1:
                    q[1] = on[j]
                    for k in (arange(nq-2)+2):
                        # Debugging: print i,j,k,np, nq, size(xn), size(on)
                        q[k] = 2.0*on[j]*q[k-1] - q[k-2]
                else:
                    q = array([1])

                f = dot(dot(p, c), q)
                w[i,j] = (f + shift)/o[j]
        w = w.transpose()

    else:
        if fullxlim:
            xn = fullxn.copy()
        w = c[0] + zeros(len(xn))
        for ii in range(1, c.size):
            if ii==1:
                zi = xn
                zi1 = array([1])
            else:
                zi2 = zi1.copy()
                zi1 = zi.copy()
                zi = 2*xn*zi1 - zi2
            w += c[ii]*zi
        

    return w


def getdisp(filename, mode='echelle'):
    """Read the most recent dispersion function values from an IRAF
       dispersion database file:
         ::

           D = nsdata.getdisp('ec/ecmar21s0165s')
       
       where:
          D[0] = coefficient matrix, C_mn

          D[1] = [min_order, max_order]

          D[2] = [min_pix,   max_pix]

          D[3] = pixel shift applied to the fit (units of PIXELS!)

          D[4] = type: 1=chebyshev, 2=legendre

       It is designed for use in conjunction with NSDATA.DISPEVAL:
         ::

             fn = 'ec/ecmar21s0165s'
             d  = nsdata.getdisp(fn)
             w  = nsdata.dispeval(d[0], d[1], d[2], shift=d[3])

       :SEE ALSO: :func:`nsdata.dispeval`

       2008-06-24 14:06 IJC"""
    # 2009-11-04 11:47 IJC: Print filename of missing file
    # 2014-12-18 10:53 IJMC: Fixed xpow/opow bug.
    # 2015-01-11 17:42 IJMC: Added 'spline3' option.

    from pylab import array

    if os.path.isfile(filename):
        infile = open(filename, 'r')
    else:
        print "File %s not found!" % filename
        return -1

    raw = infile.readlines()
    infile.close()
    raw.reverse()

    ii=0
    coef = []
    while raw[ii].find('coefficients')==-1:
        if len(raw[ii].strip())>0:
            coef.append(float(raw[ii].strip()))
        ii = ii+1

    shift = 0
    jj = 0
    while (shift==0) and (jj<10):
        if raw[ii+jj].find('shift')==1:
            shift = float(  raw[ii+jj].replace('shift', '').strip()  )
        jj=jj+1
            

    func_type = int(coef[-1])  

    # Some gymnastics to get "c" matrix in the right format:
    if mode=='echelle':
        xpow = int(coef[-2])
        opow = int(coef[-3])
        dispcoef = coef[0:xpow*opow]
        dispcoef.reverse()
        xlim = [int(coef[-5]), int(coef[-6])]
        olim = [int(coef[-7]), int(coef[-8])]
        c = array(dispcoef).reshape(opow,xpow).transpose()
        ret = [c, olim, xlim, shift, func_type]

    elif mode=='spline3':
        npieces = int(coef[-2])
        x = np.arange(coef[-3], coef[-4]+1.)
        y = np.zeros(x.shape)
        s = 1.0*(x-x.min()) * npieces / (x.max() - x.min())
        j = np.floor(s)
        a = j+1-s
        b = s-j
        zs = np.vstack([a**3, 1+3*a*(1.+a*b), 1+3*b*(1.+a*b), b**3])
        for jjj in xrange(npieces):
            jind = (j==jjj)
            y[jind] = (np.array(coef[0:-4][::-1][jjj:jjj+4]).reshape(4,1) * zs[:,jind]).sum(0)
        y[-1] = y[-2] + (y[-2] - y[-3])
        ret = y
    else:
        opow = [coef[-2]]
        xlim = [coef[-3], coef[-4]]
        c = array(coef[::-1][4:])
        olim = [0, 0]
        ret = [c, olim, xlim, shift, func_type]

    return ret

def isfits(filename):
    """ Test (CRUDELY!) whether a file is FITS format or not.
        ::

          result = nsdata.isfits(filename)

        :Inputs:
           filename  -- a string representing a filename

        :Outputs:
           result -- 0 if not a FITS file; 
                     1 if the explicitly passed filename is a FITS file; 
                     2 if conditions for 1 are not met, but 
                        'filename.fits' is a FITS file.

        If 'filename' does not exist, also checks filename+'.fits'.
        """
    # 2008-07-01 11:15 IJC: Created @ UCLA

    if os.path.isfile(filename):
        existence = True
        isFITSfile = 1
    elif os.path.isfile(filename+'.fits'):
        filename = filename + '.fits'
        existence = True
        isFITSfile = 2
    else:
        existence = False
        isFITSfile = 0

    if existence:
        infile = open(filename)
        if (not infile.read(6)=='SIMPLE'):
            isFITSfile = 0
        infile.close()

    return isFITSfile


def wl_grid(w, dispersion, method='log', mkhdr=False, verbose=False):
    """Generate a linear or logarithmic wavelength map with constant dispersion.
        ::

          w_new = wl_grid(w_old, dispersion, method='log')

        :INPUTS:
           wl_old -- array; the original wavelength map.  If composed
                     of N echelle orders of length M, should be shape
                     N x M
           dispersion -- maximum desired dispersion (wavelength units
                         per pixel).  

        :OPTIONAL_INPUTS:
           method  -- str; 'log' (DEFAULT) or 'linear' 
           mkhdr   -- bool; False (DEFAULT) or True.  If True, output
                        FITS header keys for each of the N echelle orders
           verbose -- bool; False (DEFAULT) or True.

        :OUTPUTS:
           w_new   -- the new wavelength map, with constant dispersion
           hdr     -- list of dict, each containing FITS wavelength headers
           

        :EXAMPLE1:
           ::

               w = array([1, 1.1, 1.2])
               w2, h = wl_grid(w, 0.05, method='log', mkhdr=True)
           
           Which gives: w2 = CRVAL1 * CDELT1**X, where X={0,...,M-1}

        :EXAMPLE2:
           ::

               w = array([1.00, 1.10, 1.21])
               w2, h = wl_grid(w, 0.05, method='linear', mkhdr=True)
           
           Which gives: w2 = CRVAL1 + CDELT1*X, where X={0,...,M-1}

        :SEE ALSO: :func:`interp_spec`

    """
# 2008-07-02 10:05 IJC: TBD: Make it work for linear wavelength spacing.
# 2008-12-01 13:35 IJC:  Linear wavelength spacing, plus FITS headers.
    
    w = array(w).copy()
    if len(w.shape)==1:
        w = w.reshape((1, w.shape[0]))
    n_order = w.shape[0]
    if verbose:  print "n_order>>" + str(n_order)

    if method=='linear':
        n_w = int(  (  (w[:,-1] - w[:,0]) / dispersion + 1 ).max() )
        if verbose:  print "n_w>>" + str(n_w)
        grid = meshgrid(range(n_w), w[:,0])
        w_interp = grid[1] + dispersion*grid[0]

    elif method=='log':
        c = dispersion/w.max() + 1.0   # logarithmic scaling constant
        if verbose:  print "c>>" + str(c)
        n_w = int(  (  log10(w[:,-1] / w[:,0]) / log10(c)  ).max()  )   + 1
        grid = meshgrid(range(n_w), w[:,0])
        w_interp = grid[1] * c**grid[0]
    else: 
        w_interp = wl_grid(w, dispersion, method='log')
    
    if verbose:  print "w_interp.shape>>" + str(w_interp.shape)

    # Make FITS header keywords
    if mkhdr:
        headers = []
        for ii in range(n_order):
            if method=='log':
                keys = dict(CTYPE1='log', CRPIX1=1, CRVAL1=w_interp[ii,0],
                            CDELT1=c)
            elif method=='linear':
                keys = dict(CTYPE1='linear', CRPIX1=1, CRVAL1=w_interp[ii,0],
                            CDELT1=dispersion)
            headers.append(keys)
        return w_interp, headers
    else:
        return w_interp




def interp_spec(filename, w, w_interp, suffix='int', k=1, APNUM_keyword='APNUM', badval=0, clobber=True, verbose=False):
    """ Reinterpolate a spectrum from a FITS file with a given wavelength
        calibration into a given wavelength grid.
          ::

             result = nsdata.interp_spec(filename, wl, wl_new, k=1, badval=0, clobber=True)

        :Inputs:
           filename      -- a FITS file, a list of FITS files, or a file list
                              ('.fits' optional)

           wavelengths   -- a wavelength map of the FITS files to be reinterpolated

           wl_new        -- the new wavelength map (preferably from nsdata.wl_grid)

        :Output:
           result        -- 0 if something went wrong; 1 otherwise.

        :Example:
           d_cor  = nsdata.getdisp('ec/ecscifile')

           w_old  = nsdata.dispeval(d[0], d[1], d[2], shift=d[3])

           w_new  = nsdata.wl_grid(w_old, 0.075)

           result = nsdata.interp_spec('mar21s0161s', w_old, w_new)

        :SEE ALSO: :func:`getdisp`, :func:`dispeval`, :func:`wl_grid`

    """
    # 2008-07-01 11:27 IJC: Written @ UCLA. 
    # 2008-07-22 10:14 IJC: Made it work for list input
    # 2009-07-10 10:55 IJC: Set pyfits.writeto's 'output_verify' to 'ignore'
    # 2014-12-18 10:55 IJMC: Updated PyFits header-update syntax.

    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    
    from pylab import find

    badResult  = 0
    goodResult = 1

    # ---- Check File ---
    if filename.__class__==list:
        returnlist = []
        for element in filename: 
            returnlist.append( interp_spec(element, w, w_interp, suffix=suffix, 
                                           badval=badval, clobber=clobber) )
        return returnlist

    if (not os.path.isfile(filename)) and (not os.path.isfile(filename+'.fits')):
        return badResult

    if (not isfits(filename)):   # assume it's a file list
        infile = open(filename)
        for line in infile:
            thisfile = line.strip()
            if os.path.isfile(thisfile):
                interp_spec(thisfile, w, dispersion, suffix=suffix, badval=badval, clobber=clobber)
            elif os.path.isfile(thisfile+'.fits'):
                interp_spec(thisfile+'.fits', w, dispersion, suffix=suffix, badval=badval, clobber=clobber)
            
    else:   # must be a FITS file!
        
        if isfits(filename)==2:
            filename = filename + '.fits'

        # ---- Get data; check it ---
        irafspectrum = pyfits.getdata(  filename)
        irafheader   = pyfits.getheader(filename)

        spec_dim = rank(irafspectrum)
        if spec_dim<3:
            n_bands = 1
            if spec_dim<2:
                n_ap = 1
            else:
                n_ap = size(irafspectrum, 0)
        else:
            n_bands = size(irafspectrum, 0)
            n_ap    = size(irafspectrum, 1)
        if verbose:
            print "n_bands>>" + str(n_bands)
            print "n_ap>>" + str(n_ap)
            print "w_interp.shape>>" + str(w_interp.shape)

        # Select which apnums to keep
        if (n_ap != len(w_interp)):
            # There are gaps in orders, must select correct ones
            order_numbers  = []
            for key,value in irafheader.items():
              if APNUM_keyword in key:
                  this_order_num = int(value.split(' ')[0]) - 1
                  order_numbers.append(this_order_num)

            if verbose:
              print "Pruning orders to be (1-indexed): "+str( [o+1 for o in order_numbers] )

            w_interp = w_interp[order_numbers]
            w        = w[order_numbers]

        # Interpolate
        n_w = w.size/len(w)
        s_interp = zeros([n_bands+1, n_ap, size(w_interp,1)])
        s_interp[n_bands,:,:] = w_interp
        if verbose:
            print "irafspectrum.shape>>" + str(irafspectrum.shape)
            print "s_interp.shape>>" + str(s_interp.shape)
        # ---- Reshape data; interpolate it ---
        irafspectrum = irafspectrum.reshape(n_bands, n_ap, size(irafspectrum)/n_bands/n_ap)

        for i_band in range(n_bands):
            for i_ap in range(n_ap):
                spec = irafspectrum[i_band,i_ap,:]
                spline = interpolate.UnivariateSpline(w[i_ap,:], spec, s=0.0, k=k)
                s_interp[i_band,i_ap,:] = spline(w_interp[i_ap,:])
                
                out_of_range = find(w_interp[i_ap,:]>w[i_ap,:].max())
                s_interp[i_band,i_ap,out_of_range] = badval
                
        # ---  Write data to new FITS file. ---
        irafheader['BANDID'+str(n_bands+1)] = 'lambda - reinterpolated wavelengths (IJC)'
        irafheader['WLINTERP'] =  'Reinterpolated! (IJC)'
        irafheader['BADVAL'] = badval
        pyfits.writeto(filename.replace('.fits','')+suffix+'.fits', s_interp, header=irafheader, overwrite=clobber, output_verify='ignore')


    return goodResult

def wspectext(inputspec, wlsort=False):
    """Write a FITS-file spectrum to a column-format ASCII file.
       ::

           nsdata.wspectext('filelist')   ## Not yet working
           nsdata.wspectext('target.fits')
           nsdata.wspectext(specdata)     ## Not yet working

       :Inputs:

         If 3D: The first band is the data; the next-to-last is the
                 error; the last is the wavelength

       Outputs:  TBW

       2008-07-07 10:44 IJC
    """
    # 2008-07-07 10:44 IJC: Created for use with XTELLCOR; still work to do.
    # 2008-07-18 14:23 IJC: Works for FITS files; 
    # 2016-10-16 03:38 IJMC: Added wlsort option

    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    

    if inputspec.__class__==str:  # it's a filename or file list
        filetype = isfits(inputspec)
        filename = inputspec + '.fits'*(filetype-1)
        if filetype==0:   # Filelist: recursively call each line of list
            if os.path.isfile(filename):
                infile = open(filename)
                for line in infile:
                    thisfile = line.strip()
                    wspectext(thisfile)
            else:
                warn('Filelist not found!')

        elif (filetype==1) or (filetype==2):  # FITS File
            # Initialize things
            filename_out = filename.replace('.fits','')+'.dat'

            rawspec = pyfits.getdata(  filename)
            rawhead = pyfits.getheader(filename)

            naxis = rawhead['NAXIS']
            dims  = []
            for ii in range(naxis):
                dims.append(rawhead['NAXIS'+str(ii+1)])

            if dims.__len__==2:
                rawspec.reshape(1, dims[1], dims[0])
            elif dims.__len__==1:
                rawspec.reshape(1, 1, dims[0])

            # Write a file.  The first band is the data; the
            #  next-to-last is the error; the last is the wavelength

            file = open(filename_out, 'w')
            for i_ap in range(dims[1]):
                for i_line in range(dims[0]):
                    outstr = ('%.8e' % rawspec[-1, i_ap, i_line] + '  ' +
                              '%.8e' % rawspec[1,  i_ap, i_line] + '  ' +
                              '%.8e' % rawspec[-2, i_ap, i_line] + '\n')
                    file.write(outstr)

            file.close()
            if wlsort:
                #pdb.set_trace()
                dat = np.loadtxt(filename_out)
                argind = np.argsort(dat[:,0])
                file = open(filename_out, 'w')                
                for ii in argind:
                    file.write('%.8e  %.8e  %.8e\n' % tuple(dat[ii]))
                file.close()
            

        else:
            warn("FAIL: Unexpected error")
        

    else:   # input is a spectrum
        warn("FAIL: Raw spectrum input not yet working.")

    return


def getirafap(filename, filelist=False, verbose=False):
    """ Get various parameters from an IRAF aperture file, or a list thereof
          ::

            ap = getirafap(filename, filelist=False)
            ap = getirafap(listname, filelist=True )

    :Input: filename / listname -- a string
           filelistlist        -- set to True if input is a list of files

    :Output: ap -- IRAF 'aperture' object (or a list of these objects)
    """
    # 2008-07-18 16:33 IJC: Created for files & lists

    # Subfunctions:
    def apapply(ap, filestr, keyword):
                    
        if filestr.find(keyword)!=-1:
            filestr = filestr.replace(keyword, '').strip()
            exec( 'ap.' + keyword + '.append(map(float, filestr.split()))' )
        return ap 

    # Initialize & check inputs
    ap = aperture()
    
    keywords = ['center\t', 'low\t', 'high\t']

    
    if filelist:
        infile = open(filename)
        ap = []
        for line in infile:
            if verbose:  print line.strip()
            ap.append(getirafap(line.strip()))
        infile.close()

    elif filename.__class__==list:
        ap = []
        for file in filename:
            if verbose:  print file
            ap.append(getirafap(file))

    else:
        # Read file
        if (not os.path.isfile(str(filename))):
            warn('File ' + str(filename) + ' not found!')
            return []
        ap.filename = filename
        infile = open(filename)
        raw = infile.readlines()
        infile.close()

    # Parse values for each keyword
        for line in raw:
            temp = line.strip()
            for element in keywords:
                apapply(ap, temp, element)

    return ap

def collapse_objlist(object_list, keyword, suffix='', doarray = False):
    """ Given a LIST of identical objects, collapse the objects to form a
        list of only a single keyword.
         ::

           aplist = nsdata.getirafap('ap1')
           cenlist = nsdata.collapse_objlist(aplist, 'keyword')

        If you want to get fancy, you can add a suffix to the keyword
        (e.g., to subscript):
         ::

           newlist = nsdata.collapse_objlist(object_list, 'keyword', suffix='[0]')
        """
    # 2008-07-21 10:47 IJC: Created.
    
    # allaps  = ns.getirafap(aperture, filelist=True)
    # temp    = ns.collapse_objlist(allaps, 'center', suffix='[:,1]', doarray=True)
    # centers = array(temp).ravel().reshape(len(temp), 6)
    # py.plot(x, centers - py.mean(centers))

    if type(object_list)!=list:
        warn('Input object not a list!')
        return []

    object_keywords = dir(object_list)
    found_keyword = False
    for val in dir(object_list[0]):
        found_keyword = (found_keyword or (val==keyword))
    if (not found_keyword):
        warn('Specified keyword "' + keyword + '" not found!')
        return []


    collapsed_list = []
    mainclass = object_list[0].__class__
    for element in object_list:
        if element.__class__!=mainclass:
            warn('Input list is not homogeneous!')
            return []
        elif doarray:
            collapsed_list.append(eval('array(element.' + keyword + ')' + suffix) )
        else:
            collapsed_list.append(eval('element.' + keyword + suffix))

    return collapsed_list
        


def file2list(filelist, prefix='', suffix=''):
    """Convert an IRAF-like data file list of images to a Python list.
       ::

         newlist = nsdata.file2list('filelist', prefix='', suffix='')

       A prefix or suffix can also be appended to each filename.  This
       is useful if, e.g., you need an explicit file extension identifier.


       :SEE ALSO:  :func:`wfilelist`, :func:`filelist`, :func:`sfilelist`
       """

    #2008-07-25 16:30 IJC: Created

    newlist = []
    
    if filelist.__class__!=str: 
        warn('Input to file2list must be a string identifying a filename.')
    elif (not os.path.isfile(filelist)):
        warn('Input file "' + filelist + '" not found!')

    else:   # Everything is good.
        infile = open(filelist)

        for line in infile:
            filename = prefix + line.strip() + suffix
            if len(filename)>0:
                newlist.append(filename)

        infile.close()

    return newlist


def imshow(data, x=[], y=[], aspect='auto', interpolation='nearest', cmap=None, vmin=[], vmax=[]):
    """ Version of pylab's IMSHOW with my own defaults:
    ::

      imshow(data, aspect='auto', interpolation='nearest', cmap=cm.gray, vmin=[], vmax=[])

    Other IMSHOW options are default, but a new one exists: 
          x=  and y=  let you set the axes values by passing in the x and y coordinates."""
    #2008-07-25 18:30 IJC: Created to save a little bit of time and do axes.

    from pylab import arange, cm, imshow

    if cmap==None:
        cmap = cm.gray

    def getextent(data, x, y):
        """ Gets the extent of the data for plotting.  Subfunc of IMSHOW."""
        dsh = data.shape

        if len(x)==0:
            x = arange(dsh[1])
        if len(y)==0:
            y = arange(dsh[0])

        dx = 1.0* (x.max() - x.min()) / (len(x) - 1)
        xextent = [x.min() - dx/2.0, x.max() + dx/2.0]
        xextent = [x[0] - dx/2.0, x[-1] + dx/2.0]

        dy = 1.0* (y.max() - y.min()) / (len(y) - 1)
        yextent = [y.max() + dy/2.0, y.min() - dy/2.0]
        yextent = [y[-1] + dy/2.0, y[0] - dy/2.0]

        extent = xextent + yextent
        
        return extent

    def getclim(data, vmin, vmax):
        if vmin.__class__==list:
            vmin = data.min()
        if vmax.__class__==list:
            vmax = data.max()
        return [vmin, vmax]
    
    #------------- Start the actual routine -------------

    extent = getextent(data, x,y)
    clim   = getclim(data, vmin, vmax)
    imshow(data, aspect=aspect, interpolation=interpolation, cmap=cmap, 
              vmin=clim[0], vmax=clim[1], extent=extent)
    

def subdata(data, op='median', axis=None, returndata=False):
    """Take the mean/median along a specified direction and subtract it
       from the rest of the data.

       :EXAMPLE:
         ::

          p   = [[1,2,3], [4,5,7]]
          q1 =  nsdata.subdata(p, op='mean',   axis=0)
          q2 =  nsdata.subdata(p, op='median', axis=1)

       :Gives:
             q1:   [[-1.5, -1.5, -2], [1.5, 1.5, 2]]

             q2:   [[  -1,    0,  1], [ -1,   0, 2]]

       :KEYWORDS:
            *op:   operation to perform; either 'median' (DEFAULT) or 'mean'

            *axis: axis along which to perform 'op'; if None (DEFAULT),
                   'op' is performed on the entire data set as a whole.

            *returndata: Whether to also return the data series by which the
                          division was performed.  DEFAULT is False.

        :REQUIREMENTS:
            :doc:`analysis`

        :SEE ALSO:  :func:`divdata`
     """

    # 2008-07-25 19:42 IJC: Created, and proud of it.
    import analysis as an

    data = array(data)
    dsh = list(data.shape)

    if op=='median':
        chunk = an.amedian(data, axis=axis)
    elif op=='mean':
        chunk = mean(data, axis=axis)

    if axis!=None:
        dsh[axis] = 1
        chunk = chunk.reshape(dsh)

    newdata = data - chunk

    if returndata:
        return (newdata, chunk)
    else:
        return newdata



    
def divdata(data, op='median', axis=None, badval=nan, returndata=False):
    """Take the mean/median along a specified direction and divide
       the rest of the data by it.

    :EXAMPLE:
      ::

        p   = [[1,2,3], [4,5,7]]
        q1 =  nsdata.divdata(p, op='mean',   axis=0)
        q2 =  nsdata.divdata(p, op='median', axis=1)

    :Gives:
          q1:   [[-1.5, -1.5, -2], [1.5, 1.5, 2]]

          q2:   [[  -1,    0,  1], [ -1,   0, 2]]

    :KEYWORDS:
        *op:         operation to perform; either 'median' (DEFAULT),
                       'mean', or 'none' (i.e., divide by 1)

        *axis:       axis along which to perform 'op'; if None (DEFAULT),
                      'op' is performed on the entire data set as a whole.

        *badval:     value to replace any residual nan/inf values with.
                       DEFAULT is nan.  Makes two passes, pre- and
                       post-division.

        *returndata: Whether to also return the data series by which the
                      division was performed.  DEFAULT is False.

    :REQUIREMENTS:
        :doc:`analysis`

    :SEE ALSO:  :func:`subdata`
     """
    
    # 2008-07-25 19:42 IJC: Created, and proud of it.
    # 2009-10-19 22:18 IJC: There's always room for improvement.  Uses
    #                       fixval now.

    from pylab import find
    from analysis import fixval, amedian

    data = array(data)
    dsh = list(data.shape)
    nsh = list(dsh)
    nsh[axis] = 1

    #print dsh

    fixval(data, badval)

    if op=='median':
        chunk = amedian(data, axis=axis)
    elif op=='mean':
        chunk = mean(data, axis=axis)
    elif op=='none':
        chunk = ones(nsh,float)

    if axis!=None:
        chunk = chunk.reshape(nsh)

    newdata = 1.0 * data / chunk

    fixval(newdata, badval)

    if returndata:
        return (newdata, chunk)
    else:
        return newdata

    
def repval(data, badval, newval):
    """ Replace all occurrences of one value with another value.  This
        handles nan and inf as well.

        :EXAMPLE:             To set all 'nan' values to zero, just type:
          ::
 
            C = repval(data, nan, 0)

        """
    #2008-07-29 10:25 IJC: Created
    
    from pylab import find

    data = array(data)
    dsh = data.shape
    data = data.ravel()
    
    if isfinite(badval):
        ind = find(data==badval)
    elif isnan(badval):
        ind = find(isnan(data))
    elif isinf(badval):
        ind = find(isinf(data))
    else:
        warn('Value to replace is neither finite, nan, nor inf... error!!')
        return array([])

    data[ind] = newval
    data = data.reshape(dsh)

    return data


def mederr(data, ntrials=1000, mode='median'):
    """ Return the median or mode, and the 68.3% error on that
    quantity, using bootstrapping."""
    # 2008-07-29 16:54 IJC: Created
    # 2009-08-27 12:43 IJC: Added 'mode="mean"' option.
    data = array(data.ravel())

    if mode=='median':
        medianval = median(data)
    elif mode=="mean":
        medianval = mean(data)

    nd    = len(data)
    sigcutoff = [int(0.1586*ntrials+0.5), int(0.8414*ntrials+0.5)]  # contains 68.27% of the data -- one standard deviation.

    meds = zeros(ntrials)
    for ii in range(ntrials):
        ind = (random.rand(nd)*nd - 0.5).round()
        newdata = data[ind.tolist()]
        if mode=='median':
            meds[ii] = median(newdata)
        elif mode=='mean':
            meds[ii] = mean(newdata)

    meds.sort()
    lowsig = meds[sigcutoff[0]]
    hisig  = meds[sigcutoff[1]]

    medianstd = (lowsig - hisig)/2.0

    return (medianval, medianstd)



def quadd(arg1, arg2):
    """ Add two arguments in quadrature.
      ::

        print quadd(0.1, 0.2)       -->  0.2236
        print sqrt(0.1**2 + 0.2**2) -->  0.2236
    """
    # 2008-07-30 19:53 IJC: Created
    arg1 = array(arg1, subok=True, copy=True)
    arg2 = array(arg2, subok=True, copy=True)

    return sqrt(arg1**2 + arg2**2)

def gd2jd(datestr):
    """ Convert a string Gregorian date into a Julian date using Pylab.
        If no time is given (i.e., only a date), then noon is assumed.
        Timezones can be given, but UTC is assumed otherwise.

       :EXAMPLES:
          ::

            print gd2jd('Aug 11 2007')   #---------------> 2454324.5
            print gd2jd('Aug 11 2007, 12:00 PST')  #-----> 2454324.29167
            print gd2jd('12:00 PM, January 1, 2000')  #--> 2451545.0

       :REQUIREMENTS: :doc:`matplotlib`

       :SEE ALSO: :func:`jd2gd`
       """
# 2008-08-26 14:03 IJC: Created        
# 2010-12-08 13:00 IJC: Removed "+ 3442850" from num2julian call
# 2011-05-19 11:37 IJMC: Put the factor back in for error-catching...
# 2014-10-09 12:38 IJMC: Updated string-checking.    
    import matplotlib.dates as dates
    
    try:
        junk = datestr + 'hi'
        isstr = True
    except:
        isstr = False

    if isstr:
        d = dates.datestr2num(datestr)
        jd = dates.num2julian(d) 
        if jd<0:
            jd = dates.num2julian(d + 3442850)
            print "You are probably using an old version of Matplotlib..."
    else:
        jd = []

    return jd

def jd2gd(juldat):
    """ Convert a numerial Julian date into a Gregorian date using Pylab.
        Timezone returned will be UTC.

       :EXAMPLES:
         ::

          print jd2gd(2454324.5)  #--> 2007-08-12 00:00:00
          print jd2gd(2451545)    #--> 2000-01-01 12:00:00

       :SEE ALSO: :func:`gd2jd`"""
    # 2008-08-26 14:03 IJC: Created    
    # 2011-01-22 16:24 IJC: Removed arbitrary (?) subtraction of 3442850 from 'd'
    # 2011-10-21 14:11 IJMC: Put it back in, but with MPL version-checking.
    import matplotlib.dates as dates
    from matplotlib import __version__

    if __version__ < '1.0.0':
        print "You are probably using an old version of Matplotlib..."
        d = dates.julian2num(juldat - 3442850)
    else:
        d = dates.julian2num(juldat)
    gd = dates.num2date(d )

    return gd


def fix_quadnoise(*args, **kw):
    """Fix the 8-row coherent patterns in each quadrant of NIRSPEC using
       linear least-squares after removing outliers.

    :INPUTS:
       file -- a filename or list of FITS files.  The file suffix
             '.fits' is appended if the file cannot be found.  If this
             parameter begins with the '@' ('at') symbol, it is
             interpreted as an IRAF file list.

    :OPTIONAL_INPUTS:
       prefix  -- prefix to add to the fixed files.  

       clobber -- overwrite existing files

       verbose -- boolean flag for more output printed to the screen

    :OUTPUTS:
       none

    :EXAMPLE:
       ::

       fix_quadnoise(file, verbose=False, clobber=True)
       fix_quadnoise('mar21s0165.fits')
    """
    # 2009-11-03 09:33 IJC: Created anew; fit to background levels.
    # 2010-09-07 11:04 IJC: Substantially revised to combat occasional
    #                       1e9 pedestals
    # 2014-12-18 14:30 IJMC: Major fix: use medians not means, and so
    #                        don't mess up the data frames.

    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    
    from phot import estbg
    from numpy import zeros, arange, round, hstack, dot, prod
    from numpy.linalg import pinv
    from analysis import removeoutliers

    sigma = 7
    nrow = 8

    # -------- initialize inputs --------
    if len(args)==0:
        print "No filename given.  Exiting."
        return
    else:
        file = args[0]

    defaults = dict(prefix='qfix', verbose=False, clobber=False)

    if len(kw)==0:
        verbose = False
        clobber = False
    else:
        for key in defaults:
            if (not kw.has_key(key)):
                kw[key] = defaults[key]
    
    verbose  = bool(kw['verbose'])
    prefix   = str(kw['prefix'])
    clobber = bool(kw['clobber'])

    if file.__class__==list:
        for element in file:
            if verbose: print "Python file list, file: " + str(element)
            fix_quadnoise(element, row=row, quadrant=quadrant, verbose=verbose, prefix=prefix)
        return
    elif file[0]=='@':
        f = open(file[1::])
        for line in f:
            if verbose: print "IRAF-type file list, file: " + str(line.strip())
            fix_quadnoise(line.strip(), row=row, quadrant=quadrant, verbose=verbose, prefix=prefix)
        f.close()
        return

    if (not os.path.isfile(file)):
        file = file + '.fits'
    try:
        data = pyfits.getdata(file)
        hdr  = pyfits.getheader(file)
    except:
        print "PYFITS Could not read from file '" + file + "' -- exiting."
        return

    x,y = meshgrid(arange(data.shape[1]),arange(data.shape[0]))
    basis = zeros([data.shape[0]/2,data.shape[1]/2,8],float)
    for ii in range(0,nrow):
        basis[ii:(512+ii):nrow,:,ii] += 1.0

    for ii in range(4):
        if ii==0:
            ind = ((x<512) * (y<512))
        elif ii==1:
            ind = ((x<512) * (y>=512))
        elif ii==2:
            ind = ((x>=512) * (y<512))
        elif ii==3:
            ind = ((x>=512) * (y>=512))

        rowmeans = zeros(nrow, float)
        d = data[ind].reshape(512,512)
        for jj in range(nrow):
            goodvalues, goodind = removeoutliers(d[jj::8,:], sigma, retind=True)
            rowmeans[jj] = np.median(goodvalues)
            
        new = basis * (rowmeans.reshape(1,1,nrow) - rowmeans.mean())
        data[ind] -= new.sum(2).ravel()

        print "row-means for quad %i are>>" % ii, rowmeans
        if verbose:
            print "row-means for quad %i are>>" % ii, rowmeans
            #from pylab import *



    if verbose: print "file>>" + str(file)
    if verbose: print "os.path.split(file)[0]>>" + os.path.split(file)[0]
    initpath = os.path.split(file)[0]
    if len(initpath)==0:  
        initpath = '.'
    outfn = initpath + os.sep + prefix + os.path.split(file)[1]
    if verbose: print "Writing file..." + outfn
    pyfits.writeto(outfn, data, hdr, overwrite=clobber, output_verify='ignore')

    return




def preprocess(*args, **kw):
    """
    Basic processing of NIRSPEC data: set JD and HJD fields, fix
    bad-row 'quadnoise', clean cosmic rays, flatten and remove bad
    pixels.

    :INPUTS:
         input = 

         output = 

         Input and output must both be specified, and must both be
         different files.

    :OPTIONAL_INPUTS:
         qfix    = True
           qpref = '' -- string to preface all quad-fixed frames

         flat    = None -- flat field for iraf.ccdproc
                        -- could be either string or dict of strings
                        -- dictionary keys are interpreted as altitude angle

         dark    = None -- dark frame for iraf.ccdproc

         mask    = None -- bad pixel mask for iraf.ccdproc

         clobber = False -- overwrite file if input and output files are same

         verbose = False

         cleanec = False -- run nsdata.cleanec (a "poor-man's iraf.cosmicrays")
             cthreshold = 300 -- threshold for nsdata.cleanec

             csigma = 20 -- sigma threshold for nsdata.cleanec

             cwindow = 25 -- window size for nsdata.cleanec

         cleancr = False -- run iraf.cosmicrays
             rthreshold = 300 -- threshold for iraf.cosmicrays

             rratio = 5 -- fluxratio threshold for iraf.cosmicrays

          corquad = "" -- (str) path to corquad executable

          airmass = 'AIRMASS' -- (str) fits airmass keyword

         Any inputs set to 'None' disable that part of the processing.

    :EXAMPLE:
      ::

         ns.preprocess('@'+rawcal, '@'+proccal, qfix=, qpref=\
                        flat=, mask=, clobber=, verbose=)


    Note that although the syntax is similar to Pyraf tasks, only
    Python Booleans or their equivalent should be used for flags
    (i.e., use True or False instead of 'yes' or 'no'
    """
    # 2008-11-26 19:23 IJC: Created on an Amtrak train
    # 2009-11-03 11:15 IJC: Updated to use new LLS fix_quadnoise
    # 2010-08-27 13:22 IJC: Added 'fluxratio' parameter to cosmicrays
    # 2010-08-28 07:28 IJC: Added 'cleanec' function as well
    # 2010-09-06 14:48 IJC: Better integrated cleanec, and added an option thereto
    # 2010-09-08 08:56 IJC: Re-added iraf.cosmicrays, with an option
    # 2016-10-15 02:49 IJMC: Now date='date-obs' is optional
    # 2016-10-15 19:22 IJMC: Now try bfixpix if ccdproc doesn't work
    
    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    
    from pyraf import iraf as ir
    ir.prcacheOff()

    import multiprocessing as mp
    from functools import partial
    from tqdm import tqdm
    import io

    ir.load('crutil')
    # Parse inputs:
    if len(args)<2:
        print "No output images specified!  Exiting..."
        return
    else:
        input = args[0]
        output = args[1]
        if input==output:
            print "Input and output files must not be the same! Exiting..."
            return

    defaults = dict(qfix=False, qpref='', flat=None, dark=None, mask=None, \
                        cleanec=False, clobber=False, verbose=False, \
                        cthreshold=300, cwindow=25, csigma=20, \
                        cleancr=False, rthreshold=300, rratio=5, \
                    date='date-obs', time='UTC', dofix=True, corquad="", airmass='AIRMASS', num_processors=1,
                    pytifts_outverify = 'warn', saveBadMask=True,
                    tryIRccdproc=True, badPixMethod='Median', flatInterpolationNumber=3)

    for key in defaults:
        if (not kw.has_key(key)):
            kw[key] = defaults[key]

    verbose = bool(kw['verbose'])
    doflat = kw['flat']!=None
    doDark = kw['dark']!=None
    dobfix = kw['mask']!=None
    clobber = bool(kw['clobber'])
    date = kw['date']
    time = kw['time']
    num_processors = kw['num_processors']
    pytifts_outverify = kw['pytifts_outverify']
    saveBadMask     = kw['saveBadMask']
    tryIRccdproc    = kw['tryIRccdproc']
    badPixMethod      = kw['badPixMethod']
    flatInterpolationNumber = kw['flatInterpolationNumber']

    if verbose:
        print '-------------------------------'
        print '\n\n'
        print 'Starting Preprocess'
        print '\n\n'
        print '-------------------------------'

    if doflat:
        if(type(kw['flat']) == dict):
            for angle in kw['flat'].keys():
                if type(kw['flat'][angle]) != str and type(kw['flat'][angle])!= unicode:
                    raise IOError("Flat kw input must be either a string or a dictionary of strings")
                else:
                    if not os.path.isfile(kw['flat'][angle]):
                        kw['flat'][angle] = kw['flat'][angle]+'.fits'
                        if not os.path.isfile(kw['flat'][angle]):
                            raise IOError("File "+ kw['flat'][angle] + " not found")
        else:
            if not os.path.isfile(kw['flat']):
                kw['flat'] = kw['flat'] + '.fits'
                if not os.path.isfile(kw['flat']):
                    raise IOError("File " + kw['flat'] + " not found")
    if doDark:
        if not os.path.isfile(kw['dark']):
            kw['dark'] = kw['dark'] + '.fits'
            if not os.path.isfile(kw['dark']):
                raise IOError("File " + kw['dark'] + " not found")

    if verbose:
        print "Keywords are:  "; print kw

    # Check whether inputs are lists, filelists, or files:

    if input.__class__!=output.__class__:
        print "Files or file lists must be of same type.  Exiting..."
        return
    elif input.__class__==list:
        if verbose:
            print '\n\nFile List - ' +input
            print '\n\n'

        pool = mp.Pool(processes = num_processors)
        num_to_preprocess = len(inLines)

        mp_kws = kw.copy()
        mp_kws['verbose'] = False
        mp_kws['pytifts_outverify'] = 'silentfix'

        pbar = tqdm(total=num_to_preprocess)
        for i,_ in tqdm(enumerate(pool.imap_unordered(
                            partial(preprocessEach,
                                    inLines  = inLines,
                                    outLines = outLines,
                                    kw       = mp_kws),
                            xrange(num_to_preprocess))
                        )):
            pbar.update()
        pbar.close()
        return

    elif (input.__class__==str) and (input[0]=='@'):
        fin  = open(input[ 1::])
        fout = open(output[1::])

        inLines  = fin.readlines()
        outLines = fout.readlines()

        ## Don't flatten, bad-pixel correct, or cleanec the first time through:
        if verbose:
            print "\n\nIraf-style file list- "+input
            print "\n\n"

        pool = mp.Pool(processes = num_processors)
        num_to_preprocess = len(inLines)

        mp_kws = kw.copy()
        mp_kws['verbose'] = False
        mp_kws['pytifts_outverify'] = 'silentfix'


        pbar = tqdm(total=num_to_preprocess)
        for i,_ in tqdm(enumerate(pool.imap_unordered(
                            partial(preprocessEach,
                                    inLines  = inLines,
                                    outLines = outLines,
                                    kw       = mp_kws),
                            xrange(num_to_preprocess))
                        )):
            pbar.update()
        pbar.close()
        fin.close()
        fout.close()
        return
 
    # Begin processing tasks
    if kw['clobber'] and input!=output:
        ir.imdelete(output)

    ir.imcopy(input, output, verbose=verbose)

    # Deal with possible invalid FITS header keys:
    if not os.path.isfile(output):
        outputfn = output + '.fits'
    else:
        outputfn = output + ''
    hdulist = pyfits.open(outputfn)
    hkeys = hdulist[0].header.keys()
    for key in hkeys:
        if key.find('.')>-1:
            newkey = key.replace('.','_')
            hdulist[0].header[newkey] = hdulist[0].header[key]
            hdulist[0].header.remove(key)
    hdulist.writeto(outputfn, overwrite=True, output_verify=pytifts_outverify)

    # I don't like IRAF.setjd because it gave me incorrect values.
    # Instead, use my own setjd from astrolib.py:
    setjd(output, date=date, time=time, jd='JD', hjd='HJD')
    if kw['qfix']:
        if 'aries' in str(kw['qfix']).lower():
            correct_aries_crosstalk(output, clobber=clobber,corquad=kw['corquad'],verbose=verbose)
        else:
            fix_quadnoise(output, prefix=kw['qpref'],clobber=clobber)
            ir.hedit(output, 'quadnois', \
                     'NIRSPEC bad row fixed by nsdata.fix_quadnoise', add=True, update='yes')

    if doflat or doDark:
        if doflat and type(kw['flat']) == dict:
            airmass = hdulist[0].header[kw['airmass']]
            altitude = convertAirmassToAltitude(airmass)
            gen_hdr, gen_data = interpolateFlatFrameFromAngle(kw['flat'],altitude,flatInterpolationNumber)

            fileID = output.split('/')[-1].split('_')[1][:4]
            gen_flat = os.path.split(output)[0]+'/generated_flat_' + fileID + '.fits'

            pyfits.writeto(gen_flat,gen_data,gen_hdr,output_verify=pytifts_outverify, overwrite='True')


            ir.ccdproc(output, ccdtype="", fixpix="no", overscan="no",
                    trim="no", zerocor="no",
                    flatcor=doflat,   flat=gen_flat,
                    darkcor = doDark, dark=kw['dark'],
                    fixfile=None, minreplace=0.25, interactive="no",
                    Stderr='preproc.log')

            ir.imdelete(gen_flat)

        else:
            ir.ccdproc(output, ccdtype="", fixpix="no", overscan="no",
                    trim="no", zerocor="no",
                    flatcor=doflat,   flat=kw['flat'],
                    darkcor = doDark, dark=kw['dark'],
                    fixfile=None, minreplace=0.25, interactive="no")
    if dobfix: 
        # Make an extra bad-pixel mask from any _negative_ values, and
        # combine it with the global bad-pixel mask; necessary because
        # some negative-valued pixels manage to make it through the
        # cleaning pipeline.
        indiv_mask = output + 'imask.fits'

        try:
            output_temp = pyfits.getdata(output)
            ohdr = pyfits.getheader(output)
            ofn = output + ''
        except:
            output_temp = pyfits.getdata(output + '.fits')
            ohdr = pyfits.getheader(output + '.fits')
            ofn = output + '.fits'

        # Inline version of cutoffmask()
        cutoff = [0, Inf]
        cMask = (output_temp < cutoff[0]) + (output_temp > cutoff[1])

        indiv_mask_data = np.logical_or(pyfits.getdata(kw['mask']), cMask).astype(int)

        if kw['dofix']:
            if tryIRccdproc:
                pyfits.writeto(indiv_mask, indiv_mask_data, overwrite=True, output_verify=pytifts_outverify)
                if not verbose:
                    save_stderr = sys.stderr
                    sys.stderr = io.BytesIO()
                try:
                    ir.ccdproc(output, ccdtype="", fixpix=dobfix, overscan="no",
                               trim="no", zerocor="no", darkcor="no", flatcor="no", 
                               flat=None, fixfile=indiv_mask,
                               minreplace=0.25, interactive="no")
                except:
                    output_temp = bfixpix(output_temp, indiv_mask_data, n=8,retdat=True,method=badPixMethod)
                    pyfits.writeto(ofn, output_temp, header=ohdr, output_verify=pytifts_outverify, overwrite=True)
                    if verbose:
                        print "Couldn't CCDPROC, but managed to BFIXPIX instead."
                if not verbose:
                    sys.stderr = save_stderr
                if not saveBadMask:
                    ir.delete(indiv_mask)
            else:
                output_temp = bfixpix(output_temp, indiv_mask_data, n=8,retdat=True,method=badPixMethod)
                pyfits.writeto(ofn, output_temp, header=ohdr, output_verify=pytifts_outverify, overwrite=True)
                if saveBadMask:
                    pyfits.writeto(indiv_mask, indiv_mask_data, overwrite=True, output_verify=pytifts_outverify)


            
    if kw['cleancr']:
        ir.cosmicrays(output, output, threshold=kw['rthreshold'], fluxratio=kw['rratio'], \
                          npasses=5, interactive='no')
    if kw['cleanec']:
        cleanec(output, output, npasses=1, verbose=verbose, threshold=kw['cthreshold'], nsigma=kw['csigma'], window=kw['cwindow'], clobber=True)

    if verbose:
        print "Successfully processed '" + input + "' into '" + output + "'\n\n"

    return

def preprocessEach(i,inLines,outLines,kw):
    preprocess(inLines[i].strip(), outLines[i].strip(), **kw)

def interpolateFlatFrameFromAngle(allflats, altitude, numToInterpolate=3):
    """ Generates flat field frame for a given altitude from dict of
        {altitude: flat_file}

        :INPUTS:
            allflats --- (dict) dictionary of {altitude: flat_file}
                     --- altitude- altitude angle for respective flat
                     --- flat_file- path to flat_file.fits
                     --- limited to integer angles

            altitude --- (float) altitude angle to generate flat field for

            numToInterpolate ---- How many nearby flatframes to use for interpolation

        :OUTPUTS:
            outhdr  --- (dict) header for generated flat frame
            outdat  --- (2d array) data for generated flat frame
    """

    from bisect import bisect
    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits

    angles = [int(k) for k in allflats.keys()]
    angles.sort()

    dists = [np.abs(angle-altitude) for angle in angles]
    angles_to_use = np.argsort(dists)[:numToInterpolate]

    flat_data    = []
    flat_weights = []

    for each in angles_to_use:
        #verify file is real
        flat_filename = allflats[str(angles[each])]
        if not os.path.isfile(flat_filename):
            flat_filename = flat_filename+'.fits'
            if not os.path.isfile(flat_filename):
                raise IOError("File "+ flat_filename + " not found")

        fd = pyfits.getdata(flat_filename)
        flat_data.append(fd)
        flat_weights.append(1/dists[each])

    # Average, weight by inverse distance in angle space
    outdat = np.average(flat_data,0,flat_weights)

    outhdr = pyfits.getheader(flat_filename)
    outhdr['OBJECT']   = 'Interpolated Flat Frame'
    outhdr['ALTITUDE'] = str(altitude)

    return outhdr, outdat

def convertAirmassToAltitude(airmass):
    """ Converts given airmass to altitude

    :INPUTS:
        airmass -- (float) airmass of observation
    :OUTPUTS:
        altitude -- (float) telescopes altitude
                 -- altitude is angle above horizon
    """

    return 90 - (np.arccos(1/airmass)*180/np.pi)

def setjd(filename, **kw):
    """Set JD and HJD fields in a FITS file, based on the UTC date and
    time header keywords.

    :INPUTS:
       filename : str

    :OPTIONS:
       ra=None : Right Ascension in decimal degrees.  If None, use "RA" header key

       dec=None : Declination in decimal degrees.  If None, use "DEC" header key 

    :EXAMPLE:
      ::

         dict(date='date-obs', time='UTC', epoch='equinox', jd='JD', hjd='hjd', \
            verbose=False, ra=None, dec=None)


    This does not take your position on Earth into account, so it must
    be less accurate than ~0.1 sec.
    """
    # 2010-09-07 15:14 IJC: Created

    try:
        import astrolib
    except:
        try:
            import PyAstronomy.pyasl as astrolib
        except:
            raise ImportError('Astrolib/PyAstronomy not found. Install PyAstronomy from here: http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/index.html')

    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    

    defaults = dict(date='date-obs', time='UTC', epoch='equinox', jd='JD', hjd='hjd', \
                        verbose=False, ra=None, dec=None)
    for key in defaults:
        if (not kw.has_key(key)):
            kw[key] = defaults[key]
    
    verbose = kw['verbose']
    ra = kw['ra']
    dec = kw['dec']

    if filename.__class__==list:
        for ii in range(len(filename)):
            if verbose: print "File list, file:  " + filename[ii]
            setjd(filename[ii], **kw)
        return

    elif (filename.__class__==str) and (filename[0]=='@'):
        fin  = open(filename[ 1::])
        for line in fin:
            setjd(line.strip(), **kw)

    else: 
        try:
            hdulist = pyfits.open(filename)
        except IOError:
            filename += '.fits'
            hdulist = pyfits.open(filename)

        hdr = hdulist[0].header
        datetime = hdr[kw['date']]
        if kw['time'] is not None:
            datetime += ' ' + hdr[kw['time']]
        if ra is None:
            try:
                ra = hdr['ra']
            except:
                ra = 0
        if dec is None:
            try:
                dec = hdr['dec']
            except:
                dec = 0
        epoch = hdr[kw['epoch']]
        if epoch != 2000:
            print "Epoch must be 2000!  Exiting..."
            return -1

        if verbose:
            print 'datetime>>', datetime
            print 'ra, dec>>', ra, dec
        try:
            ra += 0.0
        except:
            ra = obs.hms(ra)*15
        try:
            dec += 0.0
        except:
            dec = obs.dms(dec)
            
        jd = gd2jd(datetime)
        hjd = astrolib.helio_jd(jd - 2400000., ra, dec) + 2400000.

        hdr[kw['jd']]  =  jd
        hdr[kw['hjd']] = hjd
        if verbose: print 'jd, hjd, dt>>', jd, hjd, hjd-jd

        if 'FREQ.SPE' in hdulist[0].header:
            junk = hdulist[0].header['FREQ.SPE']
            print hdulist[0].header['GAIN.SPE']
        hdulist.writeto(filename, output_verify='ignore', overwrite=True)
    return

def linespec(loc, ew, win, **kw):
    """
    Create a delta-function line spectrum based on a wavelength grid
    and a list of line locations and equivalent widths.

    :INPUTS:
       loc -- location of lines in the emission frame of reference

       ew  -- equivalent widths of lines, in units of wavelength grid.
               Positive values are emission lines.

       w_in -- wavelength grid in the emission frame, with values
              monotonically increasing (best if it is linearly spaced)

       All inputs should be lists or one-dimensional arrays of scalars

    :OPTIONAL_INPUTS:
       cont=None -- set continuum values in the emission frame;

       nearest=False  -- if True, use full pixels instead of partial

       verbose=False  -- if True, print out various messages

    :OUTPUTS:
      s  -- delta-function line spectrum, with a continuum level of zero
    
    :EXAMPLE: (NEEDS TO BE UPDATED!):
       ::

          w   = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
          loc = [2.1, 2.35, 2.62]
          ew  = [0.1, .02, .01]
          s = linespec(loc, ew, w)
          print s  #  --->  [0, 1, 0, 0.1, 0.1, 0, 0.08, 0.02]

    :NOTE:  This may give incorrect results for saturated lines.
    """
    # 2008-12-05 13:31 IJC: Created
    # 2008-12-10 13:30 IJC: Added continuum option, reworked code some.
    # 2008-12-12 12:33 IJC: Removed RV option

    from pylab import find

    # Check inputs
    loc = array(loc).copy().ravel()
    ew  = array(ew ).copy().ravel()
    win   = array(win  ).copy().ravel()

    defaults = dict(cont=None, nearest=False, verbose=False)
    for key in defaults:
        if (not kw.has_key(key)):
            kw[key] = defaults[key]
    verbose = bool(kw['verbose'])
    nearest = bool(kw['nearest'])
    contset = kw['cont']!=None

    if contset:
        cont = array(kw['cont']).copy()
        if len(cont)!=len(win):
            print "Wavelength grid and continuum must have the same length!"
            return -1
    else:
        cont = ones(win.shape)

    nlines = len(loc)
    if nlines != len(ew):
        if verbose:  print "len(loc)>>" + str(len(loc))
        if verbose:  print "len(ew)>>" + str(len(ew))
        print "Line locations and equivalent widths must have same length!"
        return -1

    #Only use lines in the proper wavelength range
    nlineinit = len(loc)
    lind = (loc>=win.min()) * (loc<=win.max())
    loc = loc[lind]
    ew  =  ew[lind]
    nlines = len(loc)

    s = cont.copy()
    d = diff(win).mean()

    if verbose:  print "s>>" + str(s)

    for ii in range(nlines):
        lineloc = loc[ii]
        lineew  = ew[ii]
        index = (win<lineloc).sum() - 1
        if nearest:
            s[index+1] = s[index]-cont[index]*lineew/d
        elif index==len(win):
            s[index] = s[index] - cont[index]*lineew/d
        else:
            s[index] = s[index] - lineew*cont[index]* \
                (win[index+1] - lineloc)/d/d
            s[index+1] = s[index+1] - lineew*cont[index+1] * \
                (lineloc - win[index])/d/d
        
        if verbose:  
            print "(lineloc, lineew)>>" + str((lineloc, lineew))
            print "(index, d)>>" + str((index,d))

    if verbose:
        print "(nlineinit, nline)>>" + str((nlineinit, nlines))
    return s
            


def readfile(fn, cols=None):
    """ Read data from an space- or tab-delimited ASCII file."""

    # 2008-12-12 11:42 IJC: created

    f = open(fn, 'r')
    raw = f.readlines()
    f.close()

    if cols==None:
        dat = array([map(float, line.split()) for line in raw])
    else:
        dat = array([map(float, line.split()[cols]) for line in raw])

    return dat
    
def initobs(date, **kw):
    """Initialize variables for Nirspec data analysis.

    :INPUT:
       date -- a string of type YYYYMMMDD (e.g., 2008mar21 or 2008jun15a)

    :OPTIONAL_INPUT:
       remote=False -- changes processed-data directory as specified
                       within.

       interp=True -- whether to load wavelength-interpolated
                      (upsampled) spectra (if True) or the raw
                      1024-pixel spectra (if False)

        _raw       -- raw data directory. Defaults to _home + remoteStr +
                      "/proj/pcsa/data/raw/" + date + "/spec/"

        _proc      -- processed data directory. Defaults to _home + remoteStr +
                      "/proj/pcsa/data/proc/" + datadir

        _model     -- model directory. Defaults to _home + remoteStr +
                      "/proj/pcsa/data/model/"

    :OUTPUT:
       a tuple containing the following values, in order:
         planet -- name of planet for use in analysis.planet()

         datalist -- list of data file numbers to analyse

         _proc  -- processed data directory

         wavefilename -- filename of the wavelength solution

         starmodelfilename -- path and filename of stellar model

         planetmodelfilename -- path and filename of planet model

         aplist -- a list of the IRAF aperture filenames (for use with
                   nsdata.getirafap)

         telluric -- the FITS file spectrum of the telluric/A0V spectrum

         n_aperture -- number of echelle apertures for this setup

         filter -- NIRSPEC filter used

         prefix -- filename prefix

         calnod -- whether calibrator stars were nodded.

         rowfix -- list of four lists; which rows to fix in each of
                   four quadrants (see FIX_QUADNOISE).
    """

    # 2009-07-09 16:58 IJC: Added HD 189733b set (2008jun15b)
    # 2009-07-31 15:17 IJC: Added multi-chop runs (e.g. 2008jun15)
    # 2009-08-05 12:17 IJC: Now multi-chop runs only return dates
    # 2010-08-15 14:57 IJC: Flagged a bad file in 2008jun15b
    # 2010-08-24 15:42 IJC: Added 2010aug16 dataset
    # 2010-09-03 23:45 IJC: Added 2010sep04 dataset
    # 2010-11-10 17:02 IJC: Added 2008jul12 A/B nodding datasets
    # 2011-08-10 18:06 IJMC: Added 2011aug05 SpeX/GJ 1214b run.
    # 2017-10-25 15:52 NFM: Moved from conditional list to db

    import analysis as an
    import json


    _aphome = os.path.expanduser('~')[1:].replace('/','_')
    defaults = dict(remote=False, interp=True, _db='obsdb.json')
    for key in defaults:
        if (not kw.has_key(key)):
            kw[key] = defaults[key]

    with open(kw['_db'],'r') as f:
      obsdb = json.load(f)

    # If date is in db, unpack db into vars
    if(date in obsdb):
        for key, val in obsdb[date].items():
            exec(key +'=val')
    # Legacy Cases
    elif date=='2008jun15':
        suffix = 'bcdefg'
        ret = [date+suf for suf in suffix]
        return ret
    elif date=='2008jul12':
        suffix = 'abcd'
        ret = [date+suf for suf in suffix]
        return ret
    elif date=='2009jul29':
        suffix = 'abcdefgh'
        ret = [date+suf for suf in suffix]
        return ret
    # date not found error
    else:
        raise KeyError('Given date, '+date+' not found')

    if len(date)==9:
        meancal = 'avgcal'
    else:
        meancal = 'avgcal'+date[9::]

    date = date[0:9]

    if kw['interp']:
        postfix = 'sinttel.fits'
    else:
        postfix = 's.fits'

    wavefilename = 'winterp.fits'

    remoteStr = "/atwork" if kw['remote'] else ""

    if '_proc' in kw:
        _proc = kw['_proc']
    else:
        _proc = _home + remoteStr + "/proj/pcsa/data/proc/" + datadir

    if '_raw' in kw:
        _raw = kw['_raw']
    else:
        _raw = _home + remoteStr + "/proj/pcsa/data/raw/" + date + "/spec/"

    if '_model' in kw:
        _model = kw['_model']
    else:
        _model = _home + remoteStr + "/proj/pcsa/data/model/"
        
    if planet=='55 Cnc e': #'55cnce':
        starmodelfilename = _model + \
            'lte5243_4.33_+0.25_55Cnc.hires.7.fits'
        planetmodelfilename = _model + \
            'lte0125-3.5.rainout.HD75732e.redist_0.50.hires.7.fits'
    elif planet=='tau Boo b': #'tauboob':
        starmodelfilename = _model + \
            'lte6309_4.30_0.28_tau_boo.hires.7.fits'
        planetmodelfilename = _model + \
            'lte0125-3.5.rainout.HD120136b.redist=0.5.hires_id.7.fits'
    elif planet=='HD 189733 b': #'hd189733b':
        starmodelfilename = _model + \
            'XXXX_Lband_189model'
        planetmodelfilename = _model + \
            'XXXX_Lband_189bmodel'
    elif planet=='HD 209458 b': #'hd209458b':
        print "WARNING: HD209458 DOES NOT HAVE CORRECT STELLAR MODEL!"
        starmodelfilename = _model + \
            'lte5243_4.33_+0.25_55Cnc.hires.7.fits'
        print "WARNING: HD209458 DOES NOT HAVE CORRECT PLANETARY MODEL!"
        planetmodelfilename = _model + \
            'lte0125-3.5.rainout.HD75732e.redist_0.50.hires.7.fits'
    elif planet=='GJ 1214 b':
        if filter=='K':
            starmodelfilename = ''
            planetmodelfilename = _model + \
                'lte0125-3.00.rainout_irrad.GJ1214b_redist=0.25.aces_rlam.fits'
            print "WARNING: GJ1214b K-band DOES NOT HAVE CORRECT STELLAR MODEL!"
            planetmodelfilename = _model + \
                'noplanetmodel'
        elif filter=='H':
            starmodelfilename = _model + 'lte3000_5.00-0.0.GJ1214_hires_Hband.7.fits'
            planetmodelfilename = _model + \
                'lte0125-3.00.rainout_irrad.GJ1214b_redist=0.25.aces_rlam.fits'
            print "WARNING: GJ1214b H-band DOES HAS A _TRANSMISSION SPECTRUM_ MODEL!"
        elif filter=='SXD':
            starmodelfilename = _model + 'NOSTELLARMODEL'
            planetmodelfilename = _model + 'lte0125-3.00.rainout_irrad.GJ1214b_redist=0.25.aces_rlam.fits'
    elif planet=='WASP-12 b':
        print "WARNING: WASP12 DOES NOT HAVE CORRECT STELLAR MODEL!"
        starmodelfilename = _model + \
            'lte5243_4.33_+0.25_55Cnc.hires.7.fits'
        print "WARNING: WASP12 DOES NOT HAVE CORRECT PLANETARY MODEL!"
        planetmodelfilename = _model + \
            'lte0125-3.5.rainout.HD75732e.redist_0.50.hires.7.fits'
    elif planet=='HAT-P-11 b':
        print "WARNING: HAT-P-11 DOES NOT HAVE CORRECT STELLAR MODEL!"
        starmodelfilename = _model + \
            'lte5243_4.33_+0.25_55Cnc.hires.7.fits'
        print "WARNING: HAT-P-11 may not HAVE CORRECT PLANETARY MODEL!"
        planetmodelfilename = _model + \
            'lte0125_0.31_rainout.HAT-P-11p.redist=0.50.sph.rlam.spec_2.fits'
    elif planet=='HD 187123 b':
        print "WARNING:  DOES NOT HAVE CORRECT STELLAR MODEL!"
        starmodelfilename = _model + \
            'lte5243_4.33_+0.25_55Cnc.hires.7.fits'
        print "WARNING:  may not HAVE CORRECT PLANETARY MODEL!"
        planetmodelfilename = _model + \
            'lte0125_0.31_rainout.HAT-P-11p.redist=0.50.sph.rlam.spec_2.fits'
    elif planet=='WASP-33 b':
        print "WARNING:  Using WASP-12 model."
        starmodelfilename = _model + \
            'te0125-rainout.WASP-12p.redist=0.50.aces.sph.hires_all.approx_emission.fits'
        print "WARNING:  Using WASP-12 model."
        planetmodelfilename = _model + \
            'lte6309_4.30_0.28_wasp12.hires_approx.fits'
    else:
        # stop
        starmodelfilename = ""
        planetmodelfilename = ""

    datalist = filelist(prefix, postfix, framelist)


    darkfilelist     = filelist(_raw+prefix, '.fits', darklist, numplaces=4)
    darkflatlist     = filelist(_raw+prefix, '.fits', darkflatlist, numplaces=4)
    darkcallist      = filelist(_raw+prefix, '.fits', darkcallist, numplaces=4)
    rawcalfilelist   = filelist( _raw+prefix, '.fits', callist, numplaces=4)
    proccalfilelist  = filelist( _proc+prefix, 'fn', callist, numplaces=4)
    rawtargfilelist  = filelist( _raw+prefix, '.fits', framelist, numplaces=4)
    proctargfilelist = filelist( _proc+prefix, 'fn', framelist, numplaces=4)

    fullproctargfilelist = filelist( _proc+prefix, 'fn', fullframelist, numplaces=4)

    speccalfilelist  = filelist(prefix, 's', callist, numplaces=4)
    spectargfilelist = filelist(prefix, 's', framelist, numplaces=4)

    # Flat Frame Handling
    if type(flatlist) == dict:
        flatfilelist = {}
        for key in flatlist:
            flatfilelist[str(key)] = filelist(_raw+prefix, '.fits', flatlist[key], numplaces=4)
    else:
        flatfilelist = filelist(_raw+prefix, '.fits', flatlist, numplaces=4)

    aplist     = filelist(_proc+ap_suffix, 'fn', framelist)
    aplist_cal = filelist(_proc+ap_suffix, 'fn', callist)
    
    telluric_fn = _proc + prefix + meancal + 'int.fits'
    disp = 0.075

    # ret = (planet, _proc, datalist, wavefilename,
    #        starmodelfilename, planetmodelfilename, aplist, telluric_fn, _raw,
    #        darkfilelist, flatfilelist, (rawcalfilelist, proccalfilelist),
    #        (rawtargfilelist, proctargfilelist), (speccalfilelist, spectargfilelist), 
    #        n_aperture, filter, prefix, calnod, meancal, disp, aplist_cal)
    ret = {}
    ret['planet']               = planet
    ret['_proc']                = _proc
    ret['datalist']             = datalist
    ret['wavefilename']         = wavefilename
    ret['starmodelfilename']    = starmodelfilename
    ret['planetmodelfilename']  = planetmodelfilename
    ret['aplist']               = aplist
    ret['ap_suffix']            = ap_suffix
    ret['telluric_fn']          = telluric_fn
    ret['_raw']                 = _raw
    ret['darkfilelist']         = darkfilelist
    ret['darkflatlist']         = darkflatlist
    ret['darkcallist']          = darkcallist
    ret['flatfilelist']         = flatfilelist
    ret['rawcalfilelist']       = rawcalfilelist
    ret['proccalfilelist']      = proccalfilelist
    ret['rawtargfilelist']      = rawtargfilelist
    ret['proctargfilelist']     = proctargfilelist
    ret['fullproctargfilelist'] = fullproctargfilelist
    ret['speccalfilelist']      = speccalfilelist
    ret['spectargfilelist']     = spectargfilelist
    ret['n_aperture']           = n_aperture
    ret['filter']               = filter
    ret['prefix']               = prefix
    ret['calnod']               = calnod
    ret['meancal']              = meancal
    ret['disp']                 = disp
    ret['aplist_cal']           = aplist_cal
    return ret


def cleanec(input, output, **kw):
    """Clean raw echelleogram files of bad pixels.

    Similar to IRAF's 'cosmicrays' or 'crmed' tasks, but this only
    looks along the spectrum's dispersion direction.  The less
    parallel your spectra are to the pixel grid, the less well this
    may work; in this case try increasing the 'window' parameter
    somewhat.

    :INPUTS:

      file : (str or numpy.array) 
           input echellogram file to be cleaned.

    :DEFAULT_OPTIONS:
    
       dict(dispaxis=0, npasses=1, mapout=None, verbose=False, 
                        threshold=300, nsigma=15, window=25, clobber=False, 
                        hdr=None, badmask=None)
           """
    # 2010-08-27 13:34 IJC: CReated
    # 2010-08-30 17:01 IJC: Activated 'npasses' option
    # 2016-10-15 18:13 IJMC: Added "badmask" output option 

    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    
    from pylab import find

    # Check if filein is str or array; load file if the former.  Set
    # dispersion direction via transpose.
    defaults = dict(dispaxis=0, npasses=1, mapout=None, verbose=False, \
                        threshold=300, nsigma=15, window=25, clobber=False, \
                        hdr=None, badmask=None)

    for key in defaults:
        if (not kw.has_key(key)):
            kw[key] = defaults[key]

    verbose = kw['verbose']
    hdr = kw['hdr']
    if verbose: print "CLEANEC begun, input>>", input

    # Test for type of input; iterate, if necessary:
    if input.__class__!=output.__class__:
        print "Files or file lists must be of same type.  Exiting..."
        return
    elif input.__class__==list:
        for ii in range(len(input)):
            if verbose: print "File list, file:  " + input[ii]
            cleanec(input[ii], output[ii], **kw)
        return
    elif (input.__class__==str) and (input[0]=='@'):
        fin  = open(input[ 1::], 'r')
        fout = open(output[1::], 'r')
        if verbose: 
            print "using input file %s and output file %s" % (fin.name, fout.name)
        kw_bak = kw.copy()
        for infile, outfile in zip(fin, fout):
            infile = infile.strip()
            outfile = outfile.strip()
            if verbose:
                print "IRAF-style file list, file %s --> %s" % \
                    (infile.strip(), outfile.strip())
            cleanec(infile, outfile, **kw)
        fin.close()
        fout.close()
        return
    elif input.__class__==str:
        try:
            ec = pyfits.getdata(input)
            hdr = pyfits.getheader(input)
        except:
            if verbose:
                print "Couldn't open file: %s -- adding .fits extension and re-trying" % input
            try:
                input += '.fits'
                if output.find('.fit')<0:
                    output += '.fits'
                ec = pyfits.getdata(input)
                hdr = pyfits.getheader(input)
            except:
                print "PYFITS could not open file: %s" % input
                return
    else:
        try:
            ec = input.copy()
        except:
            print "Input should be a list of filenames, a string, or " + \
                "a Numpy array; it appears to be none of these."

    if kw['dispaxis']!=0:
        ec = ec.transpose()

    ec_original = ec.copy()
    ncol, nrow = ec.shape

    passNumber = 0
    while passNumber<kw['npasses']:
        t0, t1, t2, t25, t27, t3 = 0,0,0,0,0,0
        allrows = arange(nrow)
        for ipix in range(ncol):
            # Set indices and extract the segment to examine
            minind = max(0, ipix - kw['window']/2)
            maxind = min(ncol, ipix + kw['window']/2)
            segs = ec[:, minind:maxind]
            window_size = segs.shape[1]
            residuals = abs(segs - median(segs,1)[:,np.newaxis])

            maxind = argmax(residuals,1)
            goodind = tile(arange(window_size),(nrow,1)) != maxind[:,newaxis]
            goodvals = segs[goodind].reshape(nrow,window_size-1)

            segstd = std(goodvals,1)
            # Measure (a) its discrepancy and (b) stdev of the remainder
            discrepancy = residuals[allrows, maxind]
            sigma = discrepancy/segstd
            repind =  (sigma>kw['nsigma']) * (discrepancy>kw['threshold'])

            prior_value = maxind[repind] - 1
            prior_value[prior_value<0] = 0
            latter_value = maxind[repind] + 1
            latter_value[prior_value<0] = 0

            # If pixel is sufficiently discrepant, throw it out!
            #   if maximum value is _first_, return the latter value
            #   if maximum value is _last_, return the former value
            #   otherwise, return average of former & latter
            firstbadval = maxind[repind]==0
            lastbadval = maxind[repind]==(segs.shape[1]-1)
            midbadval = True ^ (firstbadval + lastbadval)
            maxind2 = maxind[repind][midbadval]
            repind2 = np.where(repind)[0][midbadval]
            
            # Need to apply averaging separately from endpoint replacement.
            segs[repind, maxind[repind]] = \
                firstbadval * segs[repind,1] + \
                lastbadval * segs[repind,segs.shape[1]-2] 
            segs[repind2, maxind2] += \
                0.5 * (segs[repind2, maxind2-1] + segs[repind2, maxind2+1])

        passNumber += 1
    
        if verbose: 
          print 'pass '+str(passNumber)+' complete'
    if hdr is None:
        pyfits.writeto(output, ec, overwrite=kw['clobber'], \
                           output_verify='warn')
    else:
        hdr['cleanec'] ='echelleogram cleaned (%i passes) by nsdata.cleanec' % passNumber
        pyfits.writeto(output, ec, overwrite=kw['clobber'], header=hdr, \
                           output_verify='warn')

    if kw['badmask'] is not None:
        pyfits.writeto(kw['badmask'], (ec!=ec_original).astype(int), overwrite=kw['clobber'], output_verify='warn')
        
    if verbose: print "CLEANEC complete, output to>>", output
    return 


#ir.crmed(f12, f12.replace('fn.fits', 'fn2.fits'), ncmed=15, nlmed=1, ncsig = 25, nlsig = 10, crmask='', median='', sigma='', residual='')

    
def bfixpix(data, badmask, n=4, method = 'Mean', compact_nodes=True,
            balanced_tree = False, n_jobs=-1, eps=0, retdat=False):
    """Replace pixels flagged as nonzero in a bad-pixel mask with the
    average of their nearest n good neighboring pixels.

    :INPUTS:
      data : numpy array (two-dimensional)

      badmask : numpy array (same shape as data)

      method : Which interpolation method to use:
            Supported Methods:
                Mean: average of nearby points
                Median: median of nearby points
                Linear: Mean of nearby points weighted by inverse distance

      rest: kdtree inputs

    :OPTIONAL_INPUTS:
      n : int
        number of nearby, good pixels to average over

    :RETURNS: 
      another numpy array

    :TO_DO:
      Implement new approach of Popowicz+2013 (http://arxiv.org/abs/1309.4224)
    """
    # 2010-09-02 11:40 IJC: Created
    #2012-04-05 14:12 IJMC: Added retdat option
    # 2012-04-06 18:51 IJMC: Added a kludgey way to work for 1D inputs
    # 2012-08-09 11:39 IJMC: Now the 'n' option actually works.
    # 2015-10-30 12:07 IJMC: Fixed (x,y) coordinate offset; thanks to
    #                        Jordan Stone of UA for this!
    # 2018-5-21  15:44 NM: Implemented KD tree 

    from scipy import spatial
    if data.ndim == 1:
      data = np.tile(data,(3,1))
      badmask = np.tile(badmask, (3,1))
      ret = bfixpix(data,badmask,n=n,retdat=True)
      return ret[1]

    if retdat:
      ret = np.array(data,copy=True)
    else:
      ret = data

    badpix = np.transpose(np.nonzero(badmask))
    goodpix = np.transpose(np.nonzero(1-badmask))

    tree  = spatial.cKDTree(goodpix,compact_nodes = compact_nodes,balanced_tree = balanced_tree)
    dd,ii = tree.query(badpix,n,eps=eps,n_jobs=n_jobs)

    all_neighbors = goodpix[ii].transpose(0,2,1)

    if method.lower().strip()   == 'mean':
        interp_values = np.mean([data[neighbors[0],neighbors[1]] for neighbors in all_neighbors],1)
    elif method.lower().strip() == 'median':
        interp_values = np.median([data[neighbors[0],neighbors[1]] for neighbors in all_neighbors],1)
    elif method.lower().strip() == 'linear':
        interp_values = np.average([data[neighbors[0],neighbors[1]] for neighbors in all_neighbors],1,1/dd)
    else:
        raise ValueError("Method argument not valid. Supported methods are: Mean, Median, Linear")

    ret[tuple(badpix[:,0]), tuple(badpix[:,1])] = interp_values

    return ret

def posofend(str1, str2):
    """returns the position immediately _after_ the end of the occurence
    of str2 in str1.  If str2 is not in str1, return -1.
    """
    pos = str1.find(str2)
    if pos>-1:
        ret= pos+len(str2)
    else:
        ret = -1
    return ret

def readart(filename, ignore='***'):
    """Read a Keck .arT telemetry file.
    """
    # 2010-09-04 01:11 IJC: Created
    # 2014-12-18 21:16 IJMC: Added 'ignore' option.

    from spitzer import genericObject


    def process(struct, valstr, names, types, delim=','):
        """Append a line of values to the output object.
        """
        vals = valstr.split(delim)
        for val, name, type in zip(vals, names, types):
            if type=='DBL':
                exec 'struct.%s.append(float(val.strip()))' % name in locals()
            else:
                exec 'struct.%s.append(val.strip())' % name in locals()
        return

    try:
        f = open(filename, 'r')
    except:
        print "Could not open file: %s" % filename
        return -1
    
    output = genericObject()

    hdrlines = [f.readline() for ii in range(3)]
    pos00, pos01 = posofend(hdrlines[0], 'INTERVAL ='), hdrlines[0].find(',')
    pos10, pos11 = posofend(hdrlines[0], 'NO_ELEMENTS ='), hdrlines[0].find('\n')
    interval = int(hdrlines[0][pos00:pos01])
    no_elements = int(hdrlines[0][pos10:pos11])

    entry_names = [name.strip().replace(':','_').replace('.','_') for name in hdrlines[1].replace('"', '').split(',')]
    entry_types = [name.strip().replace(':','_').replace('.','_') for name in hdrlines[2].replace('"', '').split(',')]
    entry_types = [name.strip() for name in hdrlines[2].replace('"', '').split(',')]

    output.interval = interval
    output.no_elements = no_elements
    output.keys = entry_names
    output.types = entry_types
    for name in entry_names:
        exec 'output.%s = []' % name in locals()
    
    for nextline in f:
        if not (ignore in nextline):
            process(output, nextline, entry_names, entry_types)

    return output

def readnstemps(filename):
    """Read the Keck telemetry file "nirspecTemps"
    """
    # 2010-09-04 17:46 IJC: Created

    from spitzer import genericObject

    try:
        f = open(filename, 'r')
    except:
        print "Could not open file: %s" % filename
        return -1

    output = genericObject()
    line0 = f.readline()
    pos0 = posofend(line0, 'GMT')
    output.time = [line0[0:pos0]]
    vals = map(float, line0[pos0::].strip().split(' '))
    nvals = len(vals)
    valnames = ['t%i' % ii for ii in range(nvals)]
    for val, name in zip(vals, valnames):
        exec 'output.%s = [val]' % name in locals()

    for line in f:
        vals = map(float, line[pos0::].strip().split(' '))        
        output.time.append(line[0:pos0])
        for val, name in zip(vals, valnames):
            exec 'output.%s.append(val)' % name in locals()

    return output

def readMagiqlog(filename, dat=['guidestats']):
    """
    Read specified types of data from the Keck Magiq telemetry log.

    filename : str -- logfile name

    dat : list of str -- types of data to return
         guidestats -- centroid x/y, fwhm, star & skyflux
         """
    # 2010-09-05 12:07 IJC: Created

    def processGuidestats(struct, string):
        pos_t0, pos_t1 = 0, string.find('[LegacyCam]')
        pos_x0, pos_x1 = posofend(string, 'Centroid x='), string.find('y=')
        pos_y0, pos_y1 = posofend(string, 'y='), string.find('fwhm=')
        pos_f0, pos_f1 = posofend(string, 'fwhm='), string.find('star=')
        pos_st0, pos_st1 = posofend(string, 'star='), string.find('sky=')
        pos_sk0, pos_sk1 = posofend(string, 'sky='), string.find('stats=')
        
        struct.HSTtime.append(string[pos_t0:pos_t1])
        struct.cenx.append(float(  string[pos_x0:pos_x1]))
        struct.ceny.append(float(  string[pos_y0:pos_y1]))
        struct.fwhm.append(float(  string[pos_f0:pos_f1]))
        struct.star.append(float(  string[pos_st0:pos_st1]))
        struct.sky.append(float(   string[pos_sk0:pos_sk1]))

        return
    
    from spitzer import genericObject

    try:
        f = open(filename, 'r')
    except:
        print "Could not open file: %s" % filename
        return -1

    output = genericObject()

    for type in dat:
        if type=='guidestats':
            output.guide = genericObject()
            output.guide.HSTtime = []
            output.guide.cenx = []
            output.guide.ceny = []
            output.guide.fwhm = []
            output.guide.star = []
            output.guide.sky = []
        else:
            print "Unknown type of Magiq log data: %s" % type
            
    for line in f:
        for type in dat:
            if type=='guidestats':
                if line.find('CamImage')<0 or line.find('Centroid')<0 or line.find('fwhm')<0:
                    pass
                else:
                    processGuidestats(output.guide, line)
            else:
                pass

    return output


def readarm(filename):
    """Read a Keck .arM telemetry file.

    So far, tested only with envAut.arM
    """
    # 2010-09-04 01:11 IJC: Created

    from spitzer import genericObject


    def process(struct, line, delim=','):
        """Append a line of values to the output object.
        """
        vals = line.split(delim)
        keyname = vals[2].strip().replace(':','_').replace('.','_').replace('---','')
        endofkey = posofend(line, vals[2])
        endofdelim = line.find(delim, endofkey)+1
        keyval = line[endofdelim::].strip()

        try:
            keyval = float(keyval)
        except:
            pass
        
        if keyval=='***':
            pass
        else:
            #struct.HSTtime.append(vals[0].strip())
            #struct.HSTdate.append(vals[1].strip())
            HSTdatetime = '%s %s' % (vals[0].strip(), vals[1].strip())
            try:
                exec 'struct.%s.append(keyval)' % keyname in locals()
                exec "struct.%sHST.append(HSTdatetime)" % keyname in locals()
            except:
                exec 'struct.%s = [keyval]' % keyname in locals()
                exec "struct.%sHST = [HSTdatetime]" % keyname in locals()
        
        return

    try:
        f = open(filename, 'r')
    except:
        print "Could not open file: %s" % filename
        return -1
    
    output = genericObject()
    hdrlines = [f.readline() for ii in range(2)]
    pos00, pos01 = posofend(hdrlines[0], 'INTERVAL ='), hdrlines[0].find(',')
    pos10, pos11 = posofend(hdrlines[0], 'NO_ELEMENTS ='), hdrlines[0].find('\n')
    interval = int(hdrlines[0][pos00:pos01])
    no_elements = int(hdrlines[0][pos10:pos11])

    entry_names = [name.strip().replace(':','_').replace('.','_') for name in hdrlines[1].replace('"', '').split(',')]

    output.interval = interval
    output.no_elements = no_elements
    output.HSTtime = []
    output.HSTdate = []
    output.HSTdatetime = []

    for line in f:
        process(output, line)
        

    return output
        
    #    entry_names = [name.strip().replace(':','_').replace('.','_') for name in hdrlines[1].replace('"', '').split(',')]
    #entry_types = [name.strip().replace(':','_').replace('.','_') for name in hdrlines[2].replace('"', '').split(',')]
    #entry_types = [name.strip() for name in hdrlines[2].replace('"', '').split(',')]

def envMet(filename, tz=-10, planet=None, date=None, ignore='***'):
    """Read the envMet.arT Keck telemetry file.

    :INPUT:
       filename -- str.

    :OPTIONAL_INPUTS:
       tz: offset (in hours) from GMT (HST = -10)

       planet: an.planet object to compute orbital phase & HJD (if desired)

       date: observation run date; grab time tags from this to use for
              averaging over (not yet implemented!!!)
    """
    # 2010-09-07 13:02 IJC: Created
    # 2014-12-18 21:16 IJMC: Added 'ignore' option.

    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    
    try:
        import astrolib
    except:
        try:
            import PyAstronomy.pyasl as astrolib
        except:
            raise ImportError('Astrolib/PyAstronomy not found. Install PyAstronomy from here: http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/index.html')
    
    met = readart(filename, ignore=ignore)

    if met!=-1:
        met.HSTdatetime = ['%s %s' % (ddate, time) for ddate,time in zip(met.HSTdate, met.HSTtime)]

        met.jd = [gd2jd(datetime)-tz/24. for datetime in met.HSTdatetime]

        if planet is not None:
            met.hjd = [astrolib.helio_jd(jjj - 2.4e6, planet.ra, planet.dec) + 2.4e6 for jjj in met.jd]
            met.phase = planet.phase(met.hjd)

        if date is not None:
            obs = initobs(date)
            rawfiles = obs['rawtargfilelist']
            jd_start = []
            exptime = []
            try:
                hdr = pyfits.getheader(file)
                jd_start.append(hdr['jd'])
                exptime.append(hdr['exptime'])
            except:
                jd_start = []

            # Loop over every observation timestap, averaging over each header keyword
            met.bin = met
            met.bin.jd = jd_start
#            for ii, jd0 in enumerate(jd_start):
#                thisindex = 
            
            

    return met


def darkbpmap(filelist, clipsigma=5, sigma=25, writeto=None, clobber=False, verbose=False, outtype=int, filtwid=9):
    """Use dark frames to construct a bad pixel map based on unusually
       variable pixels.

      Marks pixels as bad according to 3 criteria:
      1 - pixels whose (outlier subtracted) std-dev is [sigma] sigmas from typical pixel (outlier subtracted) std-dev
        i.e. pixels which have particularly high/low variance
      2 - pixels which dont vary (in outlier subtraced sense)
      3 - pixel's whose value is [SIGMA] sigma discrepent from their neighbors
        i.e. pixels which are discrepent from neighbors

    :INPUT:
        filelist: str, list, or 3D numpy array -- dark frame filenames or data
            str -- IRAF-style file list (beginning with '@' symbol)
            list -- Python-style list of strs
            numpy array -- 3D (L*M*N) stack of L dark frames

    :OPTIONS:
        clipsigma : scalar -- significance threshold for removing transient
                   (cosmic-ray-like) events

        sigma : scalar -- significance threshold for greater-than-average
                pixel variability.

        writeto : str -- filename to write output to.  If None, returns the array.

    :RETURNS:
      if writeto is None:
          a 2D boolean numpy array: True for bad pixels, False for other pixels.
      else:
          returns None
    """
    # 2010-09-08 09:56 IJC: Created
    # 2014-12-17 21:22 IJMC: Added 'ignore_missing_end' 
    # 2016-10-15 18:35 IJMC: Added "outtype" option
    # 2016-10-15 19:36 IJMC: Stuck pixels get flagged. filtwid test.
    
    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    
    from analysis import stdr, meanr
    from scipy.signal import medfilt2d
    
    darkstack = []
    if filelist.__class__==list:
        for line in filelist:
            if verbose: print "File list, file:  " + line
            try:
                darkstack.append(pyfits.getdata(line, ignore_missing_end=True))
            except:
                darkstack.append(pyfits.getdata(line+'.fits', ignore_missing_end=True))
    elif (filelist.__class__==str) and (filelist[0]=='@'):
        fin  = open(input[ 1::], 'r')
        if verbose: 
            print "using input file %s" % (fin.name)
        for line in fin:
            line = line.strip()
            if verbose: print "IRAF-style file list, file %s" % line
            try:
                darkstack.append(pyfits.getdata(line, ignore_missing_end=True))
            except:
                darkstack.append(pyfits.getdata(line+'.fits', ignore_missing_end=True))
    else:
        darkstack = filelist

    darkstack = array(darkstack)

    darkstd = stdr(darkstack, axis=0, nsigma=clipsigma, verbose=verbose-1)
    edark = stdr(darkstd.ravel(), nsigma=clipsigma, verbose=verbose-1)
    mdark = meanr(darkstd.ravel(), nsigma=clipsigma, verbose=verbose-1)

    lostd = medfilt2d(darkstd, filtwid)
    lodark = medfilt2d(np.median(darkstack, axis=0), filtwid)
    hidark = np.median(darkstack, axis=0) - lodark

    badpixelmap = ((abs(darkstd-mdark) / edark) > sigma) + \
                  (darkstd==0) + (np.abs(hidark/lostd) > sigma)
    if writeto is None:
        ret = badpixelmap
    else:
        ret = None
        pyfits.writeto(writeto, badpixelmap.astype(outtype), overwrite=clobber)

    return ret

    
def cutoffmask(filename, cutoff=[0, Inf], writeto=None, clobber=True):
    """Create a simple mask from a FITS frame or array, based on whether
       its values are above or below specified cutoff values.

    :INPUT:
        filename: str or numpy array --  frame filename or data
            str -- filename of FITS frame to load
            numpy array -- data frame

    :OPTIONS:
        cutoff : list -- of form [lower, higher]; values below 'lower' or
                above 'higher' will be flagged as bad.

        writeto : str -- filename to write output to.  If None, returns the array.

    :RETURNS:
      if writeto is None:
          a 2D boolean numpy array: True for bad pixels, False for other pixels.
      else:
          returns None
    """
    # 2010-09-08 10:36 IJC: Created
    

    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    
    from analysis import stdr, meanr

    if (filename.__class__==str):
        try:
            data = pyfits.getdata(filename)
        except:
            data = pyfits.getdata(filename+'.fits')
    else:
        data = filename

    badpixelmap = (data < cutoff[0]) + (data>cutoff[1])

    if writeto is None:
        ret = badpixelmap
    else:
        ret = None
        pyfits.writeto(writeto, badpixelmap.astype(float), overwrite=clobber)

    return ret

    

def subab(afiles, bfiles, outfiles, clobber=False):
    """
    Take raw A & B nod files from a set of observations and subtract
    them; output the difference images.

    :INPUTS:
       afiles -- list of A-position filenames
       
       bfiles -- list of B-position filenames

       outfiles -- list of output filename

    :NOTE:
       All input lists will be truncated to the length of the shortest
       input list.
       
       For now, the output file has the header of th A-file of the
       pair; this is not optimal and should be fixed to record both A
       & B headers!

    :EXAMPLE:
      import nsdata as ns
      inita = ns.initobs('2008jul12A')
      initb = ns.initobs('2008jul12B')
      afiles = inita[12][0][1:-4]
      bfiles = initb[12][0][:-4]
      outfiles = [(os.path.split(bfn)[0] + '/%s-%s' % \
            (os.path.split(afn)[1], os.path.split(bfn)[1])).replace('.fits', '') + \
            '.fits' for afn, bfn in zip(afiles, bfiles)]
      ns.subab(afiles, bfiles, outfiles)

    """
    # 2010-11-29 08:43 IJC: Created


    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits
    
    import os
    
    for afn, bfn, outfn in zip(afiles, bfiles, outfiles):
        if not os.path.isfile(afn):
            print "file %s not found, skipping." % afn
            readfiles = False
        elif not os.path.isfile(bfn):
            print "file %s not found, skipping." % bfn
            readfiles = False
        else:  # File was found
            try:
                adata = pyfits.getdata(afn, ignore_missing_end=True)
                bdata = pyfits.getdata(bfn, ignore_missing_end=True)
                ahdr = pyfits.getheader(afn, ignore_missing_end=True)
                bhdr = pyfits.getheader(bfn, ignore_missing_end=True)
                readfiles = True
            except:
                readfiles = False
                print "Could not read data or header from FITS files " + \
                    "(either %s or %s)" % (afn, bfn)

        if readfiles:
            # Test for non-standard NIRSPEC keywords
            if ahdr.has_key('gain.spe'):
                ahdr.rename_key('gain.spe', 'gain_spe')
            if ahdr.has_key('freq.spe'):
                ahdr.rename_key('freq.spe', 'freq_spe')
            if bhdr.has_key('gain.spe'):
                bhdr.rename_key('gain.spe', 'gain_spe')
            if bhdr.has_key('freq.spe'):
                bhdr.rename_key('freq.spe', 'freq_spe')


            diff = adata - bdata
            pyfits.writeto(outfn, diff, ahdr, ignore_missing_end=True, overwrite=clobber)
                
    return

def nAir_old(vaclam, T=288.15, P=101325.):
    """Return the index of refraction of air at a given wavelength.

    :INPUTS: 

       vaclam -- scalar or Numpy array -- Vacuum wavelength (in
                 microns) at which to calculate n
    
       T -- scalar -- temperature in Kelvin
       
       P -- scalar -- pressure in Pascals.

    :NOTES:

       This assumes a dry atmosphere with 0.03% CO2 by volume ("standard air")

    :REFERENCE: 

        81st CRC Handbook (c.f. Edlen 1966)
    """
    # 2011-03-10 14:12 IJC: Created

    print "This function is outdated; please use nAir() instead."

    sigma2 = 1. / vaclam**2

    nm1_stp = 1e-8 * (8342.13 + 2406030. / (130. - sigma2) + 15997. / (38.9 - sigma2))

    tp_factor =(P * (1. + P * (61.3 - (T - 273.15)) * 1e-10)) / (96095.4 * (1. + 0.003661 * (T - 273.15) ) )

    n = 1. + nm1_stp * tp_factor

    #print (nm1_stp * 1e8), (n - 1) * 1e8
    return n

def nAir(vaclam, T=293.15, P=1e5, fco2=0.0004, pph2o=0.):
    """Return the index of refraction of air at a given wavelength.

    :INPUTS: 

       vaclam: scalar or Numpy array
              Vacuum wavelength (in microns) at which to calculate n
    
       T : scalar
           temperature in Kelvin
       
       P : scalar
           pressure in Pascals

       fc02 : scalar
           carbon dioxide content, as a fraction of the total atmosphere

       pph2o : scalar
           water vapor partial pressure, in Pascals

    :REFERENCE: 
specfi        Boensch and Potulski, 1998 Metrologia 35 133
    """
    # 2011-10-07 15:14 IJMC: Created
    # 2012-12-05 20:47 IJMC: Explicitly added check for 'None' option inputs.

    if T is None:
        T = 293.15
    if P is None:
        P = 1e5
    if fco2 is None:
        fco2 = 0.0004
    if pph2o is None:
        pph2o = 0.0

    sigma = 1./vaclam
    sigma2 = sigma * sigma

    # (Eq. 6a)
    nm1_drystp =  1e-8 * (8091.37 + 2333983. / (130. - sigma2) + 15518. / (38.9 - sigma2))

    # Effect of CO2 (Eq. 7):
    nm1_dryco2 = nm1_drystp * (1. + 0.5327 * (fco2 - 0.0004))

    # Effect of temperature and pressure (Eq. 8):
    nm1_dry = ((nm1_dryco2 * P) * 1.0727933e-5) * \
        (1. + 1e-8 * (-2.10233 - 0.009876 * T) * P) / (0.0036610 * T)

    # Effect of H2O (Eq. 9):
    try:
        n = 1. + (nm1_dry - pph2o * (3.8020 - 0.0384 * sigma2) * 1e-10).astype(float64)
    except:
        n = 1. + (nm1_dry - pph2o * (3.8020 - 0.0384 * sigma2) * 1e-10)

    return n


    
def correct_aries_crosstalk(input, **kw): 
    """Correct ARIES spectroscopy frames for detector crosstalk.

    :INPUTS:
         input = 

         output : None or str
           output filename. If None, input files are overwritten (if clobber=True)

         corquad : str
           Path to "corquad" executable that does the actual correction.

         clobber : bool
           overwrite file if input and output files are the same

    :DEFAULT: 
         exec=os.path.expanduser('~/python/corquad-linux'), clobber=False

    :OUTPUTS:
         None

    :SEE_ALSO:
        the discussion on the ARIES wiki:
        http://aries.as.arizona.edu/index.php/Reducing_ARIES_Data

    NOTE: may give trouble ifinput/output don't contain full, absolute paths.
    """
    # 2016-10-18 15:25 IJMC: Created
    import shutil
    
    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits


    defaults = dict(corquad=os.path.expanduser('~/python/corquad-linux'), clobber=False, verbose=False, output=None)
    for key in defaults:
        if (not kw.has_key(key)):
            kw[key] = defaults[key]

                # Parse inputs:
    if kw['output'] is None: 
        output = input
    else:
        output = kw['output']

    if kw['corquad'] == "":
      corquad = defaults['corquad']
    else:
      corquad = kw['corquad']

    clobber = kw['clobber']
    verbose = kw['verbose']
    if input.__class__!=output.__class__:
        print "Files or file lists must be of same type.  Exiting..."
        return
    # Calls this function on each file in list
    elif input.__class__==list:
        for ii in range(len(input)):
            if verbose: print "File list, file:  " + input[ii]
            kw['output'] = output[ii]
            correct_aries_crosstalk(input[ii], **kw)
        return
    # Calls this function on each file in @string
    elif (input.__class__==str) and (input[0]=='@'):
        fin  = open(input[ 1::])
        fout = open(output[1::])

        for line in fin:
            if verbose: print "IRAF-style file list, file:  " + line.strip()
            kw['output'] = fout.readline().strip()
            correct_aries_crosstalk(line.strip(), **kw)
        fin.close()
        fout.close()
    if not os.path.isfile(corquad):
        # print "Error: 'corquad' executable '%s' not found. Exiting." % corquad
        # return
        raise RuntimeError("'corquad' executable '%s' not found. Exiting." % corquad)

    if clobber and input!=output:
        ir.imdelete(output)


    input = findfitsfile(input)
    if '.fits' not in output:
        output += '.fits'

    pwd = os.getcwd()
    input_path, input_filename = os.path.split(input)
    output_path, output_filename = os.path.split(output)
    corquad_exec = os.path.split(corquad)[1]
    execstr = './%s %s' % (corquad_exec, input_filename)



    if not os.path.isfile(pwd+'/'+corquad_exec):
        shutil.copy(corquad, pwd)


    # Move input to pwd, run corquad
    # Move input to input_path, output to output_path
    os.rename(input, pwd+'/'+input_filename)
    os.system(execstr)
    if verbose: 
        print "Executing: '%s'..." % execstr
    os.rename(pwd+'/'+input_filename, input)

    # Check corquad ran
    if os.path.isfile('q' + input_filename):
        os.rename('q'+input_filename, output)
    else:
        print "WARNING, did not find expected file 'q%s'. corquad might have failed." % input_filename

    # Check output is there
    # Does not necessarily mean corquad has run
    # If output, input have same name, isn't helpful
    if not os.path.isfile(output):
        print "WARNING, did not find expected output file '%s'. Something went wrong!" % output

    pyhdr = pyfits.open(output)
    pyhdr[0].header['quadnois'] = 'ARIES crosstalk fixed by nsdata.correct_aries_crosstalk'
    pyhdr.writeto(output, overwrite=True, output_verify='silentfix')

    return

def findfitsfile(filename, suffix='.fits'):
    """If filename exists, return filename. Otherwise, return filename + suffix. """
    # 2016-10-18 19:58 IJMC: Created
    if os.path.isfile(filename):
        ret = filename
    else:
        ret = filename + suffix
    return ret
