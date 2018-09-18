try:
  import cPickle as pickle
except:
  import pickle

import numpy as np
from astropy.io import fits
from pathlib import Path
verbose = True



if verbose:
  print("Begining Pickling Process")
  print("Reading Data...")

# Computer can only handle so many open files at a time
maxChunkSize = 123
suffix    = 'int.fits'
data     = '2016oct17' #wasp33
_proc    = '/dash/exobox/proj/pcsa/data/proc/' + data + '/'

spectarg = _proc+'spectarg'

list_spectarg = np.loadtxt(spectarg,str)

subset_ranges = np.array_split(list_spectarg, int(len(list_spectarg)/maxChunkSize) +1)

fluxes = []
errors = []

hdr_vals = {}


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
      for key in hdr.keys():
        if key == '':
          continue

        if key not in hdr_vals:
          hdr_vals[key] = [hdr[key]]
        else:
          hdr_vals[key].append(hdr[key])

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

if verbose:
  print('Writing Data')

for order in range(len(fluxes)):
  if verbose:
    print('Writing order ' + str(order+1) +'/'+ str(len(fluxes)))

  toPickle = {
    'fluxes': np.array(fluxes[order]),
    'errors': np.array(errors[order]),
    'waves' : np.array(waves[order])
  }

  saveName = _proc+'order_'+str(order)+'.pickle'

  with open(saveName,'wb') as pf:
    pickle.dump(toPickle,pf, protocol=pickle.HIGHEST_PROTOCOL)

if verbose:
  print('Writing Header')
hdr_saveName = _proc+'allHeaders.pickle'
with open(hdr_saveName,'wb') as pf:
  pickle.dump(hdr_vals, pf, protocol=pickle.HIGHEST_PROTOCOL)

if verbose:
  print('All Done!')
