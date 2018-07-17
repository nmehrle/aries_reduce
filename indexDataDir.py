from astropy.io import fits
use_tqdm=False
try:
  from tqdm import tqdm
  use_tqdm=True
except ModuleNotFoundError:
  pass
import os
'''
  This is intended to allow one to easily index a directory full of fits files. 

  It searches a directory for Fits files and outputs their prefix, filenum, and header values. Assumes files are in the 'prefix_filenum.fits' format. 
'''


def indexDir(path, headerKeys=['object'], output=None, overwrite='verify'):
  colLen = 10
  invalid = '------'.ljust(colLen)

  hdstr = 'prefix:'.ljust(colLen+1)
  hdstr += 'index:'.ljust(colLen+1)
  for key in headerKeys:
    hdstr += (str(key)+':').ljust(colLen)

  # Set parameters for writing to console vs file
  if output == 'stdout':
    printfn = print
    endline = ''
  else:
    if output == None:
      output = path.split('/')[-1]+'_summary.txt'
    if os.path.exists(output):
      print('Warning, output file ' + output+' already exists')
      if overwrite == 'verify':
        print('Enter "overwrite" to overwrite')
        user_input = input()
        if user_input.lower().strip() == 'overwrite':
          pass
        else:
          return
      elif overwrite == False:
        print('Aborting')
        return
      elif overwrite == True:
        print('Overwriting')


    outfile = open(output,'w')

    printfn = outfile.write
    endline = '\n'

  hdstr+=endline
  printfn(hdstr)

  filelist = os.listdir(path)
  filelist.sort()
  
  sequence = enumerate(filelist)
  if use_tqdm:
    sequence = tqdm(enumerate(filelist))

  for i,file in sequence:
    if not use_tqdm and i%100 == 0:
      print('On file '+str(i+1)+' of '+str(len(filelist)))
    fileStr = ""

    fileSplit = file.split('.')
    if fileSplit[-1] != 'fits':
      continue

    preIndSplit = fileSplit[0].split('_')
    if len(preIndSplit) != 2:
      print('Warning, file '+file+' found not confirming to prefix_filenum.fits standard')
      continue
    prefix = preIndSplit[0]
    filenum = preIndSplit[1]

    fileStr += prefix.ljust(colLen+1)
    fileStr += str(filenum).ljust(colLen+1)

    header = fits.getheader(path+'/'+file,ignore_missing_end=True)
    for key in headerKeys:
      if key not in header:
        fileStr += invalid
      else:
        fileStr += (str(header[key])).ljust(colLen)

    fileStr += endline
    printfn(fileStr)






