""" 
  Contains tools to read from json file and easily modify the database there
"""
import json as _json
from collections import MutableMapping
import os



"""
Loads a json database into memory as a dict
  requires:
    path: path to foo.json file
  methods:
    reload: reloads dict from json file
    write : writes dict to file - allows for db updates
"""
class json(MutableMapping):
  def __init__(self, path):
    self.store = dict()
    self.path = path
    with open(path) as f:
      data = _json.load(f)
    self.update(data)

  def __getitem__(self, key):
    return self.store[self.__keytransform__(key)]

  def __setitem__(self, key, value):
    self.store[self.__keytransform__(key)] = value

  def __delitem__(self, key):
    del self.store[self.__keytransform__(key)]

  def __iter__(self):
    return iter(self.store)

  def __len__(self):
    return len(self.store)

  def __keytransform__(self, key):
    return key

  def reload(self):
    self.store = dict()
    with open(self.path) as f:
      data = _json.load(f)
    self.update(data)

  def write(self):
    with open(self.path,'w') as f:
      _json.dump(self.store, f, indent=2)

"""
Sample code for adding data to database
"""
def addToDb(dbname):
  obsdb = json(dbname)
  
  date = '2016oct20'
  data = {}

  _aphome = os.path.expanduser('~')[1:].replace('/','_')

  data['planet']       = 'WASP-33 b'
  data['prefix']       = 'spec_'

  # Full Range
  data['fullframelist'] = range(118,191+1)
  data['framelist']    = range(118,191+1)


  data['flatlist']     = {
                            90: [26,27,28,29,30,55,56,57],
                            85: [31,54],
                            80: [32,53],
                            75: [33,52],
                            70: [34,51],
                            65: [35,50],
                            60: [36,49],
                            55: [37,48],
                            50: [38,47],
                            45: [39,46],
                            40: [40,45],
                            35: [41,44],
                            30: [42,43]
                         }
  data['darklist']     = range(193, 207+1)
  data['darkflatlist'] = range(193, 207+1)
  data['darkcallist']  = range(193, 207+1)
  data['callist']      = range(118,122+1)
  data['datadir']      = '2016oct20/'
  data['ap_suffix']    = 'database/ap_' + '_dash_exobox_proj_pcsa_data_proc_2016oct20'
  data['n_aperture']   = 6
  data['filter']       = 'OPEN6'
  data['calnod']       = False

  obsdb[date] = data
  obsdb.write()
  print "written!"

def print3(arg):
  print arg

def collectData(path,prefix,date,headers=['object'], targObject='',output=""):
  from astropy.io import fits

  outstr  = ""
  l       = 10
  outstr+= 'index:'.ljust(l+1)
  for key in headers:
    outstr += (str(key)+":").ljust(l)

  outstr += "Altitude:".ljust(l)

  #setup outputfile
  printfn = print3
  endline = ""
  if output == 'stdout':
    pass
  else:

    if output == "":
      output = date + "_summary.txt"
    
    if os.path.exists(output):
      print "WARNING, output file already exists. Aborting"
      return
    outfile = open(output,'w')

    printfn = outfile.write
    endline = "\n"
  outstr += endline
  printfn(outstr)

  #Write headers, collapse targObject
  filelist = os.listdir(path)
  print 'examining '+ str(len(filelist)) +' files.'
  objRange = []
  emptyval = ""
  invalid  = "-----"
  objDict  = dict((key,emptyval) for key in headers)


  for i,file in enumerate(filelist):
    if i % 500 == 1:
      print 'Through file ' + str(i-1)+'.'

    if not prefix in file:
      continue
    else:
      filename = file.split('.')[0]
      filenum = int(filename.split(prefix)[1])

      head = fits.getheader(path+"/"+file,ignore_missing_end=True)
      isTarg = (head['object'].lower() == targObject.lower())

      if isTarg:
        for key in headers:
          if objDict[key] == emptyval:
            objDict[key] = head[key]
          elif objDict[key] == invalid:
            continue
          else:
            if objDict[key] != head[key]:
              objDict[key] = invalid

        objRange.append(filenum)

      #not target object
      else:
        filestr = str(filenum).ljust(l+1)
        for key in headers:
          filestr += (str(head[key])).ljust(l)

        import nsdata as ns
        angle = ns.convertAirmassToAltitude(head['airmass'])
        filestr += str(angle).ljust(l)
        filestr += endline
        printfn(filestr)

  #after loop
  if len(objRange) > 0:
    objStr = (str(min(objRange))+'-'+str(max(objRange))).ljust(l+1)
    for key in headers:
      objStr += str(objDict[key]).ljust(l)
    objStr += endline

    printfn(objStr)



