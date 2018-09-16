import os
import json
from astropy.io import fits
use_tqdm=False
try:
  from tqdm import tqdm
  use_tqdm=True
except ModuleNotFoundError:
  pass

#-- User Facing Funcitons:
# Use this to write to temporary JSON file
def generateSummaryFiles():
  data_parent = '/dash/exobox/proj/pcsa/data/raw/'
  data_prefix = '2016oct'
  dates = [15,16,17,18,19,20,21]
  dest  = '/dash/exobox/proj/pcsa/data/summaries/'

  headerKeys = ['object', 'exptime']

  for date in dates: 
    data_dir = data_parent + data_prefix + str(date)
    data_sum = dest + data_prefix + str(date) + '_summary.txt'
    writeSummary(data_dir, data_sum, headerKeys)

def generateDBForValidation():
  data_parent = '/dash/exobox/proj/pcsa/data/raw/'
  data_prefix = '2016oct'
  dates = [15,16,17,18,19,20,21]
  proc_dir  = '/dash/exobox/proj/pcsa/data/summaries/'

  dbNames = ['dbtmp_'+str(date)+'.json' for date in dates]
  master_db = {}
  for i,date in enumerate(dates):
    fullDate = data_prefix+str(date)
    path = data_parent+fullDate+'/'

    date_db = addToDb(path, fullDate, proc_dir, output=dbNames[i], write=False)

    validateKeys(date_db,master_db)
    master_db.update(date_db)

  with open('dbtmp.json','w') as f:
    json.dump(master_db, f, indent=2, sort_keys=True)

def generateFullDB():
  db1 = './obsdb.json'
  db2 = './dbtmp.json'

  outdb = 'outputdb.json'

  mergeDB(db1,db2, output=outdb)
###

#-- Generate Database information
# Sorts through path and returns a collection of filenames with entered keywords
def indexDir(path, headerKeys=['object']):
  # Set up return index
  index = {}
  index['filename'] = []
  index['prefix']   = []
  index['index']    = []

  for hk in headerKeys:
    index[hk] = []

  # Collect files in dir
  filelist = os.listdir(path)
  filelist.sort()

  sequence = enumerate(filelist)
  if use_tqdm:
    sequence = tqdm(enumerate(filelist), desc=path)
  for i,file in sequence:
    if not use_tqdm and i%100 == 0:
      print('On file '+str(i+1)+' of '+str(len(filelist)))

    # Get Fits Files
    fileSplit = file.split('.')
    if fileSplit[-1] != 'fits':
      continue

    # Two naming conventions
    preIndSplit = fileSplit[0].split('_')
    if len(preIndSplit) != 2:
      preIndSplit = [fileSplit[0][:-4], fileSplit[0][-4:]]

    prefix = preIndSplit[0]
    filenum = preIndSplit[1]

    # Load in filename params 
    index['filename'].append(file)
    index['prefix'].append(prefix)
    index['index'].append(filenum)

    try:
      header = fits.getheader(path+'/'+file,ignore_missing_end=True)
    except OSError:
      print('file: '+file+' seems corrupt')
      header = []

    for hk in headerKeys:
      if hk not in header:
        index[hk].append(None)
      else:
        index[hk].append(header[hk])

  return index

def addToDb(path, date, procDir, output ='dbtmp.json',
            headerKeys=['object','exptime'], prefix='spec_',
            write=True
):
  dataIndex = indexDir(path, headerKeys)
  planets, *frameLists = basicIndexing(dataIndex) 
  allData = {}
  for i,planet in enumerate(planets):
    if len(planets) > 1:
      dateKW = date+chr(97+i)
    else:
      dateKW = date

    planetData = makePlanetDatabase(planet, prefix, dataIndex, *frameLists)
    planetData['framelist'] = planetData['fullframelist']
    planetData['datadir']   = dateKW+'/'
    planetData['date']      = dateKW
    planetData['ap_suffix'] = 'database/ap_' + procDir.replace('/','_')+dateKW
    planetData['n_aperture'] = 10
    planetData['filter']    = 'Karies'
    planetData['calnod']    = False

    allData[dateKW] = planetData

  if write:
    with open(output,'w') as f:
      json.dump(allData,f,indent=2,sort_keys=True)

  return allData
 
def basicIndexing(dataIndex):
  flatKeys = ['flat','arc','fringe','elev']
  darkKeys = ['dark']

  rejectObjs = ['test','focus','_sky']
  rejectPrefix = ['im','testimg','testimg','sky']

  flatFrames = []
  darkFrames = []
  scienceFrames = []
  
  planets = []
  
  for i,obj in enumerate(dataIndex['object']):
    isPlanet = True
    pre = dataIndex['prefix'][i]

    if pre in rejectPrefix:
      continue

    if any(rejectKey in obj for rejectKey in rejectObjs):
      continue

    if any(flatKey in obj for flatKey in flatKeys):
      flatFrames.append(i)
      isPlanet = False

    if any(darkKey in obj for darkKey in darkKeys):
      darkFrames.append(i)
      isPlanet = False

    if isPlanet:
      scienceFrames.append(i)
      if obj in planets:
        pass
      else:
        planets.append(obj)

  return planets, scienceFrames, darkFrames, flatFrames

def makePlanetDatabase(planet, prefix, dataIndex, scienceFrames, darkFrames, flatFrames):
  data = {}
  data['planet'] = planet
  data['prefix'] = prefix

  fullframelist = []
  darklist      = []
  darkflatlist  = []

  planetExpTime = 0
  flatExpTime   = 0

  # if any flat frame has an associated number: flats are angled
  # else, flats are all together
  angledFlats = False
  for i in flatFrames:
    obj = dataIndex['object'][i]
    pre = dataIndex['prefix'][i]

    if hasNumbers(obj) or hasNumbers(pre):
      angledFlats = True
      break
  if angledFlats:
    flatlist= {}
  else:
    flatlist = []

  for i,obj in enumerate(dataIndex['object']):
    if i in scienceFrames:
      if obj == planet:
        fullframelist.append(int(dataIndex['index'][i]))
        if planetExpTime == 0:
          planetExpTime = float(dataIndex['exptime'][i])

    if i in flatFrames:
      index = int(dataIndex['index'][i])
      if flatExpTime == 0:
        flatExpTime = float(dataIndex['exptime'][i])
      if not angledFlats:
        flatlist.append(index)
      else:
        # If angled Flats
        obj = dataIndex['object'][i]
        pre = dataIndex['prefix'][i]

        if hasNumbers(obj):
          angle = extractNumbers(obj)
        elif hasNumbers(pre):
          angle = extractNumbers(pre)
        else:
          angle = None

        if angle is not None:
          if angle in flatlist:
            flatlist[angle].append(index)
          else:
            flatlist[angle] = [index]

    if i in darkFrames:
      index = int(dataIndex['index'][i])
      expTime = float(dataIndex['exptime'][i])

      if expTime == planetExpTime:
        darklist.append(index)
      elif expTime == flatExpTime:
        darkflatlist.append(index)

  # Make sure lists are unique and sorted
  fullframelist = cleanList(fullframelist)
  darklist      = cleanList(darklist)
  darkflatlist  = cleanList(darkflatlist)
  if not angledFlats:
    flatlist = cleanList(flatlist)
  else:
    for key in flatlist:
      flatlist[key] = cleanList(flatlist[key])


  callist     = fullframelist[:5]
  darkcallist = darklist

  data['fullframelist'] = fullframelist
  data['flatlist'] = flatlist
  data['callist'] = callist
  data['darklist'] = darklist
  data['darkflatlist'] = darkflatlist
  data['darkcallist'] = darkcallist

  return data

# Use this to write summary to a .txt file
def writeSummary(path, output, 
                 headerKeys=['object'], overwrite='verify'
):
  # initialize output
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

  data_index = indexDir(path, headerKeys)

  # Set spacing of each value according to the longest
  longestValue = 0
  for key in data_index:
    lens = []
    for el in data_index[key]:
      lens.append(len(str(el)))

    longestValue = max(longestValue, max(lens))

  colLen = longestValue+1

  # Make summary header:
  hdStr = 'Filename:'.ljust(colLen)
  hdStr += 'Prefix:'.ljust(colLen)
  hdStr += 'Index:'.ljust(colLen)
  for hk in headerKeys:
    hdStr += (str(hk)+':').ljust(colLen)

  hdStr += endline
  printfn(hdStr)

  for i in range(len(data_index['filename'])):
    fileStr = ''
    for key in data_index:
      fileStr += str(data_index[key][i]).ljust(colLen)
    fileStr += endline
    printfn(fileStr)  

def mergeDB(*dbs, output='outdb.json'):
  '''
    Takes data from all dbs and puts it into output
    output -> Output json
  '''
  outDB = {}

  for db in dbs:
    with open(db,'r') as f:
      data = json.load(f)

      # Notify if repeated Key
      validateKeys(data, outDB)
      outDB.update(data)

  with open(output,'w') as f:
    json.dump(outDB, f, indent=2, sort_keys=True)
###

# Misc Helper Functions
def validateKeys(dict1, dict2):
  k1 = dict1.keys()
  k2 = dict2.keys()

  intersection = set(k1) & set(k2)
  if len(intersection) != 0:
    print('Warning: keys ' + str(intersection) +' found in common')

def hasNumbers(inputString):
  return any(char.isdigit() for char in inputString)

def extractNumbers(inputString):
  return int(''.join(filter(str.isdigit, inputString)))

def cleanList(l):
  l = list(set(l))
  l.sort()
  return l