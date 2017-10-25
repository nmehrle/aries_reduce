""" 
  Contains tools to read from json file and easily modify the database there
"""
import json as _json
from collections import MutableMapping



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
def addToDb():
  obsdb = json('obsdb.json')
  
  date = '20161016'
  data = {}

  import os
  _aphome = os.path.expanduser('~')[1:].replace('/','_')

  data['planet']     = 'Upsilon Andromedae b'
  data['prefix']     = 'spec_'
  data['framelist']  = range(73,3824+1)
  data['flatlist']   = range(13,37+1) 
  data['darklist']   = range(3825, 3875+1)
  data['callist']    = range(3876, 3926+1)
  data['datadir']    = '20161016/'
  data['ap_suffix']  = 'database/ap_' + _aphome + '_Documents_science_spectroscopy_20161016_proc'
  data['n_aperture'] = 5
  data['filter']     = 'OPEN5'
  data['calnod']     = False

  obsdb[date] = data
  obsdb.write()
  print("written!")
