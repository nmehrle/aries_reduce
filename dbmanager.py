""" 
  Contains tools to read from json file and easily modify the database there
"""
import json as _json
from collections import MutableMapping



"""
json loads a json database into memory as a dict
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

