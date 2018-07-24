import dbmanager as dbm
import os

# datapath = os.path.expanduser('~')+"/documents/science/spectroscopy/2016oct16/raw"
# prefix = "spec_"
# hoi  = ['object','filter','exptime','airmass']
# dbm.collectData(datapath,prefix,'2016oct16', headers=hoi,targObject="--")

dbm.addToDb('obsdb.json')
