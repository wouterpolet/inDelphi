# Config parameters
# imported by src/_config
import os

NAMES = ['GH', 'IJ']

data_dir = os.path.dirname(__file__)

GRNAS = open(os.path.join(data_dir, 'grna-libA.txt')).read().split()
TARGETS = open(os.path.join(data_dir, 'targets-libA.txt')).read().split()
# TARGETS_EXPWT = open(data_dir + 'targets_expwt.txt').read().split()
OLIGO_NAMES = open(os.path.join(data_dir, 'names-libA.txt')).read().split()

def add_mer(inp):
  new = []
  for mer in inp:
    for nt in ['A', 'C', 'G', 'T']:
      new.append(mer + nt)
  return new

threemers = add_mer(add_mer(['A', 'C', 'G', 'T']))