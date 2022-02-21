import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

def hist(data, out_fn):
  # Data can be a list of lists
  # Each row corresponds to one group
  sns.distplot(data, kde = False)
  plt.ylabel('Counts')
  plt.savefig(out_fn)
  plt.close()
  return

def scatter(d, x_name, y_name, out_fn):
  # d should be a pandas.dataframe
  sns.jointplot(x = x_name, y = y_name, data = d)
  plt.savefig(out_fn)
  plt.close()
  return

def heatmap(d, out_fn):
  sns.heatmap(d, linewidths = 0.5)
  plt.savefig(out_fn)
  plt.close()
  return

def convert_filetype(fn, in_type, out_type):
  out_fn = fn.strip(in_type) + out_type
  coptions1 = ' -density 300 -trim '
  coptions2 = ' -quality 100 '
  subprocess.call('convert' + coptions1 + fn + coptions2 + out_fn, 
    shell = True)
  return