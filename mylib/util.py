# Utility library functions: IO, OS stuff

import sys, string, csv, os, fnmatch, datetime, subprocess

#########################################
# TIME
#########################################

class Timer:
  def __init__(self, total = -1, print_interval = 20000):
    # print_interval is in units of microseconds
    self.times = [datetime.datetime.now()]
    self.num = 0
    self.last_print = 0
    self.prev_num = 0
    self.total = int(total)
    if sys.stdout.isatty():   # if undirected stdout
      self.print_interval = print_interval
    else:   # else if stdout is directed int ofile
      self.print_interval = 5000000   # 5 seconds

  def progress_update(self):
    if self.last_print == 0:
      num_secs = (datetime.datetime.now() - self.times[0]).microseconds
    else:
      num_secs = (datetime.datetime.now() - self.last_print).microseconds

    passed_print_interval = (num_secs >= self.print_interval)
    is_done = (self.num == self.total)

    if passed_print_interval or is_done:
      if self.last_print != 0:
        sys.stdout.write("\033[F\033[F\033[F\033[F\033[F")
      self.last_print = datetime.datetime.now()
      if self.total != -1:
        print('\n\t\tPROGRESS %:', '{:5.2f}'.format(float(self.num * 100) / float(self.total)), ' : ', self.num, '/', self.total)
        print('\t\t', self.progress_bar(float(self.num * 100) / float(self.total)))
      else:
        print('\n\t\tTIMER:', self.num, 'iterations done after', str(datetime.datetime.now() - self.times[0]), '\n')
      rate = float(self.num - self.prev_num) / num_secs
      a = (self.times[1] - self.times[0]) / self.num
      if rate > 1:
        print('\t\t\tRate:', '{:5.2f}'.format(rate), 'iterations/second')
      else:
        print('\t\t\tAvg. Iteration Time:', a)
        
      if self.total != -1:
        if not is_done:
          print('\t\tTIMER ETA:', a * self.total - (datetime.datetime.now() - self.times[0]))
        if is_done:
          print('\t\tCompleted in:', datetime.datetime.now() - self.times[0])

      self.prev_num = self.num

      sys.stdout.flush()

    sys.stdout.flush()
    return

  def update(self, print_progress = True):
    if len(self.times) < 2:
      self.times.append(datetime.datetime.now())
    else:
      self.times[-1] = datetime.datetime.now()
    self.num += 1

    if print_progress:
      self.progress_update()
    return

  def progress_bar(self, pct):
    RESOLUTION = 40
    bar = '['
    pct = int(pct / (100.0 / RESOLUTION))
    bar += '\x1b[6;30;42m'
    for i in range(pct):
      bar += 'X'
    bar += '\x1b[0m'      
    for i in range(RESOLUTION - pct):
      bar += '-'
    bar += ']'
    return bar

# end Timer

def time_dec(func):
  def wrapper(*args, **kwargs):
    t = datetime.datetime.now()
    print('\n', t)
    res = func(*args, **kwargs)
    print(datetime.datetime.now())
    print('Completed in', datetime.datetime.now() - t, '\n')
    return res
  return wrapper

#########################################
# I/O
#########################################
def read_delimited_text(inp_fn, dlm, verbose = False):
  # Reads in a text file with the given delimiter, like '\t'
  if verbose:
    print('Reading in', inp_fn, '...')
  with open(inp_fn) as f:
    reader = csv.reader(f, delimiter = dlm)
    d = list(reader)
  return d

def dictread_delimited_text(inp_fn, dlm, verbose = False):
  if verbose:
    print('Reading in', inp_fn, '...')
  with open(inp_fn) as f:
    reader = csv.DictReader(f, delimiter = dlm)
    d = list(reader)
  return d

def write_delimited_text(out_fn, lists, dlm):
  # Assumes input as a 2D list, with lines / words
  with open(out_fn, 'w') as f:
    writ = csv.writer(f, delimiter = dlm)
    for line in lists:
      writ.writerow(line)
  return

#########################################
# OS
#########################################
def ensure_dir_exists(directory):
  # Guarantees that input dir exists
  if not os.path.exists(directory):
    try:
      os.makedirs(directory)
    except OSError:
      if not os.path.isdir(directory):
        raise
  return

def exists_empty_fn(fn):
  ensure_dir_exists(os.path.dirname(fn))
  f = open(fn, 'w')
  f.close()
  return

def get_fn(string):
  # In: Filename (possibly with directories)
  # Out: Filename without extensions or directories
  return string.split('/')[-1].split('.')[0]

def line_count(fn):
  try:
    ans = subprocess.check_output(['wc', '-l', fn.strip()])
    ans = int(ans.split()[0])
  except OSError as err:
    print('OS ERROR:', err)
  return ans

def ld_library_path(lib_path):
  # Libraries for locally installed packages usually 
  # need to be added to LD_LIBRARY_PATH for them to work
  ldlp = subprocess.check_output('echo $LD_LIBRARY_PATH', shell = True)
  if lib_path not in ldlp:
    subprocess.call('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:' + lib_path, shell = True)
  return

def shell_cp(fn, out_dir):
  subprocess.call(['cp', fn, out_dir])
  return

def shell_mv(inp_fn, out_fn):
  subprocess.call(['mv', inp_fn, out_fn])
  return

def num_files(inp_dir):
  ans = subprocess.check_output('ls ' + inp_dir + ' | wc -l', shell = True)
  return ans

def pdf_unite(inp_dir, nm = '_united', regex = ''):
  fns = []
  out_fn = nm + '_' + regex + '.pdf'
  for fn in os.listdir(inp_dir):
    if fnmatch.fnmatch(fn, '*' + regex + '*pdf') and fn != out_fn:
      fns.append(inp_dir + fn)
  print('PDF Uniting', len(fns), 'files into', inp_dir, out_fn)

  subprocess.call('pdfunite ' + ' '.join(fns) + ' ' + inp_dir + out_fn, shell = True)

  return

#########################################
# PROJECT STRUCTURE
#########################################

def get_prev_step(f, src_dir):
  # Assumes a folder of python functions
  #   utility files like _runall, _clean start with _
  #   processing steps begin with a_, b_, ...
  # Given a step file, returns the previous step filename

  steps = [x.replace('.py', '') for x in os.listdir(src_dir) if fnmatch.fnmatch(x, '?*_*.py')]
  sorted(steps)
  name = get_fn(f)
  ind = steps.index(name)
  if ind > 0:
    return steps[ind - 1]
  else:
    print('Error: No previous step for first script', f)
  return ''

def code_dependency(src_dir):
  # Returns the input/output dependency of a collection of 
  # python scripts
  #
  # Looks for the variable DEFAULT_INP_DIR 

  dep = dict()
  for fn in os.listdir(src_dir):
    if fnmatch.fnmatch(fn, '*.py'):
      with open(src_dir + fn) as f:
        for i, line in enumerate(f):
          words = line.split()
          if len(words) > 0 and words[0] == 'DEFAULT_INP_DIR':
            inp = ' '.join(words[2:])
            dep[fn] = inp
  with open(src_dir + '_dependencies.txt', 'w') as f:
    f.write('Script Name: Expected input folder\n\n')
    for k in sorted(dep.keys()):
      f.write(k + ': ' + dep[k] + '\n')
  return


#########################################
# ERROR CATCHING / DEBUGGING
#########################################

# Decorate a function with this
# to not stop running program if function errors out
def catch_function_fail_dec(func):
  def wrapper(*args, **kwargs):
    try:
      func(*args, **kwargs)
    except:
      print('Skipping', func.__name__)
    return  
  return wrapper

def check_variable_exists(var_name):
  return var_name in vars() or var_name in globals()
