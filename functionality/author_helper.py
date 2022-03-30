# Their functionality re-used
import os
import pickle
import datetime
import sys

import autograd.numpy as np
import autograd.numpy.random as npr
import colorama
from past.builtins import xrange
from scipy.stats import pearsonr


class Timer:
  def __init__(self, total=-1, print_interval=20000):
    # print_interval is in units of microseconds
    colorama.init()
    self.times = [datetime.datetime.now()]
    self.num = 0
    self.last_print = 0
    self.prev_num = 0
    self.total = int(total)
    if sys.stdout.isatty():  # if undirected stdout
      self.print_interval = print_interval
    else:  # else if stdout is directed int ofile
      self.print_interval = 5000000  # 5 seconds

  def progress_update(self):
    if self.last_print == 0:
      num_secs = (datetime.datetime.now() - self.times[0]).microseconds
    else:
      num_secs = (datetime.datetime.now() - self.last_print).microseconds

    passed_print_interval = (num_secs >= self.print_interval)
    is_done = (self.num == self.total)

    if passed_print_interval or is_done:
      if self.last_print != 0:
        sys.stdout.write("\033[A\033[A\033[A\033[A\033[A")
      self.last_print = datetime.datetime.now()
      if self.total != -1:
        print('\n\t\tPROGRESS %:', '{:5.2f}'.format(float(self.num * 100) / float(self.total)), ' : ', self.num, '/',
              self.total)
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

  def update(self, print_progress=True):
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


def ensure_dir_exists(directory):
  # Guarantees that input dir exists
  if not os.path.exists(directory):
    try:
      os.makedirs(directory)
    except OSError:
      if not os.path.isdir(directory):
        raise
  return


def reverse_complement(dna):
  lib = {'A': 'T', 'G': 'C', 'C': 'G', 'T': 'A', 'N': 'N', 'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y', 'Y': 'R'}
  new_dna = ''
  dna = dna.upper()
  for c in dna:
    if c in lib:
      new_dna += lib[c]
    else:
      new_dna += c
  new_dna = new_dna[::-1]
  return new_dna


def sigmoid(x):
  return 0.5 * (np.tanh(x) + 1.0)


def exponential_decay(step_size):
  if step_size > 0.001:
    step_size *= 0.999
  return step_size


##
# Sequence featurization
##
def get_gc_frac(seq):
  return (seq.count('C') + seq.count('G')) / len(seq)


def find_microhomologies(left, right):  # for a given pair of resected left and right strands, equally long
  start_idx = max(len(right) - len(left), 0)  # TAGATT - TATAGG = 0
  mhs = []
  mh = [start_idx]  # [0]
  for idx in range(min(len(right), len(left))):  # for each base in the overhangs
    # {--left[idx] == right[idx]--} = {--left[idx] complementary to reverse_right[star_idx+idx]--}
    if left[idx] == right[start_idx + idx]:
      # TAGATT    2 MHs
      # || |
      # ATATCC
      # 012345
      # gt pos    123456
      # MH 1 del outcome: GTGCTCTTAACTTTCACTTTATATAGGGTTAATAAATGGGAATTTATAT
      # MH 2 del outcome: GTGCTCTTAACTTTCACTTTATAGAGGGTTAATAAATGGGAATTTATAT
      mh.append(start_idx + idx + 1)
    else:
      mhs.append(mh)
      mh = [start_idx + idx + 1]
  mhs.append(mh)
  return mhs  # 1234567890123456789012345678901234567890123456789012345
  #                              0123456789012345678901234567890123456789012345678901234
  # MH 1 outcome:                GTGCTCTTAACTTTCACTTTATA------TAGGGTTAATAAATGGGAATTTATAT, gt pos 2, del len 6
  # MH 2 outcome:                GTGCTCTTAACTTTCACTTTATAGA------GGGTTAATAAATGGGAATTTATAT, gt pos 4, del len 6


##
# Setup environment
##
def alphabetize(num):
  assert num < 26 ** 3, 'num bigger than 17576'
  mapper = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm',
            13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y',
            25: 'z'}
  hundreds = int(num / (26 * 26)) % 26
  tens = int(num / 26) % 26
  ones = num % 26
  return ''.join([mapper[hundreds], mapper[tens], mapper[ones]])


def count_num_folders(out_dir):
  for fold in os.listdir(out_dir):
    assert os.path.isdir(out_dir + fold), 'Not a folder!'
  return len(os.listdir(out_dir))


def print_and_log(text, log_fn):
  parsed_text = datetime.datetime.now().strftime("%H:%M:%S") + ' - ' + text
  with open(log_fn, 'a') as f:
    f.write(parsed_text + '\n')
  print(parsed_text)
  return


def save_train_test_names(train_nms, test_nms, out_dir):
  with open(out_dir + 'train_exps.csv', 'w') as f:
    f.write(','.join(['Exp']) + '\n')
    for i in xrange(len(train_nms)):
      f.write(','.join([train_nms[i]]) + '\n')
  with open(out_dir + 'test_exps.csv', 'w') as f:
    f.write(','.join(['Exp']) + '\n')
    for i in xrange(len(test_nms)):
      f.write(','.join([test_nms[i]]) + '\n')
  return


def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
  """Build a list of (weights, biases) tuples,
     one for each layer in the net."""
  return [(scale * rs.randn(m, n),  # weight matrix
           scale * rs.randn(n))  # bias vector
          for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]


def rsq(nn_params, nn2_params, inp, obs, obs2, del_lens, names=None):
  if names is not None:
    rsqs1, rsqs2 = {}, {}
  else:
    rsqs1, rsqs2 = [], []
  for idx in range(len(inp)):
    ##
    # MH-based deletion frequencies
    ##
    mh_scores = nn_match_score_function(nn_params, inp[idx])
    Js = np.array(del_lens[idx])
    unnormalized_fq = np.exp(mh_scores - 0.25 * Js)

    # Add MH-less contribution at full MH deletion lengths
    mh_vector = inp[idx].T[0]
    mhfull_contribution = np.zeros(mh_vector.shape)
    for jdx in range(len(mh_vector)):
      if del_lens[idx][jdx] == mh_vector[jdx]:
        dl = del_lens[idx][jdx]
        mhless_score = nn_match_score_function(nn2_params, np.array(dl))
        mhless_score = np.exp(mhless_score - 0.25 * dl)
        mask = np.concatenate([np.zeros(jdx, ), np.ones(1, ) * mhless_score, np.zeros(len(mh_vector) - jdx - 1, )])
        mhfull_contribution = mhfull_contribution + mask
    unnormalized_fq = unnormalized_fq + mhfull_contribution
    normalized_fq = np.divide(unnormalized_fq, np.sum(unnormalized_fq))

    rsq1 = pearsonr(normalized_fq, obs[idx])[0] ** 2
    if names is not None:
      rsqs1[names[idx]] = rsq1
    else:
      rsqs1.append(rsq1)

    ##
    # Deletion length frequencies, only up to 28
    #   (Restricts training to library data, else 27 bp.)
    ##
    dls = np.arange(1, 28 + 1)
    dls = dls.reshape(28, 1)
    nn2_scores = nn_match_score_function(nn2_params, dls)
    unnormalized_nn2 = np.exp(nn2_scores - 0.25 * np.arange(1, 28 + 1))

    # iterate through del_lens vector, adding mh_scores (already computed above) to the correct index
    mh_contribution = np.zeros(28, )
    for jdx in range(len(Js)):
      dl = Js[jdx]
      if dl > 28:
        break
      mhs = np.exp(mh_scores[jdx] - 0.25 * dl)
      mask = np.concatenate([np.zeros(dl - 1, ), np.ones(1, ) * mhs, np.zeros(28 - (dl - 1) - 1, )])
      mh_contribution = mh_contribution + mask
    unnormalized_nn2 = unnormalized_nn2 + mh_contribution
    normalized_fq = np.divide(unnormalized_nn2, np.sum(unnormalized_nn2))

    rsq2 = pearsonr(normalized_fq, obs2[idx])[0] ** 2
    if names is not None:
      rsqs2[names[idx]] = rsq2
    else:
      rsqs2.append(rsq2)

  return rsqs1, rsqs2


##
# Plotting and Writing
##
def save_parameters(nn_params, nn2_params, out_dir_params, letters):
  pickle.dump(nn_params, open(out_dir_params + letters + '_nn.pkl', 'wb'))
  pickle.dump(nn2_params, open(out_dir_params + letters + '_nn2.pkl', 'wb'))
  return


def nn_match_score_function(params, inputs):
  # """Params is a list of (weights, bias) tuples.
  #    inputs is an (N x D) matrix."""
  inpW, inpb = params[0]
  # inputs = swish(np.dot(inputs, inpW) + inpb)
  inputs = sigmoid(np.dot(inputs, inpW) + inpb)
  # inputs = leaky_relu(np.dot(inputs, inpW) + inpb)
  for W, b in params[1:-1]:
    outputs = np.dot(inputs, W) + b
    # inputs = swish(outputs)
    inputs = sigmoid(outputs)
    # inputs = logsigmoid(outputs)
    # inputs = leaky_relu(outputs)
  outW, outb = params[-1]
  outputs = np.dot(inputs, outW) + outb
  return outputs.flatten()
