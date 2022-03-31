import os
import re
import glob
import pickle
import pandas as pd
import functionality.helper as helper
from functionality.author_helper import reverse_complement

# TODO Comment and clean


def load_sequences_from_cutsites(inp_fn, new_targets, sample_size):
  pkl_file = os.path.dirname(inp_fn) + f'/cutsites_{sample_size}.pkl'
  if os.path.exists(pkl_file) and not new_targets:
    cutsites = helper.load_pickle(pkl_file)
    cutsites = cutsites.rename(columns={'Cutsite': 'target'})
  else:
    all_cutsites = load_genes_cutsites(inp_fn)
    # TODO - check with team, do we allow replicated elements or only unique?
    cutsites = all_cutsites.sample(n=sample_size).reset_index(drop=True)
    with open(pkl_file, 'wb') as f:
      pickle.dump(cutsites, f)
  cutsites['Location'] = cutsites['Location'].astype('int32')
  return cutsites


def load_intron_cutsites(inp_fn):
  batches = 0
  curr_seq_count = 0
  processed = 0

  pkl_file = inp_fn + '_cutsites.pkl'
  if os.path.exists(pkl_file):
    cutsites = helper.load_pickle(pkl_file)
    cutsites = cutsites.rename(columns={'Cutsite': 'target'})
    return cutsites

  with open(inp_fn, "r") as f:
    sequence, chrom = '', ''
    cutsites = []
    for line in f:
      if '>' in line:
        if sequence != '':
          processed += 1
          if processed % 100 == 0:
            print('Working on: ', processed)
          curr_cutsites = get_cutsites(chrom, sequence)
          cutsites.extend(curr_cutsites)
          curr_seq_count += len(curr_cutsites)
        chrom = line.strip().split(' ')[0]
        sequence = ''
        if curr_seq_count >= helper.BATCH_SIZE:
          batch_data = pd.DataFrame(cutsites, columns=['Cutsite', 'Chromosome'])
          pkl_file_batch = f'{inp_fn}_cutsites_{batches}.pkl'
          with open(pkl_file_batch, 'wb') as w_f:
            pickle.dump(batch_data, w_f)
          curr_seq_count = 0
          cutsites = []
          print(f'Saved batch {batches}')
          batches += 1
      else:
        sequence += line.strip()

    processed += 1  # Adding last item
    print('Last item inserted: ', processed)
    cutsites.extend(get_cutsites(chrom, sequence))

    print('Storing to file')
    all_data = pd.DataFrame(cutsites, columns=['Cutsite', 'Chromosome'])
    pkl_file_batch = f'{inp_fn}_cutsites_{batches}.pkl'
    with open(pkl_file_batch, 'wb') as w_f:
      pickle.dump(all_data, w_f)
    print('Gene cutsite complete')
    return cutsites


def load_genes_cutsites(inp_fn):
  pkl_file = os.path.dirname(inp_fn) + '/cutsites.pkl'
  if os.path.exists(pkl_file):
    cutsites = helper.load_pickle(pkl_file)
    cutsites = cutsites.rename(columns={'Cutsite': 'target'})
    return cutsites

  all_lines = open(inp_fn, "r").readlines()
  sequence, chrom = '', ''
  data, cutsites = [], []
  for line in all_lines:
    if '>' in line:
      if sequence != '':
        data.append([chrom, sequence])
        if len(data) % 100 == 0:
          print('Working on: ', len(data))
        cutsites.extend(get_cutsites(chrom, sequence))
      chrom = line.strip().split('|')[3]
      sequence = ''
    else:
      sequence += line.strip()

  data.append([chrom, sequence]) # Adding last item
  print('Last item inserted: ', len(data))
  cutsites.extend(get_cutsites(chrom, sequence))

  print('Storing to file')
  all_data = pd.DataFrame(cutsites, columns=['Cutsite', 'Chromosome', 'Location', 'Orientation'])
  with open(pkl_file, 'wb') as f:
    pickle.dump(all_data, f)
  print('Gene cutsite complete')
  return cutsites


def get_cutsites(chrom, sequence):
  all_cutsites = []
  for idx in range(len(sequence)):  # for each base in the sequence
    # this loop finishes only each of 5% of all found cutsites with 60-bp long sequences containing only ACGT
    seq = ''
    if sequence[idx: idx + 2] == 'CC':  # if on top strand find CC
      cutsite = idx + 6  # cut site of complementary GG is +6 away
      seq = sequence[cutsite - 30: cutsite + 30]  # get sequence 30bp L and R of cutsite
      seq = reverse_complement(seq)  # compute reverse strand (complimentary) to target with gRNA
      orientation = '-'
    if sequence[idx: idx + 2] == 'GG':  # if GG on top strand
      cutsite = idx - 4  # cut site is -4 away
      seq = sequence[cutsite - 30: cutsite + 30]  # get seq 30bp L and R of cutsite
      orientation = '+'
    if seq == '':
      continue
    if len(seq) != 60:
      continue

    # Sanitize input
    seq = seq.upper()
    if 'N' in seq:  # if N in collected sequence, return to start of for loop / skip rest
      continue
    if not re.match('^[ACGT]*$', seq):  # if there not only ACGT in seq, ^
      continue

    all_cutsites.append([chrom, seq])
  return all_cutsites


def find_cutsites_and_predict(inp_fn, use_file=''):
  # Loading Cutsites
  if use_file != '':
    all_data = helper.read_data(use_file + 'cutsites.pkl')
  else:
    # Calculating & Loading cutsites for all files
    cutsites = []
    for file in glob.glob(inp_fn + '*.fa'):
      file_name = os.path.basename(file)
      print('Working on: ' + file_name)
      data = open(file, "r").readlines()[1:]
      sequence = ''.join(data).replace('\n', '')
      cutsites.extend(get_cutsites(file_name, sequence))

    all_data = pd.DataFrame(cutsites, columns=['Chromosome', 'Cutsite'])
    with open(inp_fn + 'cutsites.pkl', 'wb') as f:
      pickle.dump(all_data, f)

  return
