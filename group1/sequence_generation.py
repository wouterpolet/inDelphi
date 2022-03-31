import os
import re
import glob
import pickle
import helper
import pandas as pd

BATCH_SIZE = 20000000

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


def load_genes_cutsites(inp_fn):
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
        if curr_seq_count >= BATCH_SIZE:
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

    processed += 1 # Adding last item
    print('Last item inserted: ', processed)
    cutsites.extend(get_cutsites(chrom, sequence))

    print('Storing to file')
    all_data = pd.DataFrame(cutsites, columns=['Cutsite', 'Chromosome'])
    pkl_file_batch = f'{inp_fn}_cutsites_{batches}.pkl'
    with open(pkl_file_batch, 'wb') as w_f:
      pickle.dump(all_data, w_f)
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
      seq = helper.reverse_complement(seq)  # compute reverse strand (complimentary) to target with gRNA
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

  #
  #
  # # Calculate statistics on df, saving to alldf_dict
  # # Deletion positions
  # d = defaultdict(list)            # dictionary with values as list
  # dd_shuffled = defaultdict(list)
  # num_flushed = 0
  # all_data = open(inp_fn, "r").readlines()[1:]
  # sequence = ''.join(all_data).replace('\n', '')
  #
  # bulk_predict(sequence, d)
  # generate dicts (dd and dd_shuffled) for info on each cutsite seq found in the sequence and for a shuffled cutsite sequence
  # dd, dd_shuffled, num_flushed = maybe_flush(dd, dd_shuffled, data_nm, split, num_flushed)  # likely no flush
  #                               maybe flush out the dict contents into csv and return empty dicts

  # maybe_flush(dd, dd_shuffled, data_nm, split, num_flushed, force = True)   # will flush due to forced flushing
  return


# Run for exons and introns database
if __name__ == '__main__':
  # inp_file = os.path.dirname(__file__) + '/../in/exons'
  inp_file = os.path.dirname(__file__) + '/../in/introns'
  load_genes_cutsites(inp_file)
