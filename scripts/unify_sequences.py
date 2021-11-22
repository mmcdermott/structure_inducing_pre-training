import sys
sys.path.append('..')

from graph_augmented_pt.constants import *

import os, collections, random, time
import pandas as pd
from Bio import SeqIO
from pathlib import Path
from multiprocessing import Pool

"""
Splits all-species .fa file into per-species files.
Original file:     raw_datasets/string/protein_sequences.v11.0.fa
New files:         raw_datasets/string/species/{species_id}.fa
"""

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

NUM_CHUNKS = 64

if __name__ == '__main__':
    raw_data = Path(RAW_DATASETS_DIR)
    tree_path =       raw_data / 'treeoflife'
    ixn_path =        raw_data / 'treeoflife/treeoflife.interactomes'
    str_path =        raw_data / 'string'
    species_path =    raw_data / 'string/species'
    chunks_path =     raw_data / 'string/chunks'
    all_seq_file =    raw_data / 'string/protein.sequences.v10.5.fa'

    for path in [ixn_path, str_path, all_seq_file]:
        assert path.exists()

    # THIS DOES NOT WORK, TOO MANY REQUESTS -> GETS REJECTED
    # with cd(str(raw_data / 'string/new_species')):
    #     species_list = pd.read_csv(tree_path / 'treeoflife.species.tsv', sep='\t')['Species_ID']
    #     print(len(species_list))
    #     for species_id in species_list:
    #         time.sleep(1)
    #         os.system(f'wget https://stringdb-static.org/download/protein.sequences.v11.0/{species_id}.protein.sequences.v11.0.fa.gz')

    # Split the original file into evenly-sized chunks using 'split'
    if not chunks_path.exists():
        os.mkdir(chunks_path)
        SPLIT_CMD = f'split -d -n {NUM_CHUNKS} {all_seq_file} {str_path / "sequences_"}'
        MOVE_CMD = f'mv {str_path / "sequences_*"} {chunks_path}'

        for cmd in [SPLIT_CMD, MOVE_CMD]:
            os.system(cmd)


        # We might have cut off the protein sequence at end of each file i.
        # Clean these up by removing beginning lines of file i+1 and appending those lines to file i.
        num_to_filename = dict()
        for fn in os.listdir(chunks_path):
            num = int(fn.strip('sequences_'))
            num_to_filename[num] = fn

        for i in range(NUM_CHUNKS-1):
            fn = chunks_path / num_to_filename[i]
            fn_next = chunks_path / num_to_filename[i+1]

            with open(fn_next, 'r') as f:
                lns = []
                while True:
                    line = f.readline()
                    if line[0] == '>': break
                    lns.append(line)

            with open(fn, 'a') as f:
                for line in lns:
                    f.write(line)

                if len(lns) > 0:
                    print('Fixing file', i)
                    REMOVE_LINE_CMD = f'sed -i 1,{len(lns)}d {fn_next}'
                    os.system(REMOVE_LINE_CMD)


    # Assign workers to process each chunk and append protein sequences to the relevant species files.
    if not species_path.exists():
        os.mkdir(species_path)

        # species_list = pd.read_csv(tree_path / 'treeoflife.species.tsv', sep='\t')['Species_ID']
        # print(len(species_list))

        num_to_chunk_fn = dict()
        for fn in os.listdir(chunks_path):
            num = int(fn.strip('sequences_'))
            num_to_chunk_fn[num] = fn

        def process_chunk(num):
            data_file = chunks_path / num_to_chunk_fn[num]
            species_to_records = {}
            for record in SeqIO.parse(str(data_file), 'fasta'):
                s = int(record.name.split('.')[0])
                if s in species_to_records:
                    species_to_records[s].append(record)
                else:
                    species_to_records[s] = [record]

            time.sleep(random.randrange(1,NUM_CHUNKS))  # Probably works without locking, so let's add this just in case.

            for s, records_list in species_to_records.items():
                if len(records_list) == 0: continue
                s_path = species_path / f'{s}.fa'
                if not s_path.exists():
                    f = open(s_path, 'x')
                    f.close()
                with open(s_path, 'a') as output_handle:
                    SeqIO.write(records_list, output_handle, 'fasta')

        with Pool(NUM_CHUNKS) as p:
            p.map(process_chunk, range(NUM_CHUNKS))


    # Check that all the relevant species have files.
    species_df = pd.read_csv(tree_path / 'treeoflife.species.tsv', sep='\t', index_col='Official_NCBI_name')
    string_df = pd.read_csv(str_path / 'species.v11.0.txt', sep='\t', index_col='official_name_NCBI')
    for sp_name in species_df.index:
        sp_id = species_df.loc[sp_name]['Species_ID']
        if not f"{sp_id}.fa" in os.listdir(species_path):
    #         alias_id = string_df.loc[sp_name]['## taxon_id']
            print(f"{sp_id} not found")
    # species = set()
    # data_file = all_seq_file
    # for record in SeqIO.parse(str(data_file), 'fasta'):
    #     s = int(record.name.split('.')[0])
    #     if s in species_list:
    #         print(s)
    #         species.add(s)
    # print(len(species))
