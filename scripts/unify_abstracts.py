import sys
sys.path.append('..')

from graph_augmented_pt.constants import *

import os, collections, random, time, json
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

"""
Writes mag abstracts.
Original files:     raw_datasets/ogb/mag_papers/mag_papers_{num}.txt
After Map step:     raw_datasets/ogb/abstracts/mag_abstracts_{num}.txt
After Reduce step:  raw_datasets/ogb/abstracts.txt
"""

if __name__ == '__main__':
    raw_data =                Path(RAW_DATASETS_DIR)
    ogb =                     raw_data / 'ogb'
    papers_path =             ogb / 'mag_papers'
    ogb_papers_path =         ogb / 'ogbn_mag_papers'
    reduced_ogb_papers_path = ogb / 'ogbn_mag_papers.txt'
    abstracts_path =          ogb / 'abstracts'
    reduced_abstracts_path =  ogb / 'abstracts.txt'
    mapping_path =            ogb / 'ogbn_mag' / 'mapping' / 'paper_entidx2name.csv'

    num_to_chunk_fn = dict()
    for fn in os.listdir(papers_path):
        if 'mag_papers_' not in fn: continue               # Ignore zip archives.
        num = int(fn.strip('mag_papers_').strip('.txt'))
        num_to_chunk_fn[num] = fn

    if not abstracts_path.exists():
        os.mkdir(abstracts_path)
        ent_mapping = pd.read_csv(mapping_path, sep=',', index_col='ent name')

        def construct_abstract(indexed_abstract, num, id):
            index_length = indexed_abstract['IndexLength']
            inverted_index = indexed_abstract['InvertedIndex']
            corrected_index = {}
            for k, v in inverted_index.items():
                for vi in v:
                    corrected_index[int(vi)] = k

            abstract = '"'                         # To deal with " and , in our abstracts csv, we need to surround the field in
                                                   # double quotes and represent all double-quotes by 2 consecutive double-quotes.
            for j in range(index_length):
                try:
                    cur_word = corrected_index[j]
                    cur_word = cur_word.replace('"', '""')
                    cur_word = cur_word.replace('\n', '')
                    abstract += cur_word
                    if j != index_length - 1: abstract += ' '
                except:
                    # print('Chunk', num)
                    # print('ID', id)
                    # print('Index length', index_length)
                    # print('Inv index', inverted_index)
                    # print('Malformed abstract')
                    # assert False
                    pass
                    # Seems like some really do just have bad indices..
            abstract += '"'

            return abstract


        def process_chunk(num):
            count_no_abstract = 0

            data_file = papers_path / num_to_chunk_fn[num]
            write_file = abstracts_path / f'mag_abstracts_{num}.txt'
            missing_file = abstracts_path / f'missing_{num}.txt'

            with open(data_file, 'r') as f:
                with open(write_file, 'w') as w:
                    if num == 0: w.write('ent idx,abstract\n')

                    for line in f:
                        d = json.loads(line.strip())
                        id = int(d['id'])

                        if id in ent_mapping.index:

                            try: indexed_abstract = d['indexed_abstract']   # Some don't have this.
                            except:
                                count_no_abstract += 1
                                continue

                            ent_idx = ent_mapping.loc[id]['ent idx']
                            abstract = construct_abstract(indexed_abstract, num, id)
                            w.write(f'{ent_idx},{abstract}')
                            w.write('\n')

            with open(missing_file, 'w') as m:
                m.write(str(count_no_abstract))

        with Pool(len(num_to_chunk_fn)) as p:
            p.map(process_chunk, range(len(num_to_chunk_fn)))

    if not ogb_papers_path.exists():
        os.mkdir(ogb_papers_path)
        ent_mapping = pd.read_csv(mapping_path, sep=',', index_col='ent name')

        def process_chunk(num):
            data_file = papers_path / num_to_chunk_fn[num]
            write_file = ogb_papers_path / f'ogbn_mag_papers_{num}.txt'

            with open(data_file, 'r') as f:
                with open(write_file, 'w') as w:
                    for line in f:
                        d = json.loads(line.strip())
                        id = int(d['id'])
                        if id in ent_mapping.index:
                            w.write(line)

        with Pool(len(num_to_chunk_fn)) as p:
            p.map(process_chunk, range(len(num_to_chunk_fn)))

    if not reduced_ogb_papers_path.exists():
        cat_cmd = 'cat'
        ordered_keys = sorted(num_to_chunk_fn.keys())
        for num in ordered_keys:
            cat_cmd += f' {ogb_papers_path / f"ogbn_mag_papers_{num}.txt"}'
        cat_cmd += f' > {reduced_ogb_papers_path}'
        os.system(cat_cmd)

    if not reduced_abstracts_path.exists():
        cat_cmd = 'cat'
        ordered_keys = sorted(num_to_chunk_fn.keys())
        total_missing = 0

        for num in ordered_keys:
            cat_cmd += f' {abstracts_path / f"mag_abstracts_{num}.txt"}'

            with open(abstracts_path / f'missing_{num}.txt', 'r') as f:
                total_missing += int(f.readline().strip())

        print(f'Missing {total_missing} abstracts out of 736389.')
        cat_cmd += f' > {reduced_abstracts_path}'
        os.system(cat_cmd)
        os.system(f'sed -i -e "s/\r//g" {reduced_abstracts_path}')  # Remove ^M newlines
