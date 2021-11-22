import random, sys
from pathlib import Path

species_files           = Path('/crimea/graph_augmented_pt/raw_datasets/treeoflife/species_files/')
species_file_1840       = species_files / '1840_species.txt'
species_file_train      = species_files / 'train_species.txt'
species_file_val        = species_files / 'val_species.txt'
species_file_test       = species_files / 'test_species.txt'

if species_file_train.exists():
    print("Finished.")
    sys.exit(0)

N_TOTAL_SPECIES = 1840
N_TRAIN_SPECIES = int(N_TOTAL_SPECIES * 0.50)
N_VAL_SPECIES   = int(N_TOTAL_SPECIES * 0.25)
N_TEST_SPECIES  = int(N_TOTAL_SPECIES * 0.25)

with open(species_file_1840, 'r') as f:
    species_all = [ln.strip() for ln in f]

print(f'Train: {N_TRAIN_SPECIES}, Val: {N_VAL_SPECIES}, Test: {N_TEST_SPECIES}')
assert N_TRAIN_SPECIES + N_VAL_SPECIES + N_TEST_SPECIES == N_TOTAL_SPECIES
assert len(species_all) == N_TOTAL_SPECIES

orig_species_all = species_all.copy()
random.shuffle(species_all)
assert orig_species_all != species_all

def take(list, num):
    ret = []
    for _ in range(num):
        i = random.choice(range(len(list)))
        ret.append(list.pop(i))
    return ret

species_train   = take(species_all, N_TRAIN_SPECIES)
species_val     = take(species_all, N_VAL_SPECIES)
species_test    = take(species_all, N_TEST_SPECIES)

assert len(species_train) == N_TRAIN_SPECIES
assert len(species_val)   == N_VAL_SPECIES
assert len(species_test)  == N_TEST_SPECIES

assert set(species_train).union(set(species_val), set(species_test)) == set(orig_species_all), f'{len(set(species_train).union(set(species_val), set(species_test)))} {len(set(orig_species_all))}'

def write_to_file(sp_list, fn):
    with open(fn, 'w') as f:
        for sp in sp_list:
            f.write(f'{sp}\n')

write_to_file(species_train, species_file_train)
write_to_file(species_val, species_file_val)
write_to_file(species_test, species_file_test)
