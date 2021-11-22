import pickle

def zip_dicts(*dcts):
    for i in set(dcts[0]).intersection(*dcts[1:]): yield (i,) + tuple(d[i] for d in dcts)

def depickle(filepath):
    with open(filepath, mode='rb') as f: return pickle.load(f)
def enpickle(obj, filepath):
    with open(filepath, mode='wb') as f: return pickle.dump(obj, f, 4)

def find_in_parent_dirs(current_dir, target, max_parents_to_search = 8):
    """Finds a target file or directory that is the child of one of current_dir's ancestors."""
    found = False
    for i in range(max_parents_to_search):
        possible_path = current_dir / ('../' * i) / target
        if possible_path.exists():
            found = True
            return possible_path
    raise ValueError("Couldn't find target in parent dirs!")
