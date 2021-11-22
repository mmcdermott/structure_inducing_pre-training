import pickle, argparse, os
from pathlib import Path
from tape.metrics import spearmanr, accuracy
try: from tape.metrics import sequence_accuracy   # Present in earlier versions of TAPE.
except: sequence_accuracy = accuracy              # In recent versions, handled as one function.

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir',
    type=str)
parser.add_argument('--split',
    type=str)
parser.add_argument('--metric_name')
args = parser.parse_args()

output_dir = Path(args.output_dir)
split = args.split
metric_name = args.metric_name

if metric_name == 'accuracy': metric_fn = accuracy
elif metric_name == 'sequence_accuracy': metric_fn = sequence_accuracy
elif metric_name == 'spearmanr': metric_fn = spearmanr

def depickle(filepath):
    with open(filepath, mode='rb') as f: return pickle.load(f)

for run in range(3):
    run = str(run)
    if run in os.listdir(output_dir):
        p = depickle(output_dir / run / f'{split}_preds.pkl')
        t = depickle(output_dir / run / f'{split}_targets.pkl')
        s = metric_fn(t, p)

        results_file = output_dir / run / f'{split}_results.txt'
        with open(results_file, 'w') as f:
            f.write(f'{metric_name} = {s}')
