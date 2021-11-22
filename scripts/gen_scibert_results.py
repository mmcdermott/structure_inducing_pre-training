import os, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir',
        type=str)
parser.add_argument('--do_downsample',
        action='store_true')
parser.add_argument('--do_patience',
        action='store_true')
parser.add_argument('--hf_model_name',
        type=str, default='allenai/scibert_scivocab_uncased')
args = parser.parse_args()
output_dir = Path(args.output_dir)
do_downsample = args.do_downsample
do_patience = args.do_patience
hf_model_name = args.hf_model_name

if do_downsample and do_patience:
    with open(output_dir / f'scibert_results.csv', 'w') as f_write:
        f_write.write('frac,lr,da1,da2,da3,df1,df2,df3,ta1,ta2,ta3,tf1,tf2,tf3\n')
        for frac in ['0.001', '0.01', '0.1', '0.5']:
            for lr in ['5e-6', '1e-5', '2e-5', '5e-5']:
                da, df, ta, tf = [], [], [], []
                for run in [str(x) for x in (0,1,2)]:
                    with open(output_dir /  lr / frac / run / 'dev_results.txt') as f_dev:
                        dev_acc = f_dev.readline().strip().split()[-1]
                        dev_f1 = f_dev.readline().strip().split()[-1]
                    with open(output_dir / lr / frac / run / 'test_results.txt') as f_test:
                        test_acc = f_test.readline().strip().split()[-1]
                        test_f1 = f_test.readline().strip().split()[-1]
                    da.append(dev_acc)
                    df.append(dev_f1)
                    ta.append(test_acc)
                    tf.append(test_f1)
                to_write = [frac] + [lr] + da + df + ta + tf
                for x in to_write[:-1]:
                    f_write.write(x + ',')
                f_write.write(to_write[-1] + '\n')


elif do_downsample and not do_patience:
    for frac in ['0.001', '0.01', '0.1', '0.5']:
        with open(output_dir / f'scibert_results_{frac}.csv', 'w') as f_write:
            f_write.write('epochs,lr,da1,da2,da3,df1,df2,df3,ta1,ta2,ta3,tf1,tf2,tf3\n')

            for epoch in [str(x) for x in (2,3,4,5)]:
                for lr in ['5e-6', '1e-5', '2e-5', '5e-5']:
                    da, df, ta, tf = [], [], [], []
                    for run in [str(x) for x in (0,1,2)]:
                        with open(output_dir / (epoch + '_epochs') / lr / frac / run / 'dev_results.txt') as f_dev:
                            dev_acc = f_dev.readline().strip().split()[-1]
                            dev_f1 = f_dev.readline().strip().split()[-1]
                        with open(output_dir / (epoch + '_epochs') / lr / frac / run / 'test_results.txt') as f_test:
                            test_acc = f_test.readline().strip().split()[-1]
                            test_f1 = f_test.readline().strip().split()[-1]
                        da.append(dev_acc)
                        df.append(dev_f1)
                        ta.append(test_acc)
                        tf.append(test_f1)
                    to_write = [epoch] + [lr] + da + df + ta + tf
                    for x in to_write[:-1]:
                        f_write.write(x + ',')
                    f_write.write(to_write[-1] + '\n')

else:
    with open(output_dir / 'scibert_results.csv', 'w') as f_write:
        f_write.write('epochs,lr,da1,da2,da3,df1,df2,df3,ta1,ta2,ta3,tf1,tf2,tf3\n')
        epoch_list = [str(x) for x in (10,15,20,25)] if hf_model_name == 'allenai/cs_roberta_base' else [str(x) for x in (2,3,4,5)] 
        for epoch in epoch_list:
            for lr in ['5e-6', '1e-5', '2e-5', '5e-5']:
                da, df, ta, tf = [], [], [], []
                for run in [str(x) for x in (0,1,2)]:
                    with open(output_dir / (epoch + '_epochs') / lr / run / 'dev_results.txt') as f_dev:
                        dev_acc = f_dev.readline().strip().split()[-1]
                        dev_f1 = f_dev.readline().strip().split()[-1]
                    with open(output_dir / (epoch + '_epochs') / lr / run / 'test_results.txt') as f_test:
                        test_acc = f_test.readline().strip().split()[-1]
                        test_f1 = f_test.readline().strip().split()[-1]
                    da.append(dev_acc)
                    df.append(dev_f1)
                    ta.append(test_acc)
                    tf.append(test_f1)
                to_write = [epoch] + [lr] + da + df + ta + tf
                for x in to_write[:-1]:
                    f_write.write(x + ',')
                f_write.write(to_write[-1] + '\n')
