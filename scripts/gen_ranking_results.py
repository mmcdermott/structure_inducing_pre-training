import os, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir',
        type=str)
parser.add_argument('--do_downsample',
        action='store_true')
parser.add_argument('--name',
        type=str, default='')
args = parser.parse_args()
output_dir = Path(args.output_dir)

if args.do_downsample:
    with open(output_dir / 'ir_results.csv', 'w') as f_write:
        f_write.write('frac,l1,l2,l3,n1,n2,n3,a1,a2,a3,m1,m2,m3\n')

        for frac in ['0.0001', '0.001', '0.01', '0.05', '0.1', '0.25', '0.5', '0.75', '1.0']:
            l, n, a, m = [], [], [], []
            for run in [str(x) for x in (0,1,2)]:
                with open(output_dir / frac / run / 'best_epoch' / 'all_eval_results.txt') as f_res:
                    lrap = f_res.readline().strip().split()[2]
                    ndcg = f_res.readline().strip().split()[2]
                    ap = f_res.readline().strip().split()[2]
                    mrr = f_res.readline().strip().split()[2]
                l.append(lrap)
                n.append(ndcg)
                a.append(ap)
                m.append(mrr)
            to_write = [frac] + l + n + a + m
            for x in to_write[:-1]:
                f_write.write(x + ',')
            f_write.write(to_write[-1] + '\n')

else:
    with open(output_dir / 'ir_results.csv', 'w') as f_write:
        f_write.write('lrap,ndcg,ap,mrr,p1,p5,p10,p25,p100,r1,r5,r10,r25,r100,f1,f5,f10,f25,f100\n')
        with open(output_dir / 'all_eval_results.txt') as f_res:
            to_write = [ln.strip().split()[2] for ln in f_res]
            for x in to_write[:-1]:
                f_write.write(x + ',')
            f_write.write(to_write[-1] + '\n')
