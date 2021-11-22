export CUDA_VISIBLE_DEVICES=0

for TASK in citation_intent sci-cite mag
do 
    for RUN in rel_0.1_pos2_neg2 rel_0.5_pos2_neg2 rel_0.5_pos2_neg40
    do
        for EPOCH in 2 3 4 5
        do
            for LR in 1e-5
            do 
                python run_finetuning.py --do_load_from_dir --output_dir=/crimea/graph_augmented_pt/runs/jan3Mag/multi/$RUN/0/batches/130000/finetune/$TASK/${EPOCH}_epochs/$LR
            done
        done
    done
done
