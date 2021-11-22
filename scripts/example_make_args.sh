for RUN in rel_0.1_pos2_neg2 rel_0.5_pos2_neg2 rel_0.5_pos2_neg40
do
    for TASK in citation_intent sci-cite mag
    do 
        # mkdir /crimea/graph_augmented_pt/runs/jan3Mag/multi/$RUN/0/batches/130000/finetune/$TASK
        # rm -r /crimea/graph_augmented_pt/runs/jan3Mag/multi/$RUN/0/batches/130000/finetune/$TASK/*
        python copy_scibert_args.py --template=/crimea/graph_augmented_pt/runs/jan3Mag/arg_templates/$TASK --output_dir=/crimea/graph_augmented_pt/runs/jan3Mag/multi/$RUN/0/batches/130000/finetune/$TASK
        sleep 1
    done
done