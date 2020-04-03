LRS="0.001 0.0001 0.00001"
LAYERS="3 2 1"
for lr in $LRS; do
    for layer in $LAYERS; do
        echo "$lr $layer"
        plumr dataset_preprocessing/viggo_dialog_planner.py run train --hp-LR $lr --hp-L $layer --gpu 3
        plumr dataset_preprocessing/viggo_dialog_planner.py run train_tied --hp-LR $lr --hp-L $layer --gpu 3
    done
done
