# Kaggle reproduction — TickNet-L v1

Open `Kaggle_TickNet_L_Midterm_Final.ipynb` in Kaggle and select **Run All**.
The notebook clones the exact `feature/giua-ki-model-l` branch and trains the
group's TickNet-L v1 with `--model l`; it does not implement another model.

## Kaggle inputs

Attach both datasets before running:

- [midd-32](https://www.kaggle.com/datasets/minhcuong2k5/midd-32), containing
  `Mid32/train|test/{bird,cat,dog,frog,horse}`.
- [midd-224](https://www.kaggle.com/datasets/minhcuong2k5/midd-224), containing
  `Mid224/train|test/{bird,cat,dog,frog,horse}`.

Enable a GPU and Internet in Kaggle Notebook settings.  The notebook first
validates five classes, 5,000 train images/class and 50 test images/class.  It
then trains 200 epochs on Mid32 and Mid224, stores `epochs.csv` for every epoch,
and exports final metrics, learning curves, confusion matrices and per-class
Precision/Recall/F1.

The completed evidence from the seed-42 run is in
[`../results/ticknet_l_midterm_seed42_20260928`](../results/ticknet_l_midterm_seed42_20260928/).
