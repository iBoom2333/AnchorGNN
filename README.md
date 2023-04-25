# Billion-Scale Bipartite Graph Embedding: A Global-Local Induced Approach

## Requirements

```
python 3.8
torch >= 1.8.0
numpy
tqdm
```

## Datasets

1. Yelp and Amazon-Book are obtained from [LightGCN](https://github.com/gusye1234/LightGCN-PyTorch/tree/master/data). We have preprocessed and provided both datasets in CSR format under the default dataset directory `./dataset`.

2. The remaining datasets are obtained from [GEBEp](https://renchi.ac.cn/datasets/). Please download and extract them under your specified directory `GEBE_DATA_ROOT_DIR`. 

## Preprocessing

<!-- We preprocess and split the original graph into training and test sets as follows. -->

**1. Recommendation**

```shell
# generate train/test splits
python -u ./code/preprocess_recommendation_convert_2_csr.py --dataset DATASET --edge_root_dir GEBE_DATA_ROOT_DIR --output_root_dir ./dataset
```

**2. Link prediction**

```shell
# generate train/test splits
python -u ./code/preprocess_lp_convert_2_csr.py --dataset DATASET --edge_root_dir GEBE_DATA_ROOT_DIR --output_root_dir ./dataset

# generate link prediction samples
python -u ./code/generate_lp_samples.py --dataset DATASET --edge_root_dir ./dataset --output_root_dir ./dataset
```


## Usage

The parameter settings for all datasets are provided in [run.sh](./run.sh). Several running examples are as follows.

**1. Recommendation**

Yelp (small dataset)
```shell
python -u train.py --dataset Yelp --gpu 0 --reg 0.005 --data_root_dir ./dataset
```

MAG (large dataset)
```shell
python -u train_large_graph.py --dataset MAG --gpu 0 --reg 0.002 --test_batch_interval 200 --test_batch 100 --data_root_dir ./dataset
```

**2. Link prediction**

Pinterest (small dataset)
```shell
# train AnchorGNN
python -u train.py --dataset Pinterest --gpu 0 --reg 0.005 --save_model --data_root_dir ./dataset --model_root_dir ./models

# train LP classifier
python -u train_lp.py --model_file ./models/Pinterest/model.epoch.15.pt --batch_testing --test_batch 100000
```

Orkut (large dataset)
```shell
# train AnchorGNN
python -u train_large_graph.py --dataset Orkut --reg 0.002 --test_batch_interval 1000 --test_batch 40 --save_model --data_root_dir ./dataset --model_root_dir ./models

# train LP classifier
python -u train_lp.py --model_file ./models/Orkut/model.epoch.0.batch.3000.pt --batch_testing --test_batch 100000
```


## FAQ

1. If OOM occurs when training the link prediction classifier, adjust `test_batch` and `train_ratio` to fit a smaller RAM.
