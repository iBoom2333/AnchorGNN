# 1. Use train.py to train general graphs.
# recommendation
python -u train.py --dataset Yelp --gpu 0 --reg 0.005
python -u train.py --dataset Movielens --gpu 0 --reg 0.003
python -u train.py --dataset Lastfm --gpu 0 --reg 0.005
python -u train.py --dataset Netflix --gpu 0 --reg 0.001 --full_batch 0 --batch_per_epoch 10000 --max_epoch 60
# link prediction (LP)
# Note: There are two steps in LP task: (1) model training, and (2) LP classifier training. 
#       We have to save models for LP classifier training.
python -u train.py --dataset Wikipedia --gpu 0 --reg 0.005 --save_model
python -u train.py --dataset Pinterest --gpu 0 --reg 0.005 --save_model
python -u train.py --dataset Amazon-Book --gpu 0 --reg 0.005 --save_model
python -u train.py --dataset MIND --gpu 0 --reg 0.005 --save_model

# 2. Use train_large_graph.py to train large-scale graphs.
# recommendation
python -u train_large_graph.py --dataset MAG --gpu 0 --reg 0.002 --test_batch_interval 200 --test_batch 100
# link prediction (LP)
# Note: For large-scale graphs like Orkut, one can directly save model by setting `--do_test 0 --save_model_wo_test`,
#       and conduct link prediction using the saved model.
python -u train_large_graph.py --dataset Orkut --gpu 0 --reg 0.002 --test_batch_interval 500 --test_batch 40 --save_model
