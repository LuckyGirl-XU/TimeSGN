# TimeSGN

This repo is the open-sourced code for our TimeSGN 

# Requirements

- python >= 3.6.13
- pytorch >= 1.8.1
- pandas >= 1.1.5
- numpy >= 1.19.5
- dgl >= 0.6.1
- pyyaml >= 5.4.1
- tqdm >= 4.61.0
- pybind11 >= 2.6.2
- g++ >= 7.5.0

# Datasets
The datasets are used in this paper:
- [Wikipedia]( http://snap.stanford.edu/jodie/wikipedia.csv)
- [Reddit](http://snap.stanford.edu/jodie/reddit.csv)
- [MOOC](http://snap.stanford.edu/jodie/mooc.csv)
- [LastFM](http://snap.stanford.edu/jodie/lastfm.csv)
- [GDELT_ST](https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edges.csv)
- [GDELT](https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edges.csv,https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edge_features.pt)
- [MAG](https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/edges.csv)

# Preprocessing
  ## Temporal sampler initialization

    python utils/setup.py build_ext --inplace

  ## Data preprocessing
  - We provide preprocessed data samples, which can be downloaded [here](https://drive.google.com/drive/folders/1Nr9bL6rEkioR9gzftEPP3fk4J7pLodLs?usp=sharing)
  - Raw data need to be processing:
```
  python utils/gen_graph.py --data WIKI
```



# Run Examples
  ## Single GPU training: Link prediction task and Link ranking task
  
  
      - For transductive link prediction
          python train.py --data WIKI --config ./config/TimeSGN.yml --gpu 0 --DTMP
      - For inductive link Ranking
          python train.py --data WIKI --config ./config/TimeSGN.yml --gpu 0 --eval_can_samples 100 --DTMP --use_inductive 

 
  ## Multi-GPU training for billion-scale datasets
      
      - For transductive link prediction
          python -m torch.distributed.launch --nproc_per_node=9 train_dist.py --data GDELT --config ./config/dist/TimeSGN.yml --num_gpus 8 
      

