# TimeSGN

This repo is the open-sourced code for our TimeSGN 

## Requirements

- python >= 3.6.13
- pytorch >= 1.8.1
- pandas >= 1.1.5
- numpy >= 1.19.5
- dgl >= 0.6.1
- pyyaml >= 5.4.1
- tqdm >= 4.61.0
- pybind11 >= 2.6.2
- g++ >= 7.5.0

## Datasets
- Wikipedia: http://snap.stanford.edu/jodie/wikipedia.csv
- Reddit: http://snap.stanford.edu/jodie/reddit.csv
- MOOC: http://snap.stanford.edu/jodie/mooc.csv
- LastFM: http://snap.stanford.edu/jodie/lastfm.csv
- GDELT: https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edges.csv;edge_features.pt
- MAG: https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/edges.csv

## Run
```{pytphon}
- For Transductive Link prediction
# python train.py --data WIKI --config ./config/TimeSGN.yml --gpu 0 --DTMP
```

