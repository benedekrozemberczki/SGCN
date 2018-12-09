SGCN
============================================

- [ ] **Log time.**
- [ ] **Complete refactor to make it nice.**
- [ ] **Data read from disk.**
- [ ] **Write README**
- [ ] **Post on Reddit.**
- [ ] **Add to graph based literature webpage.**
- [ ] **Add to Chihmings collections as it is node level.**

A PyTorch implementation of "Signed Graph Convolutional Network" (ICDM 2018).

<div style="text-align:center"><img src ="danmf.jpg" ,width=720/></div>
<p align="justify">
Community structure is ubiquitous in real-world complex networks. The task of community detection over these networks is of paramount importance in a variety of applications. Recently, nonnegative matrix factorization (NMF) has been widely adopted for community detection due to its great interpretability and its natural fitness for capturing the community membership of nodes. However, the existing NMF-based community detection approaches are shallow methods. They learn the community assignment by mapping the original network to the community membership space directly. Considering the complicated and diversified topology structures of real-world networks, it is highly possible that the mapping between the original network and the community membership space contains rather complex hierarchical information, which cannot be interpreted by classic shallow NMF-based approaches. Inspired by the unique feature representation learning capability of deep autoencoder, we propose a novel model, named Deep Autoencoder-like NMF (DANMF), for community detection. Similar to deep autoencoder, DANMF consists of an encoder component and a decoder component. This architecture empowers DANMF to learn the hierarchical mappings between the original network and the final community  assignment  with  implicit  low-to-high  level  hidden attributes of the original network learnt in the intermediate layers. Thus, DANMF should be better suited to the community detection task. Extensive experiments on benchmark datasets demonstrate that DANMF can achieve better performance than the state-of-the-art NMF-based community detection approaches.</p>

This repository provides an implementation for DANMF as described in the paper:

> Deep Autoencoder-like Nonnegative Matrix Factorization for Community Detection.
> Fanghua Ye, Chuan Chen, and Zibin Zheng.
> CIKM, 2018.
> [[Paper]](https://smartyfh.com/Documents/18DANMF.pdf)


### Requirements

The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx          1.11
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
sklearn           0.20.0
```
### Datasets

The code takes an input graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header. Nodes should be indexed starting with 0. Sample graphs for the `Twitch Brasilians` ,`Wikipedia Chameleons` and `Wikipedia Giraffes` are included in the  `input/` directory. 

### Options

Learning of the embedding is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options

```
  --edge-path         STR    Input graph path.       Default is `input/ptbr_edges.csv`.
  --membership-path   STR    Membership path.        Default is `output/ptbr_membership.json`.
  --output-path       STR    Embedding path.         Default is `output/ptbr_danmf.csv`.
```

#### Model options

```
  --pre-training-method   STR         Layer pre-training method.            Default is `shallow`. 
  --iterations            INT         Number of epochs.                     Default is 100.
  --pre-iterations        INT         Layer-wise epochs.                    Default is 100.
  --seed                  INT         Random seed value.                    Default is 42.
  --lamb                  FLOAT       Regularization parameter.             Default is 0.01.
  --layers                LST         Layer sizes in autoencoder model.     Default is [32, 8]
  --calculate-loss        BOOL        Loss calculation for the model.       Default is False.  
```

### Examples

The following commands learn a graph embedding and write the embedding to disk. The node representations are ordered by the ID. The layer sizes are always set manually.

Creating a DANMF embedding of the default dataset with a 128-64-32-16 architecture. Saving the embedding at the default path.
```
python src/main.py --layers 128 64 32 16
```
Creating a DANMF embedding of the default dataset with a 96-8 architecture and calculationg the loss.
```
python src/main.py --layers 96 8 --calculate-loss
```
Creating a single layer DANMF embedding with 32 factors.
```
python src/main.py --layers 32
```
Creating an embedding with some custom cluster number in the bottleneck layer.
```
python src/main.py --layers 128 64 7
```
Creating an embedding of the default dataset with a 32-8 architecture and sklearn layer pre-training.
```
python src/main.py --layers 32 4 --pre-training-method sklearn
```
Creating an embedding of another dataset the `Wikipedia Chameleons`. Saving the output in a custom folder.
```
python src/main.py --layers 32 8 --edge-path input/chameleon_edges.csv --output-path output/chameleon_danmf.csv --membership-path output/chameleon_membership.json
```
