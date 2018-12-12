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

<div style="text-align:center"><img src ="sgcn.jpg" ,width=720/></div>
<p align="justify">
Due to the fact much of today's data can be represented as graphs, there has been a demand for generalizing neural network models for graph data. One recent direction that has shown fruitful results, and therefore growing interest, is the usage of graph convolutional neural networks (GCNs). They have been shown to provide a significant improvement on a wide range of tasks in network analysis, one of which being node representation learning. The task of learning low-dimensional node representations has shown to increase performance on a plethora of other tasks from link prediction and node classification, to community detection and visualization. Simultaneously, signed networks (or graphs having both positive and negative links) have become ubiquitous with the growing popularity of social media. However, since previous GCN models have primarily focused on unsigned networks (or graphs consisting of only positive links), it is unclear how they could be applied to signed networks due to the challenges presented by negative links. The primary challenges are based on negative links having not only a different semantic meaning as compared to positive links, but their principles are inherently different and they form complex relations with positive links. Therefore we propose a dedicated and principled effort that utilizes balance theory to correctly aggregate and propagate the information across layers of a signed GCN model. We perform empirical experiments comparing our proposed signed GCN against state-of-the-art baselines for learning node representations in signed networks. More specifically, our experiments are performed on four real-world datasets for the classical link sign prediction problem that is commonly used as the benchmark for signed network embeddings algorithms. </p>

This repository provides an implementation for SGCN as described in the paper:

> Signed Graph Convolutional Network.
> Tyler Derr, Yao Ma, and Jiliang Tang
> ICDM, 2018.
> [[Paper]](https://arxiv.org/abs/1808.06354)


### Requirements

The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx           1.11
tqdm               4.28.1
numpy              1.15.4
pandas             0.23.4
texttable          1.5.0
scipy              1.1.0
argparse           1.1.0
sklearn            0.20.0
torch              0.4.1.post2
torch-cluster      1.1.5
torch-geometric    0.3.1
torch-scatter      1.0.4
torch-sparse       0.2.2
torch-spline-conv  1.0.4
torchvision        0.2.1
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
