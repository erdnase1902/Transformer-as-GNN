# Transformer-as-GNN
Transformer is the SOTA model for text modeling in NLP and
traditionally viewed as a stand-alone model. However, it can also be
formalized as a graph neural network. It models a sentence as a fully
connected graph, where words are nodes and they are connected
to each other. We will implement BERT, the bidirectional encoder
representations from transformers, as a graph neural network. The
GNN will be large and inefficient, so we will explore ways to prune
the graph.
