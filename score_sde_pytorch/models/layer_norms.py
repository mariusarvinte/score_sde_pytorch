import torch.nn as nn



def shape_to_layer_norm(normalization_shape):
    return nn.LayerNorm(normalized_shape=normalization_shape)

