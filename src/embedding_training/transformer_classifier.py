'''
transformer(encoder) for classification
'''
import math
import torch
import torch.nn as nn
from embedding_training.embedding import GCN, AvgReadout

# sources (TODO delete):
#   - https://pytorch.org/tutorials/beginner/transformer_tutorial.html
#   - https://discuss.pytorch.org/t/nn-transformerencoder-for-classification/83021
#   - https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
#   - https://n8henrie.com/2021/08/writing-a-transformer-classifier-in-pytorch/ !!!!


class PositionalEncoding(nn.Module):
    """
    This class is taken from the PyTorch tutorial
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000): #TODO check parameters
        #probably max_len is the maximum length of the sequence -> number of frames in the longest video
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

#TODO following code taken from https://n8henrie.com/2021/08/writing-a-transformer-classifier-in-pytorch/

class Net(nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """

    def __init__(
        self,
        features_size,
        embedding_size, #d_model
        nhead=8,
        dim_feedforward=2048,
        num_layers=6,
        dropout=0.1,
        activation="prelu", #TODO???
        classifier_dropout=0.1, #TODO????
    ):

        super().__init__()

        self.d_model = embedding_size
        assert self.d_model % nhead == 0, "nheads must divide evenly into d_model"
        
        #TODO decide what GCN to use
        self.emb = GCN(features_size, embedding_size, activation)

        self.readout = AvgReadout()

        self.pos_encoder = PositionalEncoding(
            d_model=embedding_size,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(embedding_size, 1) # binary classification
        self.sigmoid =  nn.Sigmoid()

    def forward(self, seq_features_and_adjs):
        print("forwarding")
        # seq is a list of (features, adj)
        #   embedding takes (features, adj)
        #   transformer takes (seq)
        
        embedded_seq = list()
        for features, adj in seq_features_and_adjs:
            #print(features.shape)
            #print(adj.shape)
            x = self.emb(features, adj) # * math.sqrt(self.d_model)
            #print(f"result of embedding's shape: {x.shape}")
            x = self.readout(x)
            #print(f"result of readout's shape: {x.shape}")
            embedded_seq.append(x)
        
        embedded_seq = torch.stack(embedded_seq, dim=1)
        print(f"shape of embedded sequence: {embedded_seq.shape}")

        print("beginning with transformer")
        x = self.pos_encoder(x)
        print(x.shape)
        x = self.transformer_encoder(x)
        print(x.shape)
        # x = x.mean(dim=1) ????
        x = self.classifier(x)
        print(x.shape)
        x = self.sigmoid(x)
        print(x.shape)
        
        return x
