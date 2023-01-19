'''
transformer(encoder) for classification
'''
import math
import torch
import torch.nn as nn
from embedding import GCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    """
    This class is taken from the PyTorch tutorial
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

#TODO following code taken from https://n8henrie.com/2021/08/writing-a-transformer-classifier-in-pytorch/

class Net(nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """

    def __init__(
        self,
        embeddings,
        nhead=8,
        dim_feedforward=2048,
        num_layers=6,
        dropout=0.1,
        activation="relu", #TODO???
        classifier_dropout=0.1, #TODO????
    ):

        super().__init__()

        vocab_size, d_model = embeddings.size()
        self.d_model = d_model
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.emb = GCN(n_in, n_h, activation) #TODO decide what to use here

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=vocab_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(d_model, 1) # binary classification
        self.sigmoid =  nn.Sigmoid()

    def forward(self, x):
        x = self.emb(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # x = x.mean(dim=1) ????
        x = self.classifier(x)
        x = self.sigmoid(x)

        return x

''' TRAINING PART '''
epochs = 50
model = Net(
    TEXT.vocab.vectors,
    nhead=5,  # the number of heads in the multiheadattention models
    dim_feedforward=50,  # the dimension of the feedforward network model in nn.TransformerEncoder
    num_layers=6,
    dropout=0.0,
    classifier_dropout=0.0,
).to(device)

criterion = nn.CrossEntropyLoss()

lr = 1e-4
optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad), lr=lr
)


torch.manual_seed(0)

print("starting")
for epoch in range(epochs):
    print(f"{epoch=}")
    epoch_loss = 0
    epoch_correct = 0
    epoch_count = 0
    for batch in iter(train_iter):
        predictions = model(batch.text.to(device))
        labels = batch.label.to(device) - 1

        loss = criterion(predictions, labels)

        correct = predictions.argmax(axis=1) == labels
        acc = correct.sum().item() / correct.size(0)

        epoch_correct += correct.sum().item()
        epoch_count += correct.size(0)

        epoch_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()

    with torch.no_grad():
        test_epoch_loss = 0
        test_epoch_correct = 0
        test_epoch_count = 0

        for batch in iter(test_iter):
            predictions = model(batch.text.to(device))
            labels = batch.label.to(device) - 1
            test_loss = criterion(predictions, labels)

            correct = predictions.argmax(axis=1) == labels
            acc = correct.sum().item() / correct.size(0)

            test_epoch_correct += correct.sum().item()
            test_epoch_count += correct.size(0)
            test_epoch_loss += loss.item()

    print(f"{epoch_loss=}")
    print(f"epoch accuracy: {epoch_correct / epoch_count}")
    print(f"{test_epoch_loss=}")
    print(f"test epoch accuracy: {test_epoch_correct / test_epoch_count}")
