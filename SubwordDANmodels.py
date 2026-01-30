from einops import reduce
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from sentiment_data import read_sentiment_examples
from tokenizer.tokenizer import Tokenizer

class SentimentDatasetSubwordDAN(Dataset):
    def __init__(
        self,
        infile,
        tokenizer: Tokenizer,
    ):
        self.tokenizer = tokenizer

        examples = read_sentiment_examples(infile)
        self.labels = [ex.label for ex in examples]
        self.token_ids_list = []
        for ex in examples:
            token_ids = self.tokenizer.encode(" ".join(ex.words))
            self.token_ids_list.append(token_ids)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.token_ids_list[idx], self.labels[idx]

def collate_fn(batch, pad_token_id=0):
    # batch: List[Tuple[List[int], int]]
    token_ids_list = [torch.tensor(x[0], dtype=torch.long) for x in batch]
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)

    token_ids = pad_sequence(
        token_ids_list,
        batch_first=True,
        padding_value=pad_token_id,
    )
    
    return {
        "input_ids": token_ids, # [bs, seq_len]
        "labels": labels, # [bs,]
    }


ACTIVATIONS = {"relu": F.relu, "silu": F.silu, "tanh": torch.tanh}


class SubwordDAN(nn.Module):
    def __init__(
        self,
        tokenizer: Tokenizer,
        emb_dim: int,
        num_hidden_layers: int,
        hidden_dim: int,
        training: bool,
        dropout_word: bool = True,
        dropout_hidden: bool = True,
        dropout_rate: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        assert num_hidden_layers > 0, "Number of hidden layers must be greater than 0"
        self.training = training
        self.num_hidden_layers = num_hidden_layers
        self.dropout_word = dropout_word
        self.dropout_hidden = dropout_hidden
        self.dropout_rate = dropout_rate
        self.activation = ACTIVATIONS.get(activation, F.relu)
        self.tokenizer = tokenizer

        if self.dropout_hidden:
            self.dropout_layer = nn.Dropout(self.dropout_rate)

        self.embeddings = nn.Embedding(
            num_embeddings=len(self.tokenizer.vocab),
            embedding_dim=emb_dim,
        )
        
        self.input_layer = nn.Linear(emb_dim, hidden_dim)
        self.ffn_layers = nn.ModuleList(
            [
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers - 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_dim, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)
            
    def forward(self, input_ids):
        x = self.embeddings(input_ids) # [bs, seq_len] -> [bs, seqlen, emb_dim]

        mask = (input_ids != 1).to(torch.long)

        if self.training and self.dropout_word:
            word_drop_mask = torch.bernoulli(
                torch.full(mask.shape, 1 - self.dropout_rate, device=input_ids.device)
            )
            mask = mask * word_drop_mask

        x = x * mask.unsqueeze(-1)
        x = reduce(x, "bs seqlen emb_dim -> bs emb_dim", "sum") 
        mask_sum = mask.sum(dim=-1, keepdim=True)
        mask_sum = torch.clamp(mask_sum, min=1e-8)
        x /= mask_sum

        x = self.activation(self.input_layer(x))
        if self.dropout_hidden:
            x = self.dropout_layer(x)
        for layer in self.ffn_layers:
            x = self.activation(layer(x))
            if self.dropout_hidden:
                x = self.dropout_layer(x)
        x = self.output_layer(x)
        x = self.log_softmax(x)

        return x
