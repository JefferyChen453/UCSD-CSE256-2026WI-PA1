from einops import reduce
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from sentiment_data import read_sentiment_examples, read_word_embeddings


class SentimentDatasetDAN(Dataset):
    def __init__(
        self,
        infile,
        emb_dim=300,
        load_pretrained_embedding=True,
    ):
        word_embeddings = read_word_embeddings(f"data/glove.6B.{emb_dim}d-relativized.txt")
        self.word_indexer = word_embeddings.word_indexer

        examples = read_sentiment_examples(infile)
        self.labels = [ex.label for ex in examples]
        self.token_ids_list = []
        for ex in examples:
            token_ids = [self.word_indexer.index_of(word) for word in ex.words]
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


class DAN(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_hidden_layers: int,
        hidden_dim: int,
        dropout: bool = True,
        dropout_rate: float = 0.2,
        load_pretrained_embedding: bool = True,
        freeze_embedding: bool = True,
        train_unk_token: bool = True
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.train_unk_token = train_unk_token
        self.dropout = dropout
        if self.dropout:
            self.dropout_layer = nn.Dropout(dropout_rate)

        word_emb = read_word_embeddings(f"data/glove.6B.{emb_dim}d-relativized.txt")
        self.word_indexer = word_emb.word_indexer
        if load_pretrained_embedding:
            self.embeddings = word_emb.get_initialized_embedding_layer(freeze_embedding)
        else:
            self.embeddings = nn.Embedding(
                num_embeddings=len(self.word_indexer),
                embedding_dim=emb_dim,
                freeze=False
            )
        
        self.input_layer = nn.Linear(emb_dim, hidden_dim)
        if num_hidden_layers:
            self.ffn_layers = nn.ModuleList(
                [
                    nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)
                ]
            )
        self.output_layer = nn.Linear(hidden_dim, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)
            
    def forward(self, input_ids):
        x = self.embeddings(input_ids) # [bs, seq_len] -> [bs, seqlen, emb_dim]

        pad_mask = (input_ids != 0).to(torch.long)
        if self.train_unk_token:
            mask = pad_mask
        else:
            unk_mask = (input_ids != 1).to(torch.long)
            mask = unk_mask * pad_mask

        x = x * mask.unsqueeze(-1)
        x = reduce(x, "bs seqlen emb_dim -> bs emb_dim", "sum") 
        mask_sum = mask.sum(dim=-1, keepdim=True)
        mask_sum = torch.clamp(mask_sum, min=1e-8)
        x /= mask_sum


        x = F.relu(self.input_layer(x))
        if self.dropout:
            x = self.dropout_layer(x)
        if self.num_hidden_layers:
            for layer in self.ffn_layers:
                x = F.relu(layer(x))
                if self.dropout:
                    x = self.dropout_layer(x)
        x = self.output_layer(x)
        x = self.log_softmax(x)

        return x
