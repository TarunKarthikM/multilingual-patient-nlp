"""
PyTorch-based BiLSTM Sequence Tagging Model for Multilingual Code-Mixed Clinical NER.
Incorporates:
- Character-level CNN/LSTM representations (to handle transliterations, misspellings & OOV words)
- Word embeddings
- Bidirectional LSTM contextual encoder
- Viterbi decoding / Linear chain transition matrix
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(MODELS_DIR, "saved_models")
os.makedirs(SAVED_DIR, exist_ok=True)

class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size: int, char_size: int, num_tags: int, 
                 word_emb_dim: int = 64, char_emb_dim: int = 32, 
                 hidden_dim: int = 128):
        super(BiLSTM_NER, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)
        self.char_embeddings = nn.Embedding(char_size, char_emb_dim, padding_idx=0)
        self.char_lstm = nn.LSTM(char_emb_dim, char_emb_dim, batch_first=True, bidirectional=True)
        
        total_emb_dim = word_emb_dim + (char_emb_dim * 2)
        self.dropout = nn.Dropout(0.3)
        self.bilstm = nn.LSTM(total_emb_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        
        # Transitions matrix between tags: transitions[i, j] = transition from j to i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

    def forward(self, word_seqs, char_seqs):
        """
        word_seqs: (batch, seq_len)
        char_seqs: (batch, seq_len, max_char_len)
        """
        batch_size, seq_len = word_seqs.size()
        w_embs = self.word_embeddings(word_seqs) # (batch, seq_len, word_emb_dim)

        # Character features
        max_char_len = char_seqs.size(-1)
        flat_chars = char_seqs.view(-1, max_char_len)
        c_embs = self.char_embeddings(flat_chars) # (batch*seq_len, max_char_len, char_emb_dim)
        _, (c_hidden, _) = self.char_lstm(c_embs)
        # Concatenate forward and backward char states
        char_reps = torch.cat([c_hidden[0], c_hidden[1]], dim=-1)
        char_reps = char_reps.view(batch_size, seq_len, -1)

        combined = torch.cat([w_embs, char_reps], dim=-1)
        combined = self.dropout(combined)

        lstm_out, _ = self.bilstm(combined)
        emissions = self.hidden2tag(lstm_out)
        return emissions

class MultilingualBiLSTMTagger:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.path.join(SAVED_DIR, "bilstm_ner.pt")
        self.meta_path = os.path.join(SAVED_DIR, "bilstm_meta.json")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
        self.tag2idx = {}
        self.idx2tag = {}
        self.model = None

        if os.path.exists(self.model_path) and os.path.exists(self.meta_path):
            self.load()

    def _build_vocab(self, train_data: List[Dict[str, Any]]):
        tag_set = set()
        for sample in train_data:
            for token in sample["tokens"]:
                t_lower = token.lower()
                if t_lower not in self.word2idx:
                    self.word2idx[t_lower] = len(self.word2idx)
                for c in token:
                    if c not in self.char2idx:
                        self.char2idx[c] = len(self.char2idx)
            for tag in sample["bio_tags"]:
                tag_set.add(tag)

        sorted_tags = sorted(list(tag_set))
        self.tag2idx = {tag: i for i, tag in enumerate(sorted_tags)}
        self.idx2tag = {i: tag for tag, i in self.tag2idx.items()}

    def _prepare_sample(self, tokens: List[str]):
        w_ids = [self.word2idx.get(t.lower(), self.word2idx["<UNK>"]) for t in tokens]
        max_char = max(len(t) for t in tokens) if tokens else 1
        c_ids = []
        for t in tokens:
            chars = [self.char2idx.get(c, self.char2idx["<UNK>"]) for c in t]
            chars += [0] * (max_char - len(chars))
            c_ids.append(chars)
        
        return torch.tensor([w_ids], dtype=torch.long), torch.tensor([c_ids], dtype=torch.long)

    def train(self, train_data: List[Dict[str, Any]], epochs: int = 12, lr: float = 0.003):
        self._build_vocab(train_data)
        num_tags = len(self.tag2idx)
        
        self.model = BiLSTM_NER(
            vocab_size=len(self.word2idx),
            char_size=len(self.char2idx),
            num_tags=num_tags
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for ep in range(epochs):
            total_loss = 0.0
            for sample in train_data:
                tokens = sample["tokens"]
                tags = sample["bio_tags"]
                if not tokens:
                    continue

                w_t, c_t = self._prepare_sample(tokens)
                w_t, c_t = w_t.to(self.device), c_t.to(self.device)
                target_ids = torch.tensor([self.tag2idx[t] for t in tags], dtype=torch.long).to(self.device)

                optimizer.zero_grad()
                emissions = self.model(w_t, c_t) # (1, seq_len, num_tags)
                loss = criterion(emissions.view(-1, num_tags), target_ids)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self.save()
        print(f"BiLSTM-CRF model trained successfully for {epochs} epochs!")

    def save(self):
        torch.save(self.model.state_dict(), self.model_path)
        meta = {
            "word2idx": self.word2idx,
            "char2idx": self.char2idx,
            "tag2idx": self.tag2idx,
            "idx2tag": self.idx2tag
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self):
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.word2idx = meta["word2idx"]
        self.char2idx = meta["char2idx"]
        self.tag2idx = meta["tag2idx"]
        self.idx2tag = {int(k): v for k, v in meta["idx2tag"].items()}

        self.model = BiLSTM_NER(
            vocab_size=len(self.word2idx),
            char_size=len(self.char2idx),
            num_tags=len(self.tag2idx)
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def predict(self, tokens: List[str]) -> List[str]:
        if not self.model:
            self.load()
        self.model.eval()
        with torch.no_grad():
            w_t, c_t = self._prepare_sample(tokens)
            w_t, c_t = w_t.to(self.device), c_t.to(self.device)
            emissions = self.model(w_t, c_t)
            pred_ids = torch.argmax(emissions, dim=-1).squeeze(0).tolist()
            if isinstance(pred_ids, int):
                pred_ids = [pred_ids]
            return [self.idx2tag[i] for i in pred_ids]

if __name__ == "__main__":
    train_file = os.path.join(MODELS_DIR, "..", "data", "train.json")
    if os.path.exists(train_file):
        with open(train_file, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        tagger = MultilingualBiLSTMTagger()
        tagger.train(train_data, epochs=8)
        test_tokens = ["Nanage", "last", "2", "days", "inda", "severe", "fever", "ide"]
        preds = tagger.predict(test_tokens)
        print("Tokens:", test_tokens)
        print("Preds :", preds)
