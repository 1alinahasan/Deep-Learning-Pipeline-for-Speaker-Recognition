import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNNLayer(nn.Module):
    """Time Delay Neural Network layer."""
    def __init__(self, input_dim, input_context, output_dim):
        super().__init__()
        assert isinstance(input_context, (list, tuple)) and len(input_context) > 0

        self.context = list(input_context)
        self.min_off = min(self.context)
        self.max_off = max(self.context)

        self.linear = nn.Linear(input_dim * len(self.context), output_dim)
        self.act = nn.ReLU()
        


    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected [B,T,F], got {x.shape}")

        B, T, F = x.shape

        # valid center times so all context frames exist
        start_t = -self.min_off
        end_t = T - self.max_off   # exclusive
        if end_t <= start_t:
            raise ValueError(f"Sequence too short: T={T} for context [{self.min_off},{self.max_off}]")

        T_out = end_t - start_t

        # splice: collect frames at offsets, then concat on feature dim
        splices = []
        for off in self.context:
            splices.append(x[:, start_t + off : start_t + off + T_out, :])  # [B, T_out, F]

        x_spliced = torch.cat(splices, dim=2)  # [B, T_out, F*len(context)]
        y = self.linear(x_spliced)             # [B, T_out, output_dim]
        return self.act(y)


class XVector(nn.Module):
    def __init__(self, size=1, depth=1, num_speakers=1000, embedding_dim=512, input_dim=24,  internal_dim=None):
        super().__init__()
        input_dim = 24  # from your FBANK collator
        if internal_dim is None:
            internal_dim = embedding_dim

        # -------- Frame-level TDNN stack --------
        self.tdnn1 = TDNNLayer(input_dim, [-2, -1, 0, 1, 2], 512)
        self.tdnn2 = TDNNLayer(512, [-2, 0, 2], 512)
        self.tdnn3 = TDNNLayer(512, [-3, 0, 3], 512)
        self.tdnn4 = TDNNLayer(512, [0], 512)
        self.tdnn5 = TDNNLayer(512, [0], 1500)

        # -------- Statistics Pooling --------
        # mean + std over time
        self.embedding_dim = embedding_dim

        # -------- Segment-level layers --------
        self.seg_fc1 = nn.Linear(3000, embedding_dim)
        self.seg_fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.relu = nn.ReLU()

        # -------- Classification head --------
        self.classifier = nn.Linear(embedding_dim, num_speakers)
        def extract_embedding(self, x):
            # Frame-level TDNN
            h = self.tdnn1(x)
            h = self.tdnn2(h)
            h = self.tdnn3(h)
            h = self.tdnn4(h)
            h = self.tdnn5(h)

            # Stats pooling
            mean = torch.mean(h, dim=1)
            std = torch.std(h, dim=1)
            stats = torch.cat([mean, std], dim=1)

            # Segment-level
            h = self.segment1(stats)
            emb = self.segment2(h)   # <-- this is the embedding (x-vector)

            return emb
        
    def forward(self, x):
        # Frame-level
        h = self.tdnn1(x)
        h = self.tdnn2(h)
        h = self.tdnn3(h)
        h = self.tdnn4(h)
        h = self.tdnn5(h)     # [B, T', 1500]

        # -------- Statistics pooling --------
        mean = h.mean(dim=1)
        std = h.std(dim=1, unbiased=False)
        stats = torch.cat([mean, std], dim=1)  # [B, 3000]

        # -------- Segment-level --------
        emb = self.seg_fc1(stats)
        h2 = self.relu(emb)
        h2 = self.relu(self.seg_fc2(h2))

        # -------- Classification --------
        logits = self.classifier(h2)

        return logits

