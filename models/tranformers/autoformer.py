import torch
import torch.nn as nn


class TimeAwareEmbedding(nn.Module):
    """Learnable embeddings for categorical time-based features"""

    def __init__(self):
        super(TimeAwareEmbedding, self).__init__()
        self.hour_embedding = nn.Embedding(24, 8)  # Hour scale
        self.day_embedding = nn.Embedding(7, 8)  # Weekday scale
        self.month_embedding = nn.Embedding(12, 8)  # Month scale
        self.holiday_embedding = nn.Embedding(2, 4)  # Holiday/non-holiday
        self.workday_embedding = nn.Embedding(2, 4)  # Workday/non-workday

    def forward(self, hour, day, month, holiday, workday):
        hour_emb = self.hour_embedding(hour)
        day_emb = self.day_embedding(day)
        month_emb = self.month_embedding(month)
        holiday_emb = self.holiday_embedding(holiday)
        workday_emb = self.workday_embedding(workday)
        return torch.cat(
            [hour_emb, day_emb, month_emb, holiday_emb, workday_emb], dim=-1
        )


class Autoformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Autoformer, self).__init__()
        self.time_embedding = TimeAwareEmbedding()
        self.feature_fc = nn.Linear(input_dim, hidden_dim)
        self.auto_corr = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hour, day, month, holiday, workday):
        time_features = self.time_embedding(hour, day, month, holiday, workday)
        x = torch.cat([x, time_features], dim=-1)  # Merge features
        x = self.feature_fc(x)  # Apply transformation
        x = x.permute(0, 2, 1)  # Reshape for Conv1D
        x = self.auto_corr(x)  # Auto-Correlation mechanism
        x = torch.relu(x)

        # Apply Multi-Head Attention
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.mean(dim=-1)  # Aggregate features

        return self.fc(x)
