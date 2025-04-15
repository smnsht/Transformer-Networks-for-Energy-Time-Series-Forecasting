import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from workalendar.europe import BadenWurttemberg

# Load dataset
data = pd.DataFrame(
    {
        "utc_timestamp": pd.date_range(start="1/1/2015", periods=500, freq="15min"),
        "load_actual_entsoe_transparency": np.random.randint(5000, 7000, 500),
        "is_holiday": np.random.choice([0, 1], size=500, p=[0.9, 0.1]),  # Holidays
    }
)

# Initialize work calendar and compute workdays
calendar = BadenWurttemberg()
data["is_workday"] = data["utc_timestamp"].apply(
    lambda dt: int(calendar.is_working_day(dt))
)

# Extract time-based features
data["hour"] = data["utc_timestamp"].dt.hour
data["dayofweek"] = data["utc_timestamp"].dt.dayofweek
data["month"] = data["utc_timestamp"].dt.month - 1  # Adjust month index for embedding

# Normalize load values
scaler = StandardScaler()
data[["load_actual_entsoe_transparency"]] = scaler.fit_transform(
    data[["load_actual_entsoe_transparency"]]
)

# ✅ Retaining your original tensor extraction process:
X_values = data[["load_actual_entsoe_transparency"]].values
hour_values = data["hour"].values
day_values = data["dayofweek"].values
month_values = data["month"].values
holiday_values = data["is_holiday"].values
workday_values = data["is_workday"].values

# Prepare sequences for model input
seq_length = 20
X, hour_list, day_list, month_list, holiday_list, workday_list, y = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
)

for i in range(len(data) - seq_length):
    X.append(X_values[i : i + seq_length])
    hour_list.append(hour_values[i : i + seq_length])
    day_list.append(day_values[i : i + seq_length])
    month_list.append(month_values[i : i + seq_length])
    holiday_list.append(holiday_values[i : i + seq_length])
    workday_list.append(workday_values[i : i + seq_length])
    y.append(X_values[i + seq_length])  # Target load value

# Convert lists to numpy arrays
X_array = np.array(X, dtype=np.float32)
hour_array = np.array(hour_list, dtype=np.int64)
day_array = np.array(day_list, dtype=np.int64)
month_array = np.array(month_list, dtype=np.int64)
holiday_array = np.array(holiday_list, dtype=np.int64)
workday_array = np.array(workday_list, dtype=np.int64)
y_array = np.array(y, dtype=np.float32)

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.from_numpy(X_array)
hour_tensor = torch.from_numpy(hour_array)
day_tensor = torch.from_numpy(day_array)
month_tensor = torch.from_numpy(month_array)
holiday_tensor = torch.from_numpy(holiday_array)
workday_tensor = torch.from_numpy(workday_array)
y_tensor = torch.from_numpy(y_array)

print("Tensor shapes before model:")
print(
    X_tensor.shape,
    hour_tensor.shape,
    day_tensor.shape,
    month_tensor.shape,
    holiday_tensor.shape,
    workday_tensor.shape,
)


# Autoformer model with workday & holiday awareness
class TimeAwareEmbedding(nn.Module):
    def __init__(self):
        super(TimeAwareEmbedding, self).__init__()
        self.hour_embedding = nn.Embedding(24, 8)
        self.day_embedding = nn.Embedding(7, 8)
        self.month_embedding = nn.Embedding(12, 8)
        self.holiday_embedding = nn.Embedding(2, 4)
        self.workday_embedding = nn.Embedding(2, 4)

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
    """Autoformer Model for Load Forecasting with Time-Based Features."""

    def __init__(self, input_dim, hidden_dim, output_dim, time_emb_dim=36):
        super(Autoformer, self).__init__()
        self.time_embedding = TimeAwareEmbedding()
        self.feature_fc = nn.Linear(
            input_dim + time_emb_dim, hidden_dim
        )  # Adjust input dim dynamically
        self.auto_corr = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hour, day, month, holiday, workday):
        # Step 1: Embed time-based categorical features
        time_features = self.time_embedding(
            hour, day, month, holiday, workday
        )  # Shape: [batch_size, embedding_dim]

        # Step 2: Manually expand time features to match `seq_length`
        batch_size, seq_length, emb_dim = (
            x.shape[0],
            x.shape[1],
            time_features.shape[-1],
        )
        expanded_time_features = torch.zeros(
            (batch_size, seq_length, emb_dim),
            dtype=time_features.dtype,
            device=time_features.device,
        )

        for i in range(batch_size):
            for j in range(seq_length):
                expanded_time_features[i, j, :] = time_features[i, j, :]  # Fill manually

        # Step 3: Concatenate time features with main input tensor
        x = torch.cat([x, expanded_time_features], dim=-1)

        # Step 4: Apply feature transformation
        x = self.feature_fc(x)  # Linear transformation

        # Step 5: Reshape for Conv1D processing
        x = x.permute(0, 1, 2)  # Shape for CNN: [batch_size, seq_length, hidden_dim]
        x = self.auto_corr(x)  # Auto-Correlation mechanism
        x = torch.relu(x)

        print("Shape before attention:", x.shape)

        # Step 6: Apply Multi-Head Attention
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.mean(dim=-1)  # Aggregate features

        # Step 7: Final output transformation
        return self.fc(x)  # Shape: [batch_size, output_dim]


# Example usage
batch_size = X_tensor.shape[0]  # Matches dataset size
seq_length = X_tensor.shape[1]  # Time steps
input_dim = X_tensor.shape[-1]  # Updated to reflect actual feature count
hidden_dim = 64
output_dim = 1
time_emb_dim = 32  # Sum of embedded feature dimensions (8+8+8+4+4)


print("Tensor shapes before Autoformer:")
print("X_tensor:", X_tensor.shape)
print("Hour_tensor:", hour_tensor.shape)
print("Day_tensor:", day_tensor.shape)
print("Month_tensor:", month_tensor.shape)
print("Holiday_tensor:", holiday_tensor.shape)
print("Workday_tensor:", workday_tensor.shape)


# Initialize and test model
model = Autoformer(input_dim, hidden_dim, output_dim, time_emb_dim)
output = model(
    X_tensor, hour_tensor, day_tensor, month_tensor, holiday_tensor, workday_tensor
)
print("Output shape:", output.shape)
