import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention cho hiệu quả tốt hơn"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        return self.out_proj(context), attn_weights

class Attention(nn.Module):
    """Simple Attention để tổng hợp sequence thành vector"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attn_weights = self.attn(lstm_output)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context, attn_weights

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block cho channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Residual Block với skip connection"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        out = out + residual
        return F.relu(out)

class WAF_Attention_Model(nn.Module):
    """
    Cải tiến model với:
    1. Multi-Head Self-Attention
    2. Residual Blocks với SE attention
    3. Layer Normalization
    4. Dropout được điều chỉnh
    5. Deeper network với skip connections
    """
    def __init__(self, vocab_size, embedding_dim, num_classes=1, dropout=0.3):
        super().__init__()

        # 1. Embedding với Dropout
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.1)

        # 2. CNN Feature Extraction với Residual Blocks
        self.conv_input = nn.Conv1d(embedding_dim, 128, kernel_size=1)
        self.res_block1 = ResidualBlock(128, 128, kernel_size=3)
        self.res_block2 = ResidualBlock(128, 256, kernel_size=3)
        self.res_block3 = ResidualBlock(256, 256, kernel_size=3)

        self.pool = nn.MaxPool1d(2)
        self.conv_dropout = nn.Dropout(0.15)

        # 3. Multi-Head Self-Attention
        self.layer_norm1 = nn.LayerNorm(256)
        self.self_attention = MultiHeadAttention(256, num_heads=8, dropout=0.1)
        self.layer_norm2 = nn.LayerNorm(256)

        # 4. Bi-Directional LSTM
        self.lstm_hidden = 256
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # 5. Attention để tổng hợp
        self.attention = Attention(self.lstm_hidden * 2)

        # 6. Classification Head với Residual
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier/Kaiming initialization cho better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def forward(self, x):
        # 1. Embedding
        x = self.embedding(x)  # (Batch, Seq, Emb)
        x = self.embed_dropout(x)

        # 2. CNN với Residual Blocks
        x = x.permute(0, 2, 1)  # (Batch, Emb, Seq)
        x = self.conv_input(x)

        x = self.res_block1(x)
        x = self.pool(x)
        x = self.conv_dropout(x)

        x = self.res_block2(x)
        x = self.pool(x)
        x = self.conv_dropout(x)

        x = self.res_block3(x)

        # 3. Self-Attention
        x = x.permute(0, 2, 1)  # (Batch, Seq, Channels)
        x_norm = self.layer_norm1(x)
        attn_out, _ = self.self_attention(x_norm)
        x = x + attn_out  # Residual connection
        x = self.layer_norm2(x)

        # 4. LSTM
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)

        # 5. Attention pooling
        context_vector, _ = self.attention(lstm_out)

        # 6. Classification (trả về logits, sigmoid sẽ apply trong loss/inference)
        output = self.classifier(context_vector)

        return output  # Trả về logits, không qua sigmoid


class FocalLoss(nn.Module):
    """
    Focal Loss để xử lý class imbalance tốt hơn
    Giảm trọng số của easy examples, tập trung vào hard examples
    Sử dụng BCEWithLogitsLoss để tương thích với mixed precision
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Sử dụng binary_cross_entropy_with_logits cho numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Tính probability từ logits
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingBCE(nn.Module):
    """
    Label Smoothing cho Binary Classification
    Giúp model generalize tốt hơn
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(inputs, targets)
