"""CRNN — convolutional-recurrent network for line-level OCR.

Architecture follows the original Shi et al. (2015) CRNN topology, downsized
for our narrow domain (~22 classes, short labels).  Smaller channel counts
keep inference cheap; we don't need a 25M-param model to read digits.

Input  : (B, 1, 32, W)  grayscale, fixed height 32
Output : (T, B, C)     log-probs over CHARSET, T = W // 4

The CNN downsamples height 32 → 1 and width by a factor of 4.  The BiLSTM
then runs over the resulting time axis.  We use ctc_loss on the output.
"""

import torch
import torch.nn as nn

from .charset import NUM_CLASSES


def _conv_bn_relu(in_ch: int, out_ch: int, k=3, s=1, p=1) -> nn.Sequential:
    """Standard conv-bn-relu block.  BN stabilises training when batches are
    mixed-content (we generate synthetic data on the fly and minibatch
    statistics swing more than with curated datasets)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CRNN(nn.Module):
    """CRNN with a lightweight VGG-like backbone."""

    # Per-layer width-pool keeps the time axis dense — we want T ≈ W/4, not
    # W/16, so the model has multiple "looks" at narrow glyphs like '1' and
    # decimal points.  Height collapses fully to 1 by the last block.

    def __init__(self, num_classes: int = NUM_CLASSES, lstm_hidden: int = 192):
        super().__init__()
        self.cnn = nn.Sequential(
            _conv_bn_relu(1,   64),
            nn.MaxPool2d(2, 2),          # 32 → 16; W → W/2

            _conv_bn_relu(64,  128),
            nn.MaxPool2d(2, 2),          # 16 → 8;  W/2 → W/4

            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),  # 8 → 4;   W/4 unchanged

            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d((2, 1), (2, 1)),  # 4 → 2;   W/4 unchanged

            # 2x1 conv collapses the remaining height; no padding so H goes 2→1.
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=1, padding=(0, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # H = 1, W = (W/4) (last conv preserves width via the +1 horizontal pad)
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=False,
        )
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 32, W) → log-probs (T, B, C)."""
        feat = self.cnn(x)                       # (B, 512, 1, T)
        assert feat.size(2) == 1, f"unexpected height {feat.size(2)}"
        feat = feat.squeeze(2).permute(2, 0, 1)  # (T, B, 512)
        rnn_out, _ = self.lstm(feat)             # (T, B, 2H)
        logits = self.fc(rnn_out)                # (T, B, C)
        return logits.log_softmax(dim=-1)


def build_model(device: str | torch.device = "cpu") -> CRNN:
    model = CRNN()
    model.to(device)
    return model
