import torch
import torch.nn as nn

class DeepLOB(nn.Module):
    """
    DeepLOB: Deep Learning for Limit Order Books.
    Input shape: (batch_size, 1, lookback_window, 40)
    Output shape: (batch_size, num_classes) where classes are [Down, Flat, Up]
    """
    def __init__(self, lookback_window=100, num_features=40, num_classes=3):
        super(DeepLOB, self).__init__()
        self.lookback_window = lookback_window
        self.num_features = num_features
        self.num_classes = num_classes

        # Conv Block 1
        # Input: (batch, 1, 100, 40) -> Outputs 16 channels, (1, 2) stride (1, 2) groups price/volume
        # Size after conv1: (batch, 16, 100, 20)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16)
        )
        
        # Temporal convs on level features
        self.conv1_temp = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1), padding=(2, 0)), # padding same on time
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16)
        )

        # Conv Block 2
        # Input: (batch, 16, 100, 20) -> Groups Bid/Ask pairs. Kernel (1, 2) stride (1, 2)
        # Size after conv2: (batch, 16, 100, 10)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16)
        )
        self.conv2_temp = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16)
        )

        # Conv Block 3
        # Input: (batch, 16, 100, 10) -> Convolves across all depth levels (1, 10)
        # Size after conv3: (batch, 16, 100, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16)
        )
        self.conv3_temp = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16)
        )

        # LSTM Layer
        # Reshaped input: (batch, 100, 16)
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # Classifier
        self.fc1 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 1, lookback_window, 40)
        
        # Block 1
        out = self.conv1(x)
        # Padding adjustment to ensure exact time length matching (same padding output height)
        res = self.conv1_temp(out)
        out = out + res[:, :, :self.lookback_window, :] # residual connection

        # Block 2
        out = self.conv2(out)
        res = self.conv2_temp(out)
        out = out + res[:, :, :self.lookback_window, :]

        # Block 3
        out = self.conv3(out)
        res = self.conv3_temp(out)
        out = out + res[:, :, :self.lookback_window, :]

        # Reshape for LSTM: (batch, channels, time, 1) -> (batch, time, channels)
        # out shape is (batch, 16, lookback_window, 1)
        out = out.squeeze(3) # (batch, 16, lookback_window)
        out = out.transpose(1, 2) # (batch, lookback_window, 16)

        # LSTM
        # lstm_out: (batch, lookback_window, hidden_size)
        # hn: (1, batch, hidden_size)
        lstm_out, (hn, cn) = self.lstm(out)

        # Classify the last timestep's representation
        # hn[-1] shape: (batch, hidden_size)
        logits = self.fc1(lstm_out[:, -1, :])
        return logits

if __name__ == "__main__":
    # Test network forward pass
    model = DeepLOB(lookback_window=100, num_features=40, num_classes=3)
    dummy_input = torch.randn(8, 1, 100, 40)
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    assert output.shape == (8, 3), "Output shape should be (8, 3)"
    print("DeepLOB architecture test passed!")
