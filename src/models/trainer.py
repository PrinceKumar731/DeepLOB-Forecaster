import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class LOBDataset(Dataset):
    """
    PyTorch Dataset for DeepLOB sequence inputs.
    Extracts rolling lookback windows of LOB states.
    """
    def __init__(self, x_data, y_labels, lookback=100):
        """
        x_data: numpy array of shape (N, 40) - normalized order book states
        y_labels: numpy array of shape (N,) - integer labels
        lookback: size of sequence window (default 100)
        """
        self.lookback = lookback
        
        # Determine valid starting indices (where the target label is not -1)
        self.valid_indices = []
        for i in range(len(x_data) - lookback + 1):
            target_idx = i + lookback - 1
            if y_labels[target_idx] != -1:
                self.valid_indices.append(i)
                
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_labels, dtype=torch.long)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.lookback
        
        # Reshape to (1, lookback, features) as required by DeepLOB Conv2d
        x_seq = self.x[start_idx:end_idx].unsqueeze(0)
        y_val = self.y[end_idx - 1]
        
        return x_seq, y_val


class DeepLOBTrainer:
    """
    Trainer for DeepLOB model. Handles scaling, dataset loading, and training loop.
    """
    def __init__(self, model, lr=0.001, device='cpu', checkpoint_dir='checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.lr = lr
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        
        # Scaler parameters
        self.means = None
        self.stds = None
        
        os.makedirs(checkpoint_dir, exist_ok=True)

    def fit_scaler(self, x_train):
        """
        Calculates means and standard deviations on the training set.
        """
        self.means = np.mean(x_train, axis=0)
        self.stds = np.std(x_train, axis=0)
        # Avoid division by zero
        self.stds[self.stds == 0] = 1.0

    def scale_data(self, x):
        """
        Standardizes data using fitted scale parameters.
        """
        if self.means is None or self.stds is None:
            raise ValueError("Scaler must be fit on training data before scaling!")
        return (x - self.means) / self.stds

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
        val_loss = total_loss / total
        val_acc = correct / total
        return val_loss, val_acc, np.array(all_preds), np.array(all_targets)

    def train(self, x_train_raw, y_train, x_val_raw, y_val, batch_size=64, epochs=10, early_stopping_rounds=5):
        """
        Full training workflow.
        """
        # Fit scaler and scale data
        self.fit_scaler(x_train_raw)
        x_train = self.scale_data(x_train_raw)
        x_val = self.scale_data(x_val_raw)
        
        train_dataset = LOBDataset(x_train, y_train)
        val_dataset = LOBDataset(x_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Dataset summary:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples:   {len(val_dataset)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            elapsed = time.time() - t0
            
            print(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) - "
                  f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
            # Save scaler parameters inside checkpoints
            scaler_path = os.path.join(self.checkpoint_dir, 'scaler.npz')
            np.savez(scaler_path, means=self.means, stds=self.stds)
            
            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'deeplob_best.pth'))
                # print("  --> Saved new best model checkpoint.")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_rounds:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
                    
        # Load best model weights
        self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, 'deeplob_best.pth')))
        print("Loaded best model weights.")

    def load_model(self, model_dir):
        """Loads weights and scaler stats."""
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'deeplob_best.pth'), map_location=self.device))
        scaler_data = np.load(os.path.join(model_dir, 'scaler.npz'))
        self.means = scaler_data['means']
        self.stds = scaler_data['stds']
        print(f"Model and scaler successfully loaded from {model_dir}")
        
    def predict(self, x_raw):
        """Predicts on raw incoming features."""
        self.model.eval()
        x_scaled = self.scale_data(x_raw)
        # Reshape to (1, 1, lookback, features)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(x_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.item()

    def predict_proba(self, x_raw):
        """Predicts probability distributions."""
        self.model.eval()
        x_scaled = self.scale_data(x_raw)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(x_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1)
        return probabilities.squeeze(0).cpu().numpy()
