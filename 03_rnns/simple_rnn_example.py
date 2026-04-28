"""
Simple RNN example (PyTorch)
- Toy dataset: noisy sine waves. Given a sequence of length T, predict the next value (regression).
- Model: simple nn.RNN (vanilla) with a linear readout. Includes optional switch to LSTM/GRU.
- Training loop with train/val split, checkpointing of best val loss, and a small prediction plot saved to file.

Run:
    python3 03_rnns/simple_rnn_example.py

Requirements: torch, numpy, matplotlib
"""
import os
import random
import math
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ---------- Utilities

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- Toy dataset: sine sequences
class SineDataset(Dataset):
    def __init__(self, n_samples=2000, seq_len=50, noise=0.1):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.noise = noise
        self.data = []
        self.targets = []
        self._generate()

    def _generate(self):
        for _ in range(self.n_samples):
            A = np.random.uniform(0.5, 1.5)
            freq = np.random.uniform(0.8, 1.2)
            phase = np.random.uniform(0, 2 * math.pi)
            xs = np.arange(0, self.seq_len + 1)
            sig = A * np.sin((xs * freq * 2 * math.pi / self.seq_len) + phase)
            sig += np.random.normal(scale=self.noise, size=sig.shape)
            seq = sig[:-1].astype(np.float32)
            target = sig[-1].astype(np.float32)
            self.data.append(seq.reshape(-1, 1))  # (seq_len, 1)
            self.targets.append(np.array([target]))
        self.data = np.stack(self.data)
        self.targets = np.stack(self.targets)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# ---------- Model
class RNNPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, rnn_type='RNN'):
        super().__init__()
        rnn_type = rnn_type.upper()
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError('Unsupported rnn_type: ' + rnn_type)
        self.out = nn.Linear(hidden_size, 1)
        self.rnn_type = rnn_type

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, h = self.rnn(x)  # out: (batch, seq_len, hidden)
        last = out[:, -1, :]
        return self.out(last)


# ---------- Training / Eval

def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


# ---------- Main

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print('Using device:', device)

    ds = SineDataset(n_samples=args.n_samples, seq_len=args.seq_len, noise=args.noise)
    n_val = int(len(ds) * args.val_frac)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = RNNPredictor(input_size=1, hidden_size=args.hidden_size, num_layers=args.num_layers, rnn_type=args.rnn_type)
    model.to(device)

    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float('inf')
    os.makedirs('03_rnns/artifacts', exist_ok=True)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f'Epoch {epoch}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}')

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = f'03_rnns/artifacts/best_model_example.pth'
            torch.save(model.state_dict(), ckpt_path)

    # Save history
    with open('03_rnns/artifacts/history.json', 'w') as f:
        json.dump(history, f)

    # Quick plot of losses
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.title('Loss curves')
    plt.savefig('03_rnns/artifacts/loss_curves.png')
    print('Saved artifacts to 03_rnns/artifacts/')

    # Load best model and show a few predictions
    model.load_state_dict(torch.load('03_rnns/artifacts/best_model_example.pth', map_location=device))
    model.to('cpu')
    model.eval()

    sample_x, sample_y = ds.data[:6], ds.targets[:6]
    with torch.no_grad():
        xb = torch.from_numpy(sample_x).float()
        pred = model(xb).numpy().squeeze()

    plt.figure(figsize=(8, 4))
    t = np.arange(args.seq_len)
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.plot(t, sample_x[i].squeeze(), '-o', label='input')
        plt.axhline(sample_y[i].item(), color='C1', label='true next')
        plt.axhline(pred[i].item(), color='C2', linestyle='--', label='pred next')
        plt.legend()
    plt.tight_layout()
    plt.savefig('03_rnns/artifacts/prediction_example.png')
    print('Saved prediction plot to 03_rnns/artifacts/prediction_example.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--seq_len', type=int, default=40)
    parser.add_argument('--noise', type=float, default=0.08)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--rnn_type', type=str, default='RNN', help='RNN | LSTM | GRU')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--use_cuda', action='store_true')
    args = parser.parse_args()
    main(args)
