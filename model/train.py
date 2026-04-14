"""
Neural Network Training
========================
Այս սկրիպտը սահմանում է նեյրոնային ցանց, մարզում է այն և պահպանում կշիռները:
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# ─── Քայլ 1: Սահմանել նեյրոնային ցանցը ───────────────────────
class MoonClassifier(nn.Module):
    """
    Պարզ feedforward նեյրոնային ցանց:
    Ճարտարապետություն:
        Input (2 features) -> Hidden 1 (32 neurons, ReLU) -> Hidden 2 (16 neurons, ReLU) -> Output (1 neuron, Sigmoid)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Ուղիղ տարածում (Forward pass)"""
        return self.net(x)

# ─── Քայլ 2: Բեռնել տվյալները ────────────────────────────────────
def load_data(path="data/dataset.csv"):
    df = pd.read_csv(path)
    X = df[["x1", "x2"]].values
    y = df["label"].values
    
    # Բաժանել: 80% մարզման, 20% վալիդացիայի համար
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Փոխակերպել PyTorch tensor-ների
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    return X_train_t, y_train_t, X_val_t, y_val_t

# ─── Քայլ 3: Մարզման ցիկլ (Training Loop) ────────────────────────────────────
def train_model(epochs=200, lr=0.01):
    X_train, y_train, X_val, y_val = load_data()
    model = MoonClassifier()
    loss_fn = nn.BCELoss() # Binary Cross-Entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_preds = model(X_train)
        train_loss = loss_fn(train_preds, y_train)
        train_loss.backward()
        optimizer.step()

        # Հաշվարկել ճշտությունը (accuracy)
        train_acc = ((train_preds > 0.5).float() == y_train).float().mean().item()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = loss_fn(val_preds, y_val)
            val_acc = ((val_preds > 0.5).float() == y_val).float().mean().item()

        # Հուշում: Use .item() on loss tensors
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss.item())
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} — Loss: {train_loss.item():.4f} | Val Acc: {val_acc:.2%}")

    return model, history

if __name__ == "__main__":
    model, history = train_model()
    with open("model/history.json", "w") as f:
        json.dump(history, f)
    
    # Հուշում: torch.save(model.state_dict(), "path/to/model.pth")
    torch.save(model.state_dict(), "path/to/model.pth")
    print("Model saved to model/model.pth")