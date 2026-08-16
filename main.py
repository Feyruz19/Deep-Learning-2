import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
DATA_FILE = 'kredit_data.xlsx'



if not os.path.exists(DATA_FILE):
    DATA_FILE = os.path.join(os.path.dirname(__file__), 'kredit_data.xlsx')

data = pd.read_excel(DATA_FILE)

features = data[['Ayliq gelir (AZN)', 'Kredit bali', 'Borc (AZN)']].values
labels = data['Netice'].values

inputs = torch.tensor(features, dtype=torch.float32)
targets = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

INCOME_SCALE = 10000.0
SCORE_SCALE = 850.0
DEBT_SCALE = 10000.0

inputs_scaled = inputs.clone()
inputs_scaled[:, 0] /= INCOME_SCALE
inputs_scaled[:, 1] /= SCORE_SCALE
inputs_scaled[:, 2] /= DEBT_SCALE


class LoanApprovalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


net = LoanApprovalModel()
loss_fn = nn.BCELoss()
opt = optim.Adam(net.parameters(), lr=0.01)

TOTAL_EPOCHS = 2000

for epoch in range(TOTAL_EPOCHS):
    prediction = net(inputs_scaled)
    loss_value = loss_fn(prediction, targets)

    opt.zero_grad()
    loss_value.backward()
    opt.step()

    if (epoch + 1) % 400 == 0:
        print(f"Epoch {epoch + 1}/{TOTAL_EPOCHS}, Loss = {loss_value.item():.4f}")

with torch.no_grad():
    train_probs = net(inputs_scaled)
    train_preds = (train_probs >= 0.5).float()
    correct_count = (train_preds == targets).float().sum()
    acc = (correct_count / len(targets)) * 100
    print(f"Train Accuracy = {acc:.2f}%")

print("\nYeni musteriler ucun proqnozlar:")

applicants = torch.tensor([
    [2800.0, 540.0, 5500.0],
    [4000.0, 640.0, 3000.0],
    [7000.0, 790.0, 1000.0]
], dtype=torch.float32)

applicants_scaled = applicants.clone()
applicants_scaled[:, 0] /= INCOME_SCALE
applicants_scaled[:, 1] /= SCORE_SCALE
applicants_scaled[:, 2] /= DEBT_SCALE

names = ["A", "B", "C"]

with torch.no_grad():
    final_probs = net(applicants_scaled)
    final_preds = (final_probs >= 0.5).float()

    for idx in range(len(applicants)):
        verdict = "TESDIQLENDI" if final_preds[idx].item() == 1 else "TESDIQLENMEDI"
        print(f"Customer={names[idx]} | Approval_Prob={final_probs[idx].item():.4f} | Result={verdict}")