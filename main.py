import os

import pandas as pd
import torch
import torch.nn as nn
from torch import optim

FILE_ADI = 'kredit_data.xlsx'

if not os.path.exists(FILE_ADI):
    FILE_ADI = os.path.join(os.path.dirname(__file__), 'kredit_data.xlsx')

df = pd.read_excel(FILE_ADI)

x = df[['Ayliq gelir (AZN)', 'Kredit bali', 'Borc (AZN)']].values
y = df['Netice'].values


x_t = torch.tensor(x, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)


max_gelir = 10000.0
max_bal = 850.0
max_borc = 10000.0

x_norm = x_t.clone()
x_norm[:, 0] /= max_gelir
x_norm[:, 1] /= max_bal
x_norm[:, 2] /= max_borc


class KreditModeli(nn.Module):
    def __init__(self):
        super(KreditModeli, self).__init__()
        self.layer1 = nn.Linear(3, 8)
        self.layer2 = nn.Linear(8, 1)

    def forward(self, giris):
        ara = torch.relu(self.layer1(giris))
        cixis = torch.sigmoid(self.layer2(ara))
        return cixis


model = KreditModeli()
itki_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

EPOX = 2000

for ep in range(EPOX):
    tahmin = model(x_norm)
    itki = itki_fn(tahmin, y_t)

    optimizer.zero_grad()
    itki.backward()
    optimizer.step()

    if (ep + 1) % 400 == 0:
        print(f"Epoch {ep + 1}/{EPOX}, Loss = {itki.item():.4f}")

with torch.no_grad():
    problar = model(x_norm)
    proqnoz = (problar >= 0.5).float()
    dogru = (proqnoz == y_t).float().sum()
    deqiqlik = (dogru / len(y_t)) * 100
    print(f"Train Accuracy = {deqiqlik:.2f}%")

print("\nYeni musteri proqnozlari:")

yeni = torch.tensor([
    [2800.0, 540.0, 5500.0],
    [4000.0, 640.0, 3000.0],
    [7000.0, 790.0, 1000.0]
], dtype=torch.float32)

yeni_norm = yeni.clone()
yeni_norm[:, 0] /= max_gelir
yeni_norm[:, 1] /= max_bal
yeni_norm[:, 2] /= max_borc

musteriler = ["A", "B", "C"]

with torch.no_grad():
    n_problar = model(yeni_norm)
    n_proqnoz = (n_problar >= 0.5).float()

    for i in range(len(yeni)):
        netice = "TESDIQLENDI" if n_proqnoz[i].item() == 1 else "TESDIQLENMEDI"
        print(f"Customer={musteriler[i]} | Approval_Prob={n_problar[i].item():.4f} | Result={netice}")
