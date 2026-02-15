# 🤏 Tiny GPT – Building a Decoder-Only Transformer from Scratch

This project implements a GPT-style (decoder-only) Transformer language model completely from scratch using PyTorch.

Unlike fine-tuning pre-trained models, this project builds:

- Custom Multi-Head Self-Attention
- Transformer Decoder Blocks
- Token + Positional Embeddings
- Causal Masking
- Text Generation with Top-k Sampling
- Training loop with Perplexity evaluation

The model is trained on a Wikipedia subset using a SentencePiece tokenizer.

---
# 📂 Dataset

The model was trained on a Wikipedia text corpus obtained from Kaggle.

🔗 **Kaggle Dataset Link:**

https://www.kaggle.com/datasets/abbbhishekkk/wikipedia-dump-database

---

## 🔹 Dataset Preparation

### Extract & Prepare Text File

```python
!unzip text.zip -d ./
!head -n 1500000 wikisent2.txt > wiki_small.txt
```

---

## 🧠 Model Architecture

- Model Type: Decoder-Only Transformer (GPT-style)
- Layers: 6 Transformer blocks
- Embedding Dimension: 320
- Heads: 5
- Feedforward Dimension: 1280
- Vocabulary Size: 12,000 (SentencePiece BPE)
- Block Size: 128
- Parameters: ~25M+

---

## 📊 Training Results

| Metric | Value |
|--------|--------|
| Epochs | 3 |
| Final Training Loss | **~2.45** |
| Final Perplexity | **~11.58** |
| Parameters | ~25M |

> Perplexity = exp(loss)

---

# ⚙️ Installation

```bash
pip install torch sentencepiece tqdm
```

---

# 🏗️ Model Definition (Core GPT Implementation)

### 🔹 Multi-Head Self Attention

```python
class Multihead(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        assert dim % heads == 0

        self.heads = heads
        self.head_dim = dim // heads

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, T, D = x.shape

        Q = self.wq(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        K = self.wk(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        V = self.wv(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.proj(out)
```

---

# 🔥 Training Loop (Core Logic)

```python
for epoch in range(epochs):

    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)

        loss = criterion(
            logits.view(-1, vocab_size),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    training_loss = total_loss / len(loader)
    train_ppl = math.exp(training_loss)

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {training_loss:.4f}")
    print(f"Train Perplexity: {train_ppl:.2f}")
```

---

# 🤖 Text Generation

```python
prompt = "Forest"

ids = sp.encode(prompt)
x = torch.tensor(ids).unsqueeze(0).to(device)

out = model.generate(
    x,
    max_new=100,
    temp=0.8,
    top_k=40
)

print(sp.decode(out[0].tolist()))
```

---

# 🧩 Features Implemented

- Causal Masking (Prevents future token leakage)
- Weight Tying (Embedding ↔ Output Layer)
- Top-k Sampling
- Temperature Scaling
- Gradient Clipping
- Perplexity Calculation
- Custom Dataset Class
- SentencePiece Tokenization

---

# 📁 Project Structure

```
.
├── tiny-gpt-from-scratch.ipynb
├── wiki_small.txt
├── wiki_spm.model
├── custom_gpt_epoch1.pt
├── custom_gpt_epoch2.pt
├── custom_gpt_epoch3.pt
├── requirments.txt
└── README.md
```

---

# 🎯 Learning Outcomes

This project demonstrates deep understanding of:

- Transformer Architecture
- Attention Mechanism Internals
- Language Modeling
- Tokenization (BPE)
- Sequence Masking
- Autoregressive Generation
- PyTorch Model Engineering

---

## ❤️ Sample Predictions

```text
Forest, he was given his best-school stroke with his best-selling album in a "Hiptes" and "PLise's last song as it was recorded.
A member of his best-known team, he was the youngest final round with a half-brother competition.
A member of his live live album, he became a member of the band in 1953.
```
---

## 👨‍💻 Author

Abhijith Babu
Passionate about ML & AI 🚀

📌 GitHub: [https://github.com/AbhijithBabu12]

📌 LinkedIn: [https://www.linkedin.com/in/abhijith-babu-856170201/]
