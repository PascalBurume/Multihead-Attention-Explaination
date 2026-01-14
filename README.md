# 🧠 MULTI-HEAD ATTENTION: COMPLETE IMPLEMENTATION GUIDE
## For Pascal - Use this for homework!

---

# PART 1: THE CONCEPT (30 seconds review)

```
MULTI-HEAD ATTENTION = Split → Attend → Concat → Project

d_model = num_heads × head_dim
  768   =    12     ×    64
```

---

# PART 2: THE COMPLETE CODE TEMPLATE

## 📝 Copy This for Homework:

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Causal Mask
    Use this for homework!
    """
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        
        # ===== STEP 0: Save configuration =====
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 768 // 12 = 64
        
        # ===== STEP 1: Create weight matrices =====
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        
        # ===== STEP 2: Create dropout and causal mask =====
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        # ===== Get input dimensions =====
        b, num_tokens, d_in = x.shape
        
        # ===== STEP 3: Project to Q, K, V =====
        # Shape: (b, num_tokens, d_out)
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)
        
        # ===== STEP 4: Split into heads =====
        # (b, num_tokens, d_out) → (b, num_tokens, num_heads, head_dim)
        Q = Q.view(b, num_tokens, self.num_heads, self.head_dim)
        K = K.view(b, num_tokens, self.num_heads, self.head_dim)
        V = V.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # (b, num_tokens, num_heads, head_dim) → (b, num_heads, num_tokens, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # ===== STEP 5: Compute attention scores =====
        # (b, num_heads, num_tokens, head_dim) @ (b, num_heads, head_dim, num_tokens)
        # → (b, num_heads, num_tokens, num_tokens)
        attn_scores = Q @ K.transpose(2, 3)
        
        # ===== STEP 6: Apply causal mask =====
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # ===== STEP 7: Scale and softmax =====
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # ===== STEP 8: Apply attention to values =====
        # (b, num_heads, num_tokens, num_tokens) @ (b, num_heads, num_tokens, head_dim)
        # → (b, num_heads, num_tokens, head_dim)
        context = attn_weights @ V
        
        # ===== STEP 9: Concat heads =====
        # (b, num_heads, num_tokens, head_dim) → (b, num_tokens, num_heads, head_dim)
        context = context.transpose(1, 2)
        
        # (b, num_tokens, num_heads, head_dim) → (b, num_tokens, d_out)
        context = context.contiguous().view(b, num_tokens, self.d_out)
        
        # ===== STEP 10: Output projection =====
        output = self.out_proj(context)
        
        return output
```

---

# PART 3: HOW TO USE IT

```python
# Configuration (GPT-2 small style)
d_model = 768        # Embedding dimension
num_heads = 12       # Number of attention heads
context_length = 1024  # Maximum sequence length
dropout = 0.1        # Dropout rate
batch_size = 4       # Batch size
seq_len = 100        # Actual sequence length

# Create the model
mha = MultiHeadAttention(
    d_in=d_model,
    d_out=d_model,
    context_length=context_length,
    dropout=dropout,
    num_heads=num_heads,
    qkv_bias=False
)

# Create dummy input
x = torch.randn(batch_size, seq_len, d_model)

# Forward pass
output = mha(x)

print(f"Input shape:  {x.shape}")      # (4, 100, 768)
print(f"Output shape: {output.shape}") # (4, 100, 768)
```

---

# PART 4: STEP-BY-STEP DIMENSION TRACKING

## 🔑 This is the KEY to understanding!

```
INPUT: x
Shape: (batch, seq, d_model) = (4, 100, 768)

STEP 3: Project to Q, K, V
├── Q = W_query(x)
├── K = W_key(x)  
├── V = W_value(x)
└── Shape: (4, 100, 768)

STEP 4: Split into heads
├── .view(b, seq, num_heads, head_dim)
│   Shape: (4, 100, 12, 64)
├── .transpose(1, 2)
│   Shape: (4, 12, 100, 64)
└── Now: (batch, HEADS, seq, head_dim)

STEP 5: Attention scores
├── Q @ K.transpose(2, 3)
├── (4, 12, 100, 64) @ (4, 12, 64, 100)
└── Shape: (4, 12, 100, 100)  ← Each head has 100×100 attention matrix!

STEP 6-7: Mask, Scale, Softmax
└── Shape: (4, 12, 100, 100)  ← Same, just normalized

STEP 8: Apply to Values
├── attn_weights @ V
├── (4, 12, 100, 100) @ (4, 12, 100, 64)
└── Shape: (4, 12, 100, 64)

STEP 9: Concat heads
├── .transpose(1, 2)
│   Shape: (4, 100, 12, 64)
├── .view(b, seq, d_out)
│   Shape: (4, 100, 768)
└── Back to original shape!

STEP 10: Output projection
└── Shape: (4, 100, 768)  ← Same as input!
```

---

# PART 5: THE 4 KEY OPERATIONS EXPLAINED

## 1️⃣ `.view()` - Reshape without copying

```python
# Example: Split 768 into 12 heads of 64
x = torch.randn(4, 100, 768)

# 768 → 12 × 64
x = x.view(4, 100, 12, 64)

# Think of it as: [||||||||] → [|  |  |  |  |  |  |  |  |  |  |  |  |]
#                   768          64 64 64 64 64 64 64 64 64 64 64 64
```

## 2️⃣ `.transpose(dim1, dim2)` - Swap dimensions

```python
x = torch.randn(4, 100, 12, 64)
#                b  seq heads dim

x = x.transpose(1, 2)  # Swap position 1 and 2
#                b  heads seq  dim
# Shape: (4, 12, 100, 64)
```

## 3️⃣ `@` (Matrix Multiply) - The attention computation

```python
Q = torch.randn(4, 12, 100, 64)  # (batch, heads, seq, head_dim)
K = torch.randn(4, 12, 100, 64)

# Q @ K.transpose(2, 3) means:
# (4, 12, 100, 64) @ (4, 12, 64, 100) → (4, 12, 100, 100)
#                           ↑ transposed last 2 dims

attn_scores = Q @ K.transpose(2, 3)
```

## 4️⃣ `.contiguous().view()` - Safe reshape after transpose

```python
# After transpose, memory might not be contiguous
# .contiguous() makes it contiguous before .view()

x = x.transpose(1, 2)          # Memory not contiguous
x = x.contiguous()             # Make contiguous
x = x.view(b, seq, d_out)      # Now safe to reshape

# Often written as one line:
x = x.transpose(1, 2).contiguous().view(b, seq, d_out)
```

---

# PART 6: COMMON HOMEWORK QUESTIONS

## Q1: "What is the purpose of each weight matrix?"

| Matrix | Purpose | Analogy |
|--------|---------|---------|
| `W_query` | Create search queries | "What am I looking for?" |
| `W_key` | Create keys for matching | "What do I contain?" |
| `W_value` | Create values to retrieve | "What info do I provide?" |
| `out_proj` | Combine all head outputs | "Mix everything together" |

## Q2: "Why do we scale by sqrt(head_dim)?"

```python
attn_scores = attn_scores / (self.head_dim ** 0.5)
```

**Answer:** 
- Large dot products → extreme softmax values → tiny gradients
- Scaling keeps values in reasonable range
- head_dim=64 → scale by √64 = 8

## Q3: "What does the causal mask do?"

```python
# Creates upper triangular matrix of 1s
mask = torch.triu(torch.ones(seq, seq), diagonal=1)

# Fills masked positions with -infinity
attn_scores.masked_fill_(mask.bool(), -torch.inf)

# After softmax: -inf → 0 (can't attend to future!)
```

## Q4: "Why multiple heads?"

**Answer:** Different heads learn different patterns:
- Head 1: Subject-verb relationships
- Head 2: Pronoun references  
- Head 3: Adjacent word context
- etc.

---

# PART 7: QUICK REFERENCE CARD

## Dimensions at Each Step:

| Step | Operation | Shape |
|------|-----------|-------|
| Input | x | (b, seq, d_model) |
| Project | Q, K, V = W(x) | (b, seq, d_model) |
| Split | .view() | (b, seq, heads, head_dim) |
| Reorder | .transpose(1,2) | (b, **heads**, seq, head_dim) |
| Scores | Q @ K.T | (b, heads, seq, **seq**) |
| Attend | weights @ V | (b, heads, seq, head_dim) |
| Reorder | .transpose(1,2) | (b, seq, heads, head_dim) |
| Concat | .view() | (b, seq, d_model) |
| Output | out_proj() | (b, seq, d_model) |

## Key Formulas:

```
head_dim = d_model / num_heads

Attention = softmax(QK^T / √head_dim) × V

MultiHead = Concat(head_1, ..., head_h) × W_out
```

---

# PART 8: MINIMAL VERSION (For Quick Reference)

```python
def forward(self, x):
    b, n, _ = x.shape
    
    # Project
    Q = self.W_q(x)
    K = self.W_k(x)
    V = self.W_v(x)
    
    # Split heads
    Q = Q.view(b, n, self.h, self.d).transpose(1, 2)
    K = K.view(b, n, self.h, self.d).transpose(1, 2)
    V = V.view(b, n, self.h, self.d).transpose(1, 2)
    
    # Attention
    scores = Q @ K.transpose(-2, -1) / (self.d ** 0.5)
    scores = scores.masked_fill(self.mask[:n, :n].bool(), -torch.inf)
    weights = torch.softmax(scores, dim=-1)
    
    # Combine
    out = (weights @ V).transpose(1, 2).contiguous().view(b, n, self.d_out)
    return self.proj(out)
```

---

# PART 9: TESTING YOUR IMPLEMENTATION

```python
# Test code - add this to verify your implementation works
def test_mha():
    # Setup
    d_model, num_heads = 768, 12
    mha = MultiHeadAttention(d_model, d_model, 1024, 0.0, num_heads)
    x = torch.randn(2, 50, d_model)
    
    # Forward
    out = mha(x)
    
    # Checks
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN!"
    print("✅ All tests passed!")

test_mha()
```

---

# SUMMARY: What to Remember for Homework

1. **The 10 steps:** Project → Split → Transpose → Attention → Concat → Project
2. **The key equation:** `d_model = num_heads × head_dim`
3. **The 4 operations:** `.view()`, `.transpose()`, `@`, `.contiguous()`
4. **Input/Output:** Same shape! (batch, seq, d_model)

Good luck with your homework, Pascal! 🚀


# 🧠 MULTI-HEAD ATTENTION: COMPLETE IMPLEMENTATION GUIDE
## For Pascal - Use this for homework!

---

# 📌 QUICK SUMMARY (30 seconds)

## The ONE Sentence:
> **Multi-Head Attention = Split → Attend → Concat**
> (Multiple detectives looking at the same sentence, each finding different clues)

## The ONE Formula:
```
d_model = num_heads × head_dim
  768   =    12     ×    64
```

## The 6 Steps:
```
PROJECT → SPLIT → TRANSPOSE → ATTENTION → CONCAT → PROJECT
   1        2         3           4          5         6
```

---

# 📊 THE 6 STEPS EXPLAINED

## Visual Overview:

```
┌─────────┐   ┌─────────┐   ┌───────────┐   ┌───────────┐   ┌─────────┐   ┌─────────┐
│ PROJECT │ → │  SPLIT  │ → │ TRANSPOSE │ → │ ATTENTION │ → │ CONCAT  │ → │ PROJECT │
└─────────┘   └─────────┘   └───────────┘   └───────────┘   └─────────┘   └─────────┘
     1             2              3               4              5             6
```

---

## STEP 1: PROJECT (Create Q, K, V)

### What happens:
Multiply input by 3 weight matrices to create Query, Key, Value

### Visual:
```
INPUT x: (batch, seq, 768)
              │
              ▼
    ┌─────────────────────┐
    │   x @ W_query → Q   │
    │   x @ W_key   → K   │
    │   x @ W_value → V   │
    └─────────────────────┘
              │
              ▼
Q, K, V: (batch, seq, 768)
```

### Code:
```python
Q = self.W_query(x)  # (batch, seq, 768)
K = self.W_key(x)    # (batch, seq, 768)
V = self.W_value(x)  # (batch, seq, 768)
```

### Shape Change:
```
(batch, seq, 768) → (batch, seq, 768)  # Same shape, different content
```

---

## STEP 2: SPLIT (Divide into heads)

### What happens:
Reshape 768 → 12 heads × 64 dimensions each

### Visual:
```
BEFORE: Q is (batch, seq, 768)
        One long vector of 768

        [████████████████████████████████████████████]
                         768

AFTER:  Q is (batch, seq, 12, 64)
        Split into 12 heads, each with 64 dimensions

        [████] [████] [████] [████] [████] [████] [████] [████] [████] [████] [████] [████]
         64     64     64     64     64     64     64     64     64     64     64     64
        head1  head2  head3  head4  head5  head6  head7  head8  head9  head10 head11 head12
```

### Code:
```python
Q = Q.view(batch, seq, num_heads, head_dim)
# (batch, seq, 768) → (batch, seq, 12, 64)
```

### Shape Change:
```
(batch, seq, 768) → (batch, seq, 12, 64)
```

---

## STEP 3: TRANSPOSE (Reorder dimensions)

### What happens:
Move "heads" dimension to position 1 (so we can do batch matrix multiply)

### Why:
PyTorch can then compute ALL 12 heads in parallel!

### Visual:
```
BEFORE: (batch, seq, heads, head_dim)
        (  2,  100,   12,     64   )
               ↑      ↑
              pos 1  pos 2

AFTER:  (batch, heads, seq, head_dim)
        (  2,    12,  100,    64   )
                 ↑     ↑
               pos 1  pos 2   ← SWAPPED!
```

### Code:
```python
Q = Q.transpose(1, 2)
# (batch, seq, heads, head_dim) → (batch, heads, seq, head_dim)
```

### Shape Change:
```
(batch, seq, 12, 64) → (batch, 12, seq, 64)
```

---

## STEP 4: ATTENTION (The main computation)

### What happens:
This step has 4 sub-steps:

### Visual:
```
┌─────────────────────────────────────────────────────────────────┐
│                      STEP 4: ATTENTION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  4a. SCORES: Q @ K^T                                           │
│      (batch, heads, seq, 64) @ (batch, heads, 64, seq)         │
│      → (batch, heads, seq, seq)                                │
│                                                                 │
│  4b. MASK: Hide future tokens                                  │
│      scores.masked_fill_(mask, -infinity)                      │
│                                                                 │
│  4c. SCALE + SOFTMAX: Normalize                                │
│      scores = scores / sqrt(64)                                │
│      weights = softmax(scores)                                 │
│                                                                 │
│  4d. APPLY TO VALUES: weights @ V                              │
│      (batch, heads, seq, seq) @ (batch, heads, seq, 64)        │
│      → (batch, heads, seq, 64)                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Code:
```python
# 4a. Scores
attn_scores = Q @ K.transpose(2, 3)   # (batch, heads, seq, seq)

# 4b. Mask
attn_scores.masked_fill_(mask, -torch.inf)

# 4c. Scale + Softmax  
attn_scores = attn_scores / (head_dim ** 0.5)
attn_weights = torch.softmax(attn_scores, dim=-1)

# 4d. Apply to values
context = attn_weights @ V            # (batch, heads, seq, head_dim)
```

### Shape Changes:
```
4a: (batch, 12, seq, 64) @ (batch, 12, 64, seq) → (batch, 12, seq, seq)
4b: (batch, 12, seq, seq) → (batch, 12, seq, seq)  # Same shape, masked
4c: (batch, 12, seq, seq) → (batch, 12, seq, seq)  # Same shape, normalized
4d: (batch, 12, seq, seq) @ (batch, 12, seq, 64) → (batch, 12, seq, 64)
```

---

## STEP 5: CONCAT (Put heads back together)

### What happens:
Reverse of Split - combine all heads back into one

### Visual:
```
BEFORE: 12 separate heads, each (batch, heads, seq, 64)

        [Head1] [Head2] [Head3] ... [Head12]
          64      64      64          64

                         │
                         ▼
        
        Step 5a: Transpose back
        (batch, heads, seq, 64) → (batch, seq, heads, 64)
        
                         │
                         ▼
        
        Step 5b: View/Reshape
        (batch, seq, 12, 64) → (batch, seq, 768)

AFTER:  One combined vector of 768

        [████████████████████████████████████████████]
                         768
```

### Code:
```python
context = context.transpose(1, 2)                    # (batch, seq, heads, head_dim)
context = context.contiguous().view(batch, seq, 768) # (batch, seq, 768)
```

### Shape Change:
```
(batch, 12, seq, 64) → (batch, seq, 12, 64) → (batch, seq, 768)
```

---

## STEP 6: PROJECT (Final output)

### What happens:
One more linear layer to mix information from all heads

### Visual:
```
INPUT:  (batch, seq, 768)
              │
              ▼
    ┌─────────────────────┐
    │  context @ W_output │
    └─────────────────────┘
              │
              ▼
OUTPUT: (batch, seq, 768)
```

### Code:
```python
output = self.out_proj(context)  # (batch, seq, 768)
```

### Shape Change:
```
(batch, seq, 768) → (batch, seq, 768)  # Same shape
```

---

# 📊 COMPLETE PICTURE (All 6 Steps)

```
INPUT (batch, seq, 768)
         │
         ▼
┌─────────────────────┐
│  1. PROJECT         │  Q = W_q(x), K = W_k(x), V = W_v(x)
│     Create Q, K, V  │  
└─────────────────────┘
         │
         ▼ (batch, seq, 768)
┌─────────────────────┐
│  2. SPLIT           │  .view(batch, seq, 12, 64)
│     Into 12 heads   │  
└─────────────────────┘
         │
         ▼ (batch, seq, 12, 64)
┌─────────────────────┐
│  3. TRANSPOSE       │  .transpose(1, 2)
│     Move heads dim  │  
└─────────────────────┘
         │
         ▼ (batch, 12, seq, 64)
┌─────────────────────┐
│  4. ATTENTION       │  
│    4a. Q @ K^T      │  → scores
│    4b. Mask future  │  → masked scores
│    4c. Scale+Softmax│  → weights
│    4d. weights @ V  │  → context
└─────────────────────┘
         │
         ▼ (batch, 12, seq, 64)
┌─────────────────────┐
│  5. CONCAT          │  .transpose(1,2).view(batch, seq, 768)
│     Combine heads   │  
└─────────────────────┘
         │
         ▼ (batch, seq, 768)
┌─────────────────────┐
│  6. PROJECT         │  out_proj(context)
│     Final mixing    │  
└─────────────────────┘
         │
         ▼
OUTPUT (batch, seq, 768)
```

---

# 📋 SUMMARY TABLE

| Step | Name | Operation | Shape Change |
|------|------|-----------|--------------|
| 1 | PROJECT | `W_q(x), W_k(x), W_v(x)` | (b,s,768) → (b,s,768) |
| 2 | SPLIT | `.view(b,s,12,64)` | (b,s,768) → (b,s,12,64) |
| 3 | TRANSPOSE | `.transpose(1,2)` | (b,s,12,64) → (b,12,s,64) |
| 4 | ATTENTION | `softmax(QK^T/√d) @ V` | (b,12,s,64) → (b,12,s,64) |
| 5 | CONCAT | `.transpose(1,2).view()` | (b,12,s,64) → (b,s,768) |
| 6 | PROJECT | `out_proj()` | (b,s,768) → (b,s,768) |

---

# 💻 COMPLETE CODE TEMPLATE

## Copy this for homework:

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Causal Mask
    Based on Chapter 3 of 'Build a Large Language Model From Scratch'
    """
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        
        # Validate: d_out must be divisible by num_heads
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        # Save configuration
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 768 // 12 = 64
        
        # STEP 1 weights: Q, K, V projections
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # STEP 6 weights: Output projection
        self.out_proj = nn.Linear(d_out, d_out)
        
        # Dropout and causal mask
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        # ========== STEP 1: PROJECT ==========
        # Create Q, K, V
        Q = self.W_query(x)  # (b, seq, d_out)
        K = self.W_key(x)
        V = self.W_value(x)
        
        # ========== STEP 2: SPLIT ==========
        # (b, seq, d_out) → (b, seq, num_heads, head_dim)
        Q = Q.view(b, num_tokens, self.num_heads, self.head_dim)
        K = K.view(b, num_tokens, self.num_heads, self.head_dim)
        V = V.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # ========== STEP 3: TRANSPOSE ==========
        # (b, seq, num_heads, head_dim) → (b, num_heads, seq, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # ========== STEP 4: ATTENTION ==========
        # 4a. Compute attention scores: Q @ K^T
        attn_scores = Q @ K.transpose(2, 3)  # (b, heads, seq, seq)
        
        # 4b. Apply causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # 4c. Scale and softmax
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 4d. Apply attention to values
        context = attn_weights @ V  # (b, heads, seq, head_dim)
        
        # ========== STEP 5: CONCAT ==========
        # (b, heads, seq, head_dim) → (b, seq, heads, head_dim) → (b, seq, d_out)
        context = context.transpose(1, 2)
        context = context.contiguous().view(b, num_tokens, self.d_out)
        
        # ========== STEP 6: PROJECT ==========
        output = self.out_proj(context)
        
        return output
```

---

# 🧪 HOW TO USE IT

```python
# Configuration (GPT-2 small style)
d_model = 768          # Embedding dimension
num_heads = 12         # Number of attention heads  
context_length = 1024  # Maximum sequence length
dropout = 0.1          # Dropout rate

# Create the model
mha = MultiHeadAttention(
    d_in=d_model,
    d_out=d_model,
    context_length=context_length,
    dropout=dropout,
    num_heads=num_heads,
    qkv_bias=False
)

# Create dummy input
batch_size = 4
seq_len = 100
x = torch.randn(batch_size, seq_len, d_model)

# Forward pass
output = mha(x)

print(f"Input shape:  {x.shape}")      # (4, 100, 768)
print(f"Output shape: {output.shape}") # (4, 100, 768)
```

---

# 🔑 THE 4 KEY OPERATIONS EXPLAINED

## 1️⃣ `.view()` - Reshape without copying

```python
# Split 768 into 12 heads of 64
x = torch.randn(4, 100, 768)
x = x.view(4, 100, 12, 64)

# Think: [████████] → [██] [██] [██] [██]...
#           768        64   64   64   64
```

## 2️⃣ `.transpose(dim1, dim2)` - Swap dimensions

```python
x = torch.randn(4, 100, 12, 64)
#                b  seq heads dim

x = x.transpose(1, 2)
#                b  heads seq  dim
# Shape: (4, 12, 100, 64)
```

## 3️⃣ `@` (Matrix Multiply) - The attention computation

```python
Q = torch.randn(4, 12, 100, 64)
K = torch.randn(4, 12, 100, 64)

# Q @ K^T
attn_scores = Q @ K.transpose(2, 3)
# (4, 12, 100, 64) @ (4, 12, 64, 100) → (4, 12, 100, 100)
```

## 4️⃣ `.contiguous().view()` - Safe reshape after transpose

```python
# After transpose, memory may not be contiguous
x = x.transpose(1, 2)          # Memory not contiguous
x = x.contiguous()             # Make contiguous
x = x.view(b, seq, d_out)      # Now safe to reshape

# Often written as:
x = x.transpose(1, 2).contiguous().view(b, seq, d_out)
```

---

# ❓ COMMON HOMEWORK QUESTIONS

## Q1: "What is the purpose of each weight matrix?"

| Matrix | Purpose | Analogy |
|--------|---------|---------|
| `W_query` | Create search queries | "What am I looking for?" |
| `W_key` | Create keys for matching | "What do I contain?" |
| `W_value` | Create values to retrieve | "What info do I provide?" |
| `out_proj` | Combine all head outputs | "Mix everything together" |

## Q2: "Why do we scale by sqrt(head_dim)?"

```python
attn_scores = attn_scores / (self.head_dim ** 0.5)
```

**Answer:** 
- Large dot products → extreme softmax values → tiny gradients
- Scaling keeps values in reasonable range
- head_dim=64 → scale by √64 = 8

## Q3: "What does the causal mask do?"

```python
mask = torch.triu(torch.ones(seq, seq), diagonal=1)
attn_scores.masked_fill_(mask.bool(), -torch.inf)
```

**Answer:**
- Creates upper triangular matrix of 1s
- Fills future positions with -infinity
- After softmax: -inf → 0 (can't attend to future!)

## Q4: "Why multiple heads?"

**Answer:** Different heads learn different patterns:
- Head 1: Subject-verb relationships
- Head 2: Pronoun references  
- Head 3: Adjacent word context
- Head 4: Sentence structure
- etc.

---

# 🎯 MEMORY TRICKS

## The Octopus 🐙
```
        🐙 OCTOPUS
       /  |  |  \
      /   |  |   \
     🦑  🦑  🦑  🦑   ← 8 arms = 8 heads
     │   │   │   │
     ▼   ▼   ▼   ▼
   Each arm grabs
   different things!
         │
         ▼
    🧠 Brain combines
    all information
```

## Easy Phrase
> **"Project, Split, Transpose, Attend, Concat, Project"**

Or shorter: **"PSTACP"** (Project-Split-Transpose-Attention-Concat-Project)

---

# ✅ FINAL CHECKLIST FOR HOMEWORK

- [ ] Understand the 6 steps
- [ ] Know the formula: `d_model = num_heads × head_dim`
- [ ] Know the 4 operations: `.view()`, `.transpose()`, `@`, `.contiguous()`
- [ ] Input shape = Output shape: `(batch, seq, d_model)`
- [ ] Can explain: Why scale? Why mask? Why multiple heads?

---

# 📊 REAL-WORLD MODEL CONFIGURATIONS

| Model | d_model | num_heads | head_dim |
|-------|---------|-----------|----------|
| GPT-2 Small | 768 | 12 | 64 |
| GPT-2 Medium | 1024 | 16 | 64 |
| GPT-2 Large | 1280 | 20 | 64 |
| GPT-2 XL | 1600 | 25 | 64 |
| GPT-3 | 12288 | 96 | 128 |

---

Good luck with your homework, Pascal! 🚀
