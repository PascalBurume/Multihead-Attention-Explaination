# 🧠 Attention Mechanisms: Complete Implementation Guide

> **Author:** Pascal Burume  
> **Purpose:** Research Reference for LLM Implementation  
> **Based On:** "Build a Large Language Model From Scratch" by Sebastian Raschka (Chapter 3)  
> **Last Updated:** January 2025

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [The Attention Formula](#the-attention-formula)
4. [Implementation Roadmap](#implementation-roadmap)
5. [Variant 1: Simplified Self-Attention](#variant-1-simplified-self-attention)
6. [Variant 2: Self-Attention with Trainable Weights](#variant-2-self-attention-with-trainable-weights)
7. [Variant 3: Causal Attention](#variant-3-causal-attention)
8. [Variant 4: Multi-Head Attention](#variant-4-multi-head-attention) ⭐
9. [Complete Production Code](#complete-production-code)
10. [GPT-2 Specifications](#gpt-2-specifications)
11. [Common Errors & Solutions](#common-errors--solutions)
12. [Quick Reference Cheat Sheet](#quick-reference-cheat-sheet)

---

## Overview

### What is Attention?

Attention is a mechanism that allows a model to **focus on relevant parts** of the input when producing an output. Instead of treating all input tokens equally, attention learns which tokens are most important for each position.

### Why Attention Matters

| Problem with RNNs | Solution with Attention |
|-------------------|------------------------|
| Sequential processing (slow) | Parallel processing (fast) |
| Forgets long-range dependencies | Direct access to all positions |
| Fixed context representation | Dynamic, query-specific context |

### The Four Variants (Progressive Complexity)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Simplified    │ ──▶ │  Self-Attention │ ──▶ │     Causal      │ ──▶ │   Multi-Head    │
│ Self-Attention  │     │   (Trainable)   │     │    Attention    │     │    Attention    │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ No learnable    │     │ Adds W_q, W_k,  │     │ Masks future    │     │ Multiple        │
│ weights         │     │ W_v matrices    │     │ tokens          │     │ parallel heads  │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Key Concepts

### Query, Key, Value (Q, K, V)

| Component | Symbol | Description | Analogy |
|-----------|--------|-------------|---------|
| **Query** | Q | "What am I looking for?" | Search query in Google |
| **Key** | K | "What do I contain?" | Page titles/metadata |
| **Value** | V | "What information do I provide?" | Actual page content |

### How Attention Works (Intuition)

```
Input: "Your journey starts with one step"

For the word "journey" (query):
1. Compare "journey" with ALL words (using Q·K)
2. Get similarity scores: [0.14, 0.24, 0.23, 0.13, 0.11, 0.16]
3. These scores = attention weights (sum to 1.0)
4. Weighted sum of all values = context vector for "journey"

Result: "journey" now contains information from ALL words,
        weighted by relevance!
```

### Dimensions Explained

```python
# Input dimensions
batch_size = 2          # Number of sequences in a batch
seq_length = 6          # Number of tokens per sequence (context length)
d_in = 768              # Input embedding dimension
d_out = 768             # Output embedding dimension (often d_in == d_out)

# Multi-head dimensions
num_heads = 12          # Number of attention heads
head_dim = d_out // num_heads  # Dimension per head (768/12 = 64)

# Tensor shapes through the pipeline:
# Input:        (batch_size, seq_length, d_in)      = (2, 6, 768)
# After Q/K/V:  (batch_size, seq_length, d_out)     = (2, 6, 768)
# Split heads:  (batch_size, num_heads, seq_length, head_dim) = (2, 12, 6, 64)
# Output:       (batch_size, seq_length, d_out)     = (2, 6, 768)
```

---

## The Attention Formula

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### Breaking Down the Formula

```
Step 1: QK^T
        ─────
        Compute similarity between all query-key pairs
        Shape: (seq_len, d_k) @ (d_k, seq_len) = (seq_len, seq_len)

Step 2: ÷ √d_k
        ──────
        Scale down to prevent extreme softmax values
        Keeps gradients stable during training

Step 3: softmax(...)
        ────────────
        Convert scores to probabilities (sum to 1 per row)
        
Step 4: × V
        ───
        Weighted sum of values using attention weights
        Shape: (seq_len, seq_len) @ (seq_len, d_v) = (seq_len, d_v)
```

### Why Scale by √d_k?

```python
# Without scaling:
d_k = 64
# Dot products grow with dimension: mean ≈ 0, variance ≈ d_k
# Large values → softmax becomes nearly one-hot → tiny gradients

# With scaling:
# Divide by √64 = 8 → variance ≈ 1 → stable softmax → healthy gradients
```

---

## Implementation Roadmap

```
┌────────────────────────────────────────────────────────────────────┐
│                    IMPLEMENTATION CHECKLIST                        │
├────────────────────────────────────────────────────────────────────┤
│ □ Step 1: Understand simplified attention (no weights)             │
│ □ Step 2: Add trainable W_q, W_k, W_v matrices                     │
│ □ Step 3: Implement scaling (÷ √d_k)                               │
│ □ Step 4: Add causal mask for autoregressive generation            │
│ □ Step 5: Add dropout for regularization                           │
│ □ Step 6: Extend to multi-head attention                           │
│ □ Step 7: Add output projection layer                              │
│ □ Step 8: Test with batch inputs                                   │
└────────────────────────────────────────────────────────────────────┘
```

---

## Variant 1: Simplified Self-Attention

> **Purpose:** Understand the core concept without trainable weights

### Code

```python
import torch

# Sample input: "Your journey starts with one step"
# Each word is a 3D embedding vector
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your    (x^1)
    [0.55, 0.87, 0.66],  # journey (x^2)
    [0.57, 0.85, 0.64],  # starts  (x^3)
    [0.22, 0.58, 0.33],  # with    (x^4)
    [0.77, 0.25, 0.10],  # one     (x^5)
    [0.05, 0.80, 0.55]   # step    (x^6)
])

# Step 1: Compute attention scores (dot products)
# Query = inputs[1] ("journey")
query = inputs[1]
attn_scores = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores[i] = torch.dot(x_i, query)
# Result: tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

# Step 2: Normalize with softmax
attn_weights = torch.softmax(attn_scores, dim=0)
# Result: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
# Sum = 1.0

# Step 3: Compute context vector (weighted sum)
context_vec = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec += attn_weights[i] * x_i
# Result: tensor([0.4419, 0.6515, 0.5683])

# ═══════════════════════════════════════════════════════════════
# EFFICIENT VERSION: Compute ALL context vectors at once
# ═══════════════════════════════════════════════════════════════
attn_scores = inputs @ inputs.T          # (6, 6) attention matrix
attn_weights = torch.softmax(attn_scores, dim=-1)  # Normalize rows
all_context_vecs = attn_weights @ inputs  # (6, 3) all context vectors
```

### Key Takeaways

- ✅ Simple dot product measures similarity
- ✅ Softmax converts scores to probabilities
- ✅ Context vector = weighted sum of all inputs
- ❌ No learnable parameters (can't train!)

---

## Variant 2: Self-Attention with Trainable Weights

> **Purpose:** Add learnable parameters that improve during training

### Code

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """
    Self-attention with trainable weight matrices.
    
    Args:
        d_in: Input embedding dimension
        d_out: Output embedding dimension
        qkv_bias: Whether to include bias in Q, K, V projections
    """
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()  # Initialize parent class (nn.Module)
        
        # Trainable weight matrices
        # nn.Linear performs: output = input @ weight.T + bias
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_in)
        Returns:
            Context vectors of shape (batch_size, seq_length, d_out)
        """
        # Project inputs to Q, K, V spaces
        queries = self.W_query(x)  # (batch, seq_len, d_out)
        keys    = self.W_key(x)    # (batch, seq_len, d_out)
        values  = self.W_value(x)  # (batch, seq_len, d_out)
        
        # Compute attention scores
        attn_scores = queries @ keys.transpose(-2, -1)  # (batch, seq_len, seq_len)
        
        # Scale by √d_k for numerical stability
        d_k = keys.shape[-1]
        attn_scores = attn_scores / (d_k ** 0.5)
        
        # Convert to probabilities
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Compute context vectors
        context_vec = attn_weights @ values  # (batch, seq_len, d_out)
        
        return context_vec


# Usage example
d_in, d_out = 3, 2
model = SelfAttention(d_in, d_out)

# Input: (batch_size=1, seq_length=6, d_in=3)
x = inputs.unsqueeze(0)  # Add batch dimension
output = model(x)
print(output.shape)  # torch.Size([1, 6, 2])
```

### Key Takeaways

- ✅ W_query, W_key, W_value are learned during training
- ✅ Scaling by √d_k prevents gradient issues
- ✅ Input/output dimensions can differ (d_in ≠ d_out)
- ❌ Can still "see" future tokens (problematic for generation)

---

## Variant 3: Causal Attention

> **Purpose:** Mask future tokens for autoregressive (left-to-right) generation

### The Masking Problem

```
Standard Attention:              Causal Attention:
(Can see everything)             (Can only see past + current)

     Your journey starts         Your journey starts
Your  [✓]   [✓]    [✓]          [✓]   [✗]    [✗]    
journey[✓]   [✓]    [✓]          [✓]   [✓]    [✗]    
starts [✓]   [✓]    [✓]          [✓]   [✓]    [✓]    

Problem: When predicting         Solution: Mask out future
"starts", model can cheat        positions with -∞ before
by looking at future words!      applying softmax
```

### Code

```python
import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    """
    Causal (masked) self-attention for autoregressive models.
    
    Args:
        d_in: Input embedding dimension
        d_out: Output embedding dimension  
        context_length: Maximum sequence length
        dropout: Dropout probability
        qkv_bias: Whether to include bias in Q, K, V projections
    """
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        
        # Projection layers
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask: upper triangular matrix of 1s
        # register_buffer: saves with model but not trained
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_length, d_in)
        Returns:
            Context vectors (batch_size, seq_length, d_out)
        """
        batch_size, num_tokens, d_in = x.shape
        
        # Project to Q, K, V
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)
        
        # Compute attention scores
        attn_scores = queries @ keys.transpose(1, 2)  # (batch, seq, seq)
        
        # Apply causal mask BEFORE softmax
        # masked_fill_ replaces positions where mask==True with -inf
        # e^(-inf) = 0, so these positions get zero attention
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )
        
        # Scale and normalize
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1
        )
        
        # Apply dropout (only during training)
        attn_weights = self.dropout(attn_weights)
        
        # Compute context vectors
        context_vec = attn_weights @ values
        
        return context_vec


# Usage example
context_length = 1024  # Max sequence length
dropout = 0.1          # 10% dropout during training

model = CausalAttention(
    d_in=768,
    d_out=768,
    context_length=context_length,
    dropout=dropout
)
```

### Key Takeaways

- ✅ Causal mask prevents "cheating" during generation
- ✅ `register_buffer` saves mask with model (not trained)
- ✅ Dropout helps prevent overfitting
- ✅ `.masked_fill_()` is efficient (in-place operation)

---

## Variant 4: Multi-Head Attention

> ⭐ **This is the complete implementation used in GPT models!**

### Why Multiple Heads?

```
Single Head:                     Multi-Head:
├── Can only learn ONE           ├── Head 1: Syntactic patterns
│   type of relationship         ├── Head 2: Semantic similarity
│   at a time                    ├── Head 3: Positional relationships
│                                ├── Head 4: Coreference resolution
│                                └── ... (learns diverse patterns)
```

### Architecture Diagram

```
                    Input (batch, seq_len, d_model)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
           W_query         W_key          W_value
              │               │               │
              ▼               ▼               ▼
         Q (b,s,d)       K (b,s,d)       V (b,s,d)
              │               │               │
              └───────┬───────┴───────┬───────┘
                      │               │
              ┌───────▼───────┐       │
              │  Split into   │       │
              │   num_heads   │       │
              └───────┬───────┘       │
                      │               │
         ┌────────────┼────────────┐  │
         ▼            ▼            ▼  │
      Head 1       Head 2   ...  Head h
    (b,s,head_dim) (b,s,hd)     (b,s,hd)
         │            │            │
         └────────────┼────────────┘
                      │
              ┌───────▼───────┐
              │  Concatenate  │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │ Output Proj.  │
              └───────┬───────┘
                      │
                      ▼
              Output (batch, seq_len, d_model)
```

### Complete Implementation

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as used in GPT models.
    
    This implementation uses the efficient "split" approach:
    1. Project input to Q, K, V with single large matrices
    2. Reshape to split into multiple heads
    3. Compute attention for all heads in parallel
    4. Concatenate and project output
    
    Args:
        d_in: Input embedding dimension
        d_out: Output embedding dimension (must be divisible by num_heads)
        context_length: Maximum sequence length for causal mask
        dropout: Dropout probability for attention weights
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in Q, K, V projections
    
    Example:
        >>> mha = MultiHeadAttention(
        ...     d_in=768, d_out=768, context_length=1024,
        ...     dropout=0.1, num_heads=12
        ... )
        >>> x = torch.randn(2, 100, 768)  # (batch, seq_len, d_in)
        >>> output = mha(x)
        >>> output.shape
        torch.Size([2, 100, 768])
    """
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        
        # ═══════════════════════════════════════════════════════════
        # VALIDATION
        # ═══════════════════════════════════════════════════════════
        assert d_out % num_heads == 0, \
            f"d_out ({d_out}) must be divisible by num_heads ({num_heads})"
        
        # ═══════════════════════════════════════════════════════════
        # STORE CONFIGURATION
        # ═══════════════════════════════════════════════════════════
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Dimension per head
        
        # ═══════════════════════════════════════════════════════════
        # PROJECTION LAYERS
        # ═══════════════════════════════════════════════════════════
        # Single large projection instead of num_heads small ones
        # More efficient due to batched matrix multiplication
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # Output projection to combine heads
        self.out_proj = nn.Linear(d_out, d_out)
        
        # ═══════════════════════════════════════════════════════════
        # REGULARIZATION
        # ═══════════════════════════════════════════════════════════
        self.dropout = nn.Dropout(dropout)
        
        # ═══════════════════════════════════════════════════════════
        # CAUSAL MASK
        # ═══════════════════════════════════════════════════════════
        # Upper triangular matrix: 1s above diagonal, 0s elsewhere
        # Used to mask future tokens
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_in)
        
        Returns:
            Output tensor of shape (batch_size, seq_length, d_out)
        """
        # ═══════════════════════════════════════════════════════════
        # STEP 1: GET INPUT DIMENSIONS
        # ═══════════════════════════════════════════════════════════
        batch_size, num_tokens, d_in = x.shape
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: PROJECT TO Q, K, V
        # ═══════════════════════════════════════════════════════════
        # Shape: (batch_size, num_tokens, d_out)
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: SPLIT INTO MULTIPLE HEADS
        # ═══════════════════════════════════════════════════════════
        # Reshape: (batch, seq, d_out) → (batch, seq, num_heads, head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys    = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        
        # Transpose: (batch, seq, heads, head_dim) → (batch, heads, seq, head_dim)
        # This groups all positions for each head together
        queries = queries.transpose(1, 2)
        keys    = keys.transpose(1, 2)
        values  = values.transpose(1, 2)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 4: COMPUTE ATTENTION SCORES
        # ═══════════════════════════════════════════════════════════
        # (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq)
        # = (batch, heads, seq, seq)
        attn_scores = queries @ keys.transpose(2, 3)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 5: APPLY CAUSAL MASK
        # ═══════════════════════════════════════════════════════════
        # Mask future positions with -inf (becomes 0 after softmax)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 6: SCALE AND NORMALIZE
        # ═══════════════════════════════════════════════════════════
        # Scale by √head_dim for numerical stability
        attn_weights = torch.softmax(
            attn_scores / self.head_dim ** 0.5,
            dim=-1
        )
        
        # Apply dropout (only active during training)
        attn_weights = self.dropout(attn_weights)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 7: COMPUTE CONTEXT VECTORS
        # ═══════════════════════════════════════════════════════════
        # (batch, heads, seq, seq) @ (batch, heads, seq, head_dim)
        # = (batch, heads, seq, head_dim)
        context_vec = attn_weights @ values
        
        # ═══════════════════════════════════════════════════════════
        # STEP 8: COMBINE HEADS
        # ═══════════════════════════════════════════════════════════
        # Transpose back: (batch, heads, seq, head_dim) → (batch, seq, heads, head_dim)
        context_vec = context_vec.transpose(1, 2)
        
        # Reshape to concatenate heads: (batch, seq, heads * head_dim) = (batch, seq, d_out)
        # .contiguous() ensures memory layout is correct for .view()
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 9: OUTPUT PROJECTION
        # ═══════════════════════════════════════════════════════════
        # Final linear transformation to mix information from all heads
        context_vec = self.out_proj(context_vec)
        
        return context_vec
```

### Usage Examples

```python
# ═══════════════════════════════════════════════════════════════
# EXAMPLE 1: Basic Usage
# ═══════════════════════════════════════════════════════════════
mha = MultiHeadAttention(
    d_in=768,
    d_out=768,
    context_length=1024,
    dropout=0.1,
    num_heads=12
)

# Random input: (batch_size=2, seq_length=100, d_in=768)
x = torch.randn(2, 100, 768)
output = mha(x)
print(f"Output shape: {output.shape}")  # torch.Size([2, 100, 768])

# ═══════════════════════════════════════════════════════════════
# EXAMPLE 2: GPT-2 Small Configuration
# ═══════════════════════════════════════════════════════════════
gpt2_small_mha = MultiHeadAttention(
    d_in=768,           # Embedding dimension
    d_out=768,          # Same as d_in in GPT
    context_length=1024, # Max sequence length
    dropout=0.1,
    num_heads=12        # 12 attention heads
)

# ═══════════════════════════════════════════════════════════════
# EXAMPLE 3: Count Parameters
# ═══════════════════════════════════════════════════════════════
total_params = sum(p.numel() for p in mha.parameters())
print(f"Total parameters: {total_params:,}")
# W_query: 768 * 768 = 589,824
# W_key:   768 * 768 = 589,824
# W_value: 768 * 768 = 589,824
# out_proj: 768 * 768 = 589,824
# Total: 2,359,296 parameters

# ═══════════════════════════════════════════════════════════════
# EXAMPLE 4: Training Mode vs Eval Mode
# ═══════════════════════════════════════════════════════════════
mha.train()  # Dropout is active
mha.eval()   # Dropout is disabled

# ═══════════════════════════════════════════════════════════════
# EXAMPLE 5: Move to GPU
# ═══════════════════════════════════════════════════════════════
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mha = mha.to(device)
x = x.to(device)
output = mha(x)
```

### Shape Transformations (Visual Guide)

```
Input x:           (batch=2, seq=6, d_in=768)
                          │
                          ▼ W_query, W_key, W_value
Q, K, V:           (batch=2, seq=6, d_out=768)
                          │
                          ▼ .view(batch, seq, num_heads, head_dim)
Split:             (batch=2, seq=6, heads=12, head_dim=64)
                          │
                          ▼ .transpose(1, 2)
Transposed:        (batch=2, heads=12, seq=6, head_dim=64)
                          │
                          ▼ Q @ K.T
Attention Scores:  (batch=2, heads=12, seq=6, seq=6)
                          │
                          ▼ mask, softmax, dropout
Attention Weights: (batch=2, heads=12, seq=6, seq=6)
                          │
                          ▼ @ V
Context:           (batch=2, heads=12, seq=6, head_dim=64)
                          │
                          ▼ .transpose(1, 2)
Transposed Back:   (batch=2, seq=6, heads=12, head_dim=64)
                          │
                          ▼ .view(batch, seq, d_out)
Concatenated:      (batch=2, seq=6, d_out=768)
                          │
                          ▼ out_proj
Output:            (batch=2, seq=6, d_out=768)
```

---

## Complete Production Code

> Copy this entire class for your research projects

```python
"""
Multi-Head Attention Module for Transformer Models
Based on "Attention Is All You Need" (Vaswani et al., 2017)
Implementation follows "Build a Large Language Model From Scratch" by Sebastian Raschka

Author: Pascal Burume
Date: January 2025
"""

import torch
import torch.nn as nn
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with causal masking for autoregressive models.
    
    This is the standard attention mechanism used in GPT-style models.
    It supports:
    - Multiple attention heads for learning diverse patterns
    - Causal masking for left-to-right generation
    - Dropout for regularization
    - Efficient batched computation
    
    Attributes:
        d_out (int): Output dimension
        num_heads (int): Number of attention heads
        head_dim (int): Dimension per head (d_out // num_heads)
        W_query (nn.Linear): Query projection
        W_key (nn.Linear): Key projection
        W_value (nn.Linear): Value projection
        out_proj (nn.Linear): Output projection
        dropout (nn.Dropout): Dropout layer
        mask (torch.Tensor): Causal attention mask
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False
    ):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_in: Input embedding dimension
            d_out: Output embedding dimension
            context_length: Maximum sequence length
            dropout: Dropout probability (0.0 to 1.0)
            num_heads: Number of attention heads
            qkv_bias: Use bias in Q, K, V projections
            
        Raises:
            AssertionError: If d_out is not divisible by num_heads
        """
        super().__init__()
        
        assert d_out % num_heads == 0, \
            f"d_out ({d_out}) must be divisible by num_heads ({num_heads})"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        # Projection layers
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask (upper triangular)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_in)
            
        Returns:
            Output tensor of shape (batch_size, seq_length, d_out)
        """
        b, n, _ = x.shape
        
        # Project to Q, K, V
        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)
        
        # Split into heads
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = q @ k.transpose(-2, -1)
        
        # Apply causal mask
        scores.masked_fill_(self.mask.bool()[:n, :n], float('-inf'))
        
        # Scale, normalize, dropout
        weights = torch.softmax(scores / (self.head_dim ** 0.5), dim=-1)
        weights = self.dropout(weights)
        
        # Compute output
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(b, n, self.d_out)
        out = self.out_proj(out)
        
        return out


# ═══════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS FOR COMMON CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════

def create_gpt2_small_attention(dropout: float = 0.1) -> MultiHeadAttention:
    """Create attention module with GPT-2 Small configuration."""
    return MultiHeadAttention(
        d_in=768, d_out=768, context_length=1024,
        dropout=dropout, num_heads=12
    )

def create_gpt2_medium_attention(dropout: float = 0.1) -> MultiHeadAttention:
    """Create attention module with GPT-2 Medium configuration."""
    return MultiHeadAttention(
        d_in=1024, d_out=1024, context_length=1024,
        dropout=dropout, num_heads=24
    )

def create_gpt2_large_attention(dropout: float = 0.1) -> MultiHeadAttention:
    """Create attention module with GPT-2 Large configuration."""
    return MultiHeadAttention(
        d_in=1280, d_out=1280, context_length=1024,
        dropout=dropout, num_heads=36
    )

def create_gpt2_xl_attention(dropout: float = 0.1) -> MultiHeadAttention:
    """Create attention module with GPT-2 XL configuration."""
    return MultiHeadAttention(
        d_in=1600, d_out=1600, context_length=1024,
        dropout=dropout, num_heads=25
    )
```

---

## GPT-2 Specifications

| Model | Parameters | Layers | d_model | Heads | head_dim | Context |
|-------|------------|--------|---------|-------|----------|---------|
| GPT-2 Small | 117M | 12 | 768 | 12 | 64 | 1024 |
| GPT-2 Medium | 345M | 24 | 1024 | 24 | ~43 | 1024 |
| GPT-2 Large | 762M | 36 | 1280 | 36 | ~36 | 1024 |
| GPT-2 XL | 1.5B | 48 | 1600 | 25 | 64 | 1024 |

### Configuration Dictionary

```python
GPT2_CONFIGS = {
    "gpt2-small": {
        "vocab_size": 50257,
        "context_length": 1024,
        "d_model": 768,
        "num_heads": 12,
        "num_layers": 12,
        "dropout": 0.1,
        "qkv_bias": False
    },
    "gpt2-medium": {
        "vocab_size": 50257,
        "context_length": 1024,
        "d_model": 1024,
        "num_heads": 24,
        "num_layers": 24,
        "dropout": 0.1,
        "qkv_bias": False
    },
    "gpt2-large": {
        "vocab_size": 50257,
        "context_length": 1024,
        "d_model": 1280,
        "num_heads": 36,
        "num_layers": 36,
        "dropout": 0.1,
        "qkv_bias": False
    },
    "gpt2-xl": {
        "vocab_size": 50257,
        "context_length": 1024,
        "d_model": 1600,
        "num_heads": 25,
        "num_layers": 48,
        "dropout": 0.1,
        "qkv_bias": False
    }
}
```

---

## Common Errors & Solutions

### Error 1: d_out not divisible by num_heads

```python
# ❌ ERROR
mha = MultiHeadAttention(d_in=768, d_out=768, num_heads=7, ...)
# AssertionError: d_out (768) must be divisible by num_heads (7)

# ✅ SOLUTION: Use num_heads that divides d_out evenly
mha = MultiHeadAttention(d_in=768, d_out=768, num_heads=12, ...)  # 768/12=64 ✓
```

### Error 2: Missing super().__init__()

```python
# ❌ ERROR
class MyAttention(nn.Module):
    def __init__(self, ...):
        # Forgot super().__init__()
        self.W_query = nn.Linear(...)  # RuntimeError!

# ✅ SOLUTION: Always call super().__init__() first
class MyAttention(nn.Module):
    def __init__(self, ...):
        super().__init__()  # ← Add this!
        self.W_query = nn.Linear(...)
```

### Error 3: Dimension Mismatch

```python
# ❌ ERROR: Input dimension doesn't match d_in
mha = MultiHeadAttention(d_in=768, ...)
x = torch.randn(2, 100, 512)  # d_in=512, but expected 768!
output = mha(x)  # RuntimeError: mat1 and mat2 shapes cannot be multiplied

# ✅ SOLUTION: Ensure input dimension matches d_in
x = torch.randn(2, 100, 768)  # Correct dimension
output = mha(x)
```

### Error 4: Forgetting .contiguous() before .view()

```python
# ❌ ERROR
context_vec = context_vec.transpose(1, 2)
context_vec = context_vec.view(b, n, self.d_out)  # RuntimeError!

# ✅ SOLUTION: Call .contiguous() after transpose
context_vec = context_vec.transpose(1, 2).contiguous()
context_vec = context_vec.view(b, n, self.d_out)  # Works!
```

### Error 5: Mask Shape Mismatch

```python
# ❌ ERROR: Sequence longer than context_length
mha = MultiHeadAttention(context_length=512, ...)
x = torch.randn(2, 1024, 768)  # seq_length=1024 > context_length=512
output = mha(x)  # IndexError!

# ✅ SOLUTION: Ensure seq_length ≤ context_length
mha = MultiHeadAttention(context_length=1024, ...)  # Or truncate input
```

---

## Quick Reference Cheat Sheet

### Core Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

### Key Shapes

```python
# Input
x:       (batch, seq_len, d_in)

# After projection
Q, K, V: (batch, seq_len, d_out)

# After splitting into heads
Q, K, V: (batch, num_heads, seq_len, head_dim)

# Attention scores
scores:  (batch, num_heads, seq_len, seq_len)

# Output
output:  (batch, seq_len, d_out)
```

### Essential PyTorch Operations

```python
# Matrix multiplication
A @ B                      # Standard matmul
torch.bmm(A, B)            # Batch matmul

# Reshaping
x.view(b, n, h, d)         # Reshape tensor
x.transpose(1, 2)          # Swap dimensions
x.contiguous()             # Ensure memory layout

# Masking
x.masked_fill_(mask, val)  # In-place fill where mask is True
torch.triu(x, diagonal=1)  # Upper triangular matrix

# Normalization
torch.softmax(x, dim=-1)   # Softmax along last dimension

# Registration
self.register_buffer('name', tensor)  # Save with model, not trained
```

### Hyperparameters

| Parameter | Typical Values | Notes |
|-----------|---------------|-------|
| d_model | 256, 512, 768, 1024 | Model dimension |
| num_heads | 4, 8, 12, 16 | Must divide d_model |
| head_dim | 64 | Usually d_model / num_heads |
| dropout | 0.0, 0.1, 0.2 | Higher = more regularization |
| context_length | 512, 1024, 2048 | Max sequence length |

---

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
2. Raschka, S. (2024). "Build a Large Language Model From Scratch." Manning Publications.
3. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." OpenAI.

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2025 | Initial version |

---

*This document is part of the Mwalimu-STEM-GenAI research project.*
