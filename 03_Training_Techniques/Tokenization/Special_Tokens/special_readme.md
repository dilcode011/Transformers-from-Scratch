# 🏷️ Special Tokens Management

In Transformer models, specific non-text tokens are injected into sequences to provide structural cues. This utility ensures that every sequence is properly formatted before entering the model.

## 🔑 The Core Tokens
| Token | Meaning | Purpose |
| :--- | :--- | :--- |
| `[PAD]` | Padding | Ensures all sequences in a batch have the same length. |
| `[CLS]` | Classification | Represents the aggregate sequence representation (BERT). |
| `[SEP]` | Separator | Marks the end of a sentence or a boundary between two. |
| `[MASK]` | Masking | Hides a token during Masked Language Modeling (MLM). |
| `<SOS>` | Start of Sentence | Signals the decoder to begin generating (GPT/Vanilla). |
| `<EOS>` | End of Sentence | Signals the model to stop generating. |

## 🛠️ Functionality
The scripts in this directory handle:
1. **Sequence Formatting:** Adding the necessary boundary tokens.
2. **Truncation:** Cutting sequences that exceed the `max_len`.
3. **Padding:** Filling empty space so tensors are rectangular.
4. **Mask Generation:** Creating the binary Attention Mask (1 for real, 0 for pad).