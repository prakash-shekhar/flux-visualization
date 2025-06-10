# flux-visualization
A nice visualization for the architecture of the FLUX model

# FLUX Architecture

- Based on this https://github.com/black-forest-labs/flux/

## INPUT STAGE
1. Text Conditioning
  - CLIP Encoder (flux/modules/conditioner.py:12-14): OpenAI CLIP for pooled text features
  - T5 Encoder (flux/modules/conditioner.py:16-17): Google T5 for sequential text embeddings
  - Text Processing (flux/sampling.py:52-55): T5 generates context tokens, CLIP generates vector embeddings
2. Image Input Processing
  - VAE Encoder (flux/modules/autoencoder.py:109-180): Encodes images to latent space (16 channels)
  - Patch Embedding (flux/sampling.py:41): Converts 2×2 patches to sequence tokens
  - Positional IDs (flux/sampling.py:45-48): Spatial position encoding for image patches
3. Conditioning Variants
  - Standard: Text-only conditioning
  - Fill/Inpainting (flux/sampling.py:107-157): Image + mask conditioning
  - Control (flux/sampling.py:70-104): Canny/Depth image conditioning via specialized encoders
  - Redux (flux/sampling.py:160-207): Image variation via SigLIP vision encoder
4. Temporal & Guidance Embeddings
  - Timestep Embedding (flux/modules/layers.py:28-49): Sinusoidal time encoding
  - Guidance Embedding (flux/model.py:100-103): Classifier-free guidance strength
  - Vector Conditioning (flux/model.py:104): Combined time + guidance + CLIP features
## DUAL-STREAM PROCESSING
1. DoubleStreamBlock Architecture (flux/modules/layers.py:129-191)
  - Separate Processing: Independent attention for image and text streams
  - Cross-Attention: Joint attention computation across both modalities
  - Modulated Components:
    - Attention with QK normalization
    - MLP with GELU activation
    - AdaLN modulation using vector conditioning
2. Processing Flow (flux/model.py:110-111)
  - Image tokens: [batch, img_seq_len, hidden_dim]
  - Text tokens: [batch, txt_seq_len, hidden_dim]
  - Joint attention: concat(txt_q,img_q), concat(txt_k,img_k), concat(txt_v,img_v)
  - RoPE positional encoding applied across combined sequence
## SINGLE-STREAM PROCESSING
1. Stream Fusion (flux/model.py:113)
  - Concatenation: img = concat(txt, img) - text first, then image tokens
  - Unified Processing: Single attention stream processes combined sequence
2. SingleStreamBlock Architecture (flux/modules/layers.py:194-239)
  - Parallel Architecture: QKV + MLP computed in parallel linear layers
  - Efficient Processing: linear1 → [qkv, mlp] → attention + activation → linear2
  - Modulation: AdaLN with shift, scale, gate parameters
3. Processing Flow (flux/model.py:114-116)
  - Combined sequence length: txt_seq_len + img_seq_len
  - After processing, extract image tokens: img[:, txt.shape[1]:, ...]
## OUTPUT STAGE
1. Final Layer (flux/modules/layers.py:242-253)
  - AdaLN Normalization: Adaptive layer normalization with vector conditioning
  - Linear Projection: Maps hidden dimension to patch channels
  - Output Shape: [batch, img_seq_len, patch_size² × out_channels]
2. Denoising Process (flux/sampling.py:241-271)
  - Flow Matching: Euler method integration from noise to data
  - Prediction: Model predicts velocity field v(x,t)
  - Update Rule: x_{t-1} = x_t + (t_{prev} - t_{curr}) × pred
3. VAE Decoding
  - Unpack Patches (flux/sampling.py:274-282): Reshape sequence back to spatial
  - VAE Decoder (flux/modules/autoencoder.py:313-315): Latent → RGB image
  - Normalization: Scale and shift latent values before decoding
