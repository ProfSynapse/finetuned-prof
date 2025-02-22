## Professor Synapse Training Notebook 🧙🏿‍♂️
### A Fine-tuning Implementation using Unsloth

This Colab notebook implements a specialized training pipeline for creating Professor Synapse, an AI assistant with advanced reasoning capabilities, using the Unsloth framework and Pixtral-12B model.

### Prerequisites
- Google Colab with T4/V100 GPU
- At least 25GB RAM
- Python 3.8+
- Access to `prof_synapse.jsonl` training data

### Installation
```python
!pip install "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git"
!pip install datasets trl
```

### Key Components

#### 1. Model Architecture
- Base Model: `unsloth/Pixtral-12B-Base-2409`
- Max Sequence Length: 4096 tokens
- Quantization: 4-bit with Flash Attention 2
- Custom reasoning schema with specialized tokens

#### 2. Training Configuration
- Learning Rate: 5e-7
- Epochs: 3
- Batch Size: 1 (with gradient accumulation steps = 8)
- Optimizer: PagedAdamW8bit
- Mixed Precision: BF16 (where supported)

#### 3. Special Features
- Custom reasoning framework with semantic parsing
- Knowledge graph integration
- Quality-based dataset filtering
- Banned words handling
- MoE-specific LoRA optimization

### Usage Instructions

1. **Setup**
   - Mount Google Drive
   - Upload your `prof_synapse.jsonl` dataset
   - Run installation cell

2. **Training**
   - Execute cells sequentially
   - Monitor training metrics
   - Check GPU memory usage

3. **Export**
   - Model saves in GGUF format
   - Quantized for efficient deployment
   - Expert layers specially handled

### Quality Metrics
The training implements a custom scoring system:
```python
score = 1.0 * conversation_quality
       - 0.5 * bias_detection
       + 0.75 * reasoning_quality
```

### Memory Requirements
- Training: ~24GB GPU RAM
- Inference: ~16GB GPU RAM
- Dataset: Variable (depends on size)

### Optimization Notes
- Uses Unsloth's gradient checkpointing
- Implements flash attention 2
- MoE-specific LoRA parameters
- Custom token handling for reasoning framework

### Output
The final model is saved in two formats:
- GGUF format with q4_k_m quantization
- Expert layers with q6_k quantization

### Limitations
- Requires significant GPU resources
- Limited to specific reasoning framework
- Training time depends on dataset size

### Support
For issues and updates, visit:
- Unsloth GitHub repository
- Project documentation
- Original codebase reference

### License
Refer to Unsloth and base model licenses for usage terms.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/9992018/0b27e48e-0633-4efc-828e-6f8a7dbf1c49/paste.txt
