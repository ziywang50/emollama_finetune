# Emollama Mental Health Fine-tuning

This project fine-tunes the Emollama-7b model for mental health counseling using LoRA (Low-Rank Adaptation). The model is trained on a combination of real and synthetic mental health conversations to provide empathetic and helpful responses.

## Features

- Fine-tunes Emollama-7b using LoRA
- Uses 4-bit quantization for efficient training
- Handles conversation format with proper prompt engineering
- Includes evaluation metrics and sample generation
- Supports training checkpoint management

## Prerequisites

- Python 3.11+
- PyTorch 2.0+
- Transformers library
- PEFT library
- At least 16GB GPU memory
- CUDA support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emollama-finetune.git
cd emollama-finetune
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Format

The training data should be CSV files with three columns:
- `instruction`: The system prompt for the model
- `input`: User messages/queries
- `output`: Counselor responses

## Training

To start training:
```bash
python train.py
```

The script will:
- Load and preprocess the data
- Initialize the model with LoRA configuration
- Train using specified hyperparameters
- Save checkpoints during training
- Generate evaluation metrics

## Model Artifacts

The following artifacts are generated during training:
- Checkpoint directories: `./emollama-mental-health-finetuned/checkpoint-*`
- Final model: `emollama-mental-health-lora_[timestamp]`
- Latest model symlink: `emollama-mental-health-lora_latest`

## Configuration

Key hyperparameters in `train.py`:
- Learning rate: 5e-5
- Batch size: 4
- Training epochs: 3
- LoRA rank: 16
- LoRA alpha: 32

## Results

The model is evaluated on:
- Perplexity
- Evaluation loss
- Sample generations

## File Structure

```
emollama_finetune/
├── train.py           # Main training script
├── requirements.txt   # Python dependencies
├── .gitignore        # Git ignore rules
└── README.md         # This file
```

## Notes

- Large model files and checkpoints are not included in the repository
- Training data is not included due to privacy concerns
- See .gitignore for excluded files

## License

[Your chosen license]

## Acknowledgments

- Original Emollama-7b model by [lzw1008](https://huggingface.co/lzw1008/Emollama-7b)
- LoRA implementation from Microsoft