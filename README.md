# Semantic-Aware IoT Resource Allocation and Latency Optimization Using Deep Reinforcement Learning

## 📋 Project Overview

This project implements a Deep Q-Network (DQN) with LSTM for optimizing IoT data transmission decisions using semantic embeddings from the Mistral-7B language model. The system processes accelerometer data from the WISDM dataset and learns to make intelligent transmission decisions to balance data utility and transmission costs.

## 🏗️ Project Structure

### Core Files:

1. **`Change_model.py`** - Main DQN-LSTM agent implementation with training loop
2. **`Train.py`** - Basic DQN agent implementation (simpler version)
3. **`splitting_datatset.py`** - Dataset preprocessing and train/test split
4. **`results.py`** - Evaluation and visualization of trained models
5. **`results_with_plots.py`** - Enhanced evaluation with comprehensive plotting
6. **`XAI_clean_results.py`** - SHAP-based explainable AI analysis
7. **`XAI_LIME.py`** - Alternative XAI implementation using LIME

## 🎯 Key Features

### Data Processing
- **WISDM Dataset**: Human activity recognition using accelerometer data
- **Sliding Windows**: 50-sample windows with 25-sample steps
- **Semantic Prompts**: Natural language prompts for LLM processing
- **Train/Test Split**: User-based splitting to prevent data leakage

### Reinforcement Learning
- **5 Actions**:
  - 0: Drop data
  - 1: Transmit raw data
  - 2: Transmit semantic summary
  - 3: Compress and transmit
  - 4: Delay transmission

- **DQN with LSTM**: Handles temporal dependencies in sequential data
- **Experience Replay**: Buffer with sequence-based sampling
- **Target Network**: Stable learning with periodic updates

### Semantic Processing
- **Mistral-7B**: Large language model for embedding generation
- **8-bit Quantization**: Memory-efficient model loading
- **Fallback Mechanism**: Numeric feature extraction if LLM unavailable

### Explainable AI (XAI)
- **SHAP Analysis**: Feature importance for model decisions
- **LIME Integration**: Local interpretable model explanations
- **Visualization**: Summary plots and force plots

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers pandas numpy matplotlib scikit-learn tqdm shap lime
```

### Basic Usage

1. **Preprocess Data**:
```bash
python splitting_datatset.py
```

2. **Train Model**:
```bash
python Change_model.py
```

3. **Evaluate Results**:
```bash
python results_with_plots.py
```

4. **Explain Decisions**:
```bash
python XAI_clean_results.py
```

## ⚙️ Configuration

### Key Parameters in `Change_model.py`:

```python
# RL Hyperparameters
LR = 1e-4                    # Learning rate
GAMMA = 0.99                 # Discount factor
EPS_START = 0.9              # Initial exploration rate
EPS_END = 0.1                # Final exploration rate
BUFFER_SIZE = 100000         # Replay buffer size
BATCH_SIZE = 256             # Training batch size
HIDDEN_SIZE = 256            # LSTM hidden size
SEQUENCE_LENGTH = 5          # Sequence length for LSTM
```

### Model Paths:
- **Trained Model**: `dqn_lstm_agent.pth`
- **Processed Data**: `processed_wisdm/` directory
- **Original Dataset**: `WISDM_ar_v1.1_raw.txt`

## 📊 Outputs

### Training Metrics:
- Episode rewards and epsilon decay
- Action distribution over time
- Loss curves and Q-value convergence

### Evaluation Results:
- Test set performance
- Action distribution analysis
- Reward trends across samples

### XAI Visualizations:
- SHAP summary plots
- Feature importance bars
- Individual sample explanations

## 🧠 Technical Details

### Reward Function:
Balances utility and cost:
- **Utility**: Based on action type and compression efficiency
- **Cost**: Transmission cost proportional to data size
- **Scaling**: Normalized to reasonable range

### State Representation:
- **Semantic Embeddings**: 4096-dim vectors from Mistral-7B
- **Fallback Features**: Statistical features (mean, std, min, max, variance)
- **Sequence Processing**: LSTM handles temporal patterns

### Network Architecture:
```python
DQN_LSTM(
  (lstm): LSTM(input_size, 256, batch_first=True)
  (fc1): Linear(256, 256)
  (fc2): Linear(256, 5)  # 5 actions
)
```

## 🔧 Customization

### Adding New Actions:
1. Update `ACTIONS` dictionary
2. Modify `calculate_reward()` function
3. Adjust network output layer

### Changing Dataset:
1. Implement new data loader function
2. Update prompt generation logic
3. Modify feature extraction as needed

### Tuning Rewards:
Edit the `calculate_reward()` function to reflect your specific utility-cost tradeoffs.

## 📈 Performance Monitoring

The system provides:
- Real-time training progress with tqdm
- Episode-wise statistics printing
- Model checkpointing every 10 episodes
- Comprehensive evaluation metrics

## 🐛 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use smaller sequence length
   - Enable 8-bit quantization

2. **Model Loading Failures**:
   - Check internet connection for model downloads
   - Use fallback feature extraction
   - Verify model path exists

3. **Training Instability**:
   - Adjust learning rate
   - Increase replay buffer size
   - Tune exploration parameters

## 📄 License

This project is for research purposes. Please ensure proper attribution when using the WISDM dataset and Mistral-7B model.

## 🤝 Contributing

Feel free to submit issues and enhancement requests to improve the system's performance and functionality.
