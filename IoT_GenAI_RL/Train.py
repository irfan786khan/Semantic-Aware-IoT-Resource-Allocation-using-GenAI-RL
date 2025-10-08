import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import math
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# ========================
# Configuration
# ========================
MODEL_SAVE_PATH = "dqn_agent.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ========================
# Step 1: Load WISDM Datasetco
# ========================
def load_wisdm_raw_dataset(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().strip(';').split(',')
            if len(parts) == 6:
                try:
                    data.append([
                        int(parts[0]),       # user
                        parts[1].strip(),     # activity
                        int(parts[2]),        # timestamp
                        float(parts[3]),      # x
                        float(parts[4]),      # y
                        float(parts[5])       # z
                    ])
                except ValueError:
                    continue
    df = pd.DataFrame(data, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])
    return df

# ========================
# Step 2: Create Sliding Windows & Prompts
# ========================
def create_prompts_from_windows(df, window_size=50, step=25):
    prompts = []
    labels = []
    activities = df['activity'].unique().tolist()

    for i in range(0, len(df) - window_size, step):
        window = df.iloc[i:i+window_size]
        x = window['x'].round(2).tolist()
        y = window['y'].round(2).tolist()
        z = window['z'].round(2).tolist()
        activity = window['activity'].mode()[0]

        prompt = f"""Analyze these accelerometer readings and classify the activity:
Possible activities: {', '.join(activities)}.

Example:
X=[0.12, 0.15, 0.18], Y=[-0.98, -0.97, -0.96], Z=[0.05, 0.06, 0.04] → "Walking"

Current Data:
X={x}
Y={y}
Z={z}
The most likely activity is:"""

        prompts.append(prompt)
        labels.append(activity)
    return prompts, labels

# ========================
# Step 3: Load Mistral-7B Model
# ========================
def load_model(model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return tokenizer, model

# ========================
# Step 4: Extract Semantic Embedding
# ========================
def extract_semantic_embedding(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Use mean pooling of all tokens
    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
    return embedding

# ========================
# Reinforcement Learning Components
# ========================
ACTIONS = {
    0: "Drop data",
    1: "Transmit raw",
    2: "Transmit semantic summary",
    3: "Compress and transmit",
    4: "Delay transmission"
}

# Hyperparameters
LR = 1e-4
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 5000
TAU = 0.005
HIDDEN_SIZE = 256
BUFFER_SIZE = 100000
BATCH_SIZE = 256
UPDATE_EVERY = 100

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.LongTensor(np.array(actions)).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(DEVICE)
        )

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=HIDDEN_SIZE):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        self.to(DEVICE)

    def forward(self, state):
        return self.net(state)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPS_START
        self.steps_done = 0
        self.action_counts = np.zeros(action_dim)

        # Networks
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR, eps=1e-4)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, state, evaluate=False):
        if evaluate:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                return self.policy_net(state).argmax().item()
        
        self.steps_done += 1
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.epsilon = eps_threshold

        if random.random() > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                # Add diversity bonus
                action_probs = self.action_counts / (self.action_counts.sum() + 1e-5)
                diversity_bonus = 0.1 * (1 - action_probs)
                q_values = self.policy_net(state_tensor).cpu().numpy() + diversity_bonus
                action = np.argmax(q_values)
        else:
            action = random.randrange(self.action_dim)
        
        self.action_counts[action] += 1
        return action

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q = rewards + (1 - dones) * GAMMA * next_q

        loss = nn.SmoothL1Loss()(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)

    def save_model(self, path):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'action_counts': self.action_counts
        }, path)

    def load_model(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)  # <-- updated
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.steps_done = checkpoint['steps_done']
            self.epsilon = checkpoint['epsilon']
            self.action_counts = checkpoint.get('action_counts', np.zeros(self.action_dim))


def calculate_reward(action, embedding, original_data_size=50*3*4, transmission_cost=0.01):
    """Improved reward function with better scaling"""
    base_utility = 1.0
    compression_factor = embedding.shape[0] * 4 / original_data_size
    
    rewards = {
        0: (0.1 * base_utility, 0),                     # Drop
        1: (base_utility, original_data_size * transmission_cost),  # Raw
        2: (0.9 * base_utility * (1 + 0.5*compression_factor), 0.05 * transmission_cost),  # Semantic
        3: (0.8 * base_utility, (original_data_size * 0.3) * transmission_cost),  # Compressed
        4: (0.7 * base_utility, 0)                      # Delay
    }
    
    utility, cost = rewards[action]
    return (utility - cost) / 10  # Scaled to reasonable range

# ========================
# Main Training Loop
# ========================
if __name__ == "__main__":
    # Load and prepare data
    df = load_wisdm_raw_dataset("WISDM_ar_v1.1_raw.txt")
    prompts, labels = create_prompts_from_windows(df)
    
    # Load model
    tokenizer, model = load_model()
    
    # Initialize agent
    test_embedding = extract_semantic_embedding(prompts[0], tokenizer, model)
    agent = DQNAgent(test_embedding.shape[0], len(ACTIONS))
    
    # Training parameters
    num_episodes = 2
    num_samples = min(200, len(prompts))  # Use up to 200 samples
    
    for episode in range(num_episodes):
        episode_rewards = []
        action_distribution = np.zeros(len(ACTIONS))
        
        for i in tqdm(range(num_samples), desc=f"Episode {episode+1}/{num_episodes}"):
            try:
                emb = extract_semantic_embedding(prompts[i], tokenizer, model)
                if emb is not None:
                    action = agent.select_action(emb)
                    reward = calculate_reward(action, emb)
                    
                    agent.replay_buffer.add(emb, action, reward, emb, False)
                    episode_rewards.append(reward)
                    action_distribution[action] += 1
                    
                    if i % UPDATE_EVERY == 0:
                        agent.update()
                        
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue
        
        # Print statistics
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        action_percent = action_distribution / action_distribution.sum() * 100
        print(f"\nEpisode {episode+1}:")
        print(f"Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.2f}")
        print("Action Distribution:")
        for act, percent in zip(ACTIONS.values(), action_percent):
            print(f"{act}: {percent:.1f}%")
        
        if (episode+1) % 10 == 0:
            agent.save_model(MODEL_SAVE_PATH)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    test_actions = []
    for i in range(min(20, len(prompts))):  # Test on 20 samples
        emb = extract_semantic_embedding(prompts[i], tokenizer, model)
        action = agent.select_action(emb, evaluate=True)
        test_actions.append(action)
        print(f"Sample {i+1} ({labels[i]}): {ACTIONS[action]}")
    
    print("\nTest Action Distribution:")
    unique, counts = np.unique(test_actions, return_counts=True)
    for action, count in zip(unique, counts):
        print(f"{ACTIONS[action]}: {count} samples")