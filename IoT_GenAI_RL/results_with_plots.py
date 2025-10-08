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
import matplotlib.pyplot as plt

# ========================
# Configuration
# ========================
MODEL_SAVE_PATH = "dqn_lstm_agent.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ========================
# Step 1: Load WISDM Dataset
# ========================
def load_wisdm_raw_dataset(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().strip(';').split(',')
            if len(parts) == 6:
                try:
                    data.append([int(parts[0]),       # user
                                 parts[1].strip(),    # activity
                                 int(parts[2]),       # timestamp
                                 float(parts[3]),     # x
                                 float(parts[4]),     # y
                                 float(parts[5])])    # z
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

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Tokenizer download failed or offline: {e}")
        print("Proceeding without tokenizer/model. extract_semantic_embedding will fall back to numeric parsing.")
        return None, None

    try:
        # Try memory-efficient 8-bit load to GPU with low_cpu_mem_usage
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("Loaded model with 8-bit quantization on GPU (device_map='auto').")
    except Exception as e:
        print(f"GPU/8-bit load failed: {e}")
        print("Falling back to CPU-only load (this may be slower) or skipping model load if offline.")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map={'': 'cpu'},
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            print("Loaded model on CPU.")
        except Exception as e2:
            print(f"CPU load also failed or offline: {e2}")
            print("Proceeding without tokenizer/model. extract_semantic_embedding will fall back to numeric parsing.")
            return None, None

    return tokenizer, model

# ========================
# Step 4: Extract Semantic Embedding
# ========================
def extract_semantic_embedding(prompt, tokenizer, model):
    # If model/tokenizer couldn't be loaded (offline or OOM), fall back to numeric parsing
    if tokenizer is None or model is None:
        # Try to parse X, Y, Z arrays from the prompt (created by create_prompts_from_windows)
        import re
        try:
            def parse_array(label):
                m = re.search(rf"{label}=\[([^\]]+)\]", prompt)
                if not m:
                    return None
                nums = [float(x.strip()) for x in m.group(1).split(',') if x.strip()]
                return np.array(nums)

            x = parse_array('X')
            y = parse_array('Y')
            z = parse_array('Z')

            features = []
            for arr in (x, y, z):
                if arr is None or arr.size == 0:
                    # pad with zeros if missing
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    features.append(float(np.mean(arr)))
                    features.append(float(np.std(arr)))
                    features.append(float(np.min(arr)))
                    features.append(float(np.max(arr)))
                    features.append(float(np.var(arr)))

            return np.array(features)
        except Exception as e:
            # As a last resort, create a deterministic hash-based vector
            print(f"Numeric parse failed, creating hash-based fallback embedding: {e}")
            bs = np.frombuffer(prompt.encode('utf-8'), dtype=np.uint8)
            # create fixed-size vector (64) by folding
            vec = np.zeros(64, dtype=np.float32)
            for i, b in enumerate(bs):
                vec[i % 64] += b
            # normalize
            vec = vec / (np.linalg.norm(vec) + 1e-6)
            return vec

    # If model is available, use it
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Use mean pooling of all tokens
    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
    return embedding

# ========================
# Reinforcement Learning Components with LSTM
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
SEQUENCE_LENGTH = 5  # Number of consecutive states to form a sequence

class ReplayBuffer:
    def __init__(self, buffer_size, sequence_length=SEQUENCE_LENGTH):
        self.buffer = deque(maxlen=buffer_size)
        self.sequence_length = sequence_length
        self.current_sequence = []

    def add(self, state, action, reward, next_state, done):
        self.current_sequence.append((state, action, reward, next_state, done))

        # When we have enough steps in the sequence, add to buffer
        if len(self.current_sequence) >= self.sequence_length:
            # Get the last 'sequence_length' steps
            sequence = self.current_sequence[-self.sequence_length:]

            # Stack states and next_states
            states = np.array([s[0] for s in sequence])
            actions = np.array([s[1] for s in sequence])
            rewards = np.array([s[2] for s in sequence])
            next_states = np.array([s[3] for s in sequence])
            dones = np.array([s[4] for s in sequence])

            self.buffer.append((states, actions, rewards, next_states, dones))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors with proper dimensions
        states_tensor = torch.FloatTensor(np.array(states)).to(DEVICE)  # [batch, seq_len, state_dim]
        actions_tensor = torch.LongTensor(np.array(actions)).to(DEVICE)  # [batch, seq_len]
        rewards_tensor = torch.FloatTensor(np.array(rewards)).to(DEVICE)  # [batch, seq_len]
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(DEVICE)  # [batch, seq_len, state_dim]
        dones_tensor = torch.FloatTensor(np.array(dones)).to(DEVICE)  # [batch, seq_len]

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self):
        return len(self.buffer)

class DQN_LSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=HIDDEN_SIZE):
        super(DQN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim

        # LSTM layer
        self.lstm = nn.LSTM(state_dim, hidden_size, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

        self.to(DEVICE)

    def forward(self, x, hidden=None):
        # x shape: [batch_size, sequence_length, state_dim]
        batch_size = x.size(0)

        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(1, batch_size, self.hidden_size).to(DEVICE)
            c0 = torch.zeros(1, batch_size, self.hidden_size).to(DEVICE)
            hidden = (h0, c0)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)

        # Only use the last output in the sequence
        lstm_out = lstm_out[:, -1, :]  # [batch_size, hidden_size]

        # Fully connected layers
        x = torch.relu(self.fc1(lstm_out))
        q_values = self.fc2(x)

        return q_values, hidden

    def init_hidden(self, batch_size=1):
        # Initialize hidden state and cell state
        return (torch.zeros(1, batch_size, self.hidden_size).to(DEVICE),
                torch.zeros(1, batch_size, self.hidden_size).to(DEVICE))

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPS_START
        self.steps_done = 0
        self.action_counts = np.zeros(action_dim)

        # Networks
        self.policy_net = DQN_LSTM(state_dim, action_dim)
        self.target_net = DQN_LSTM(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR, eps=1e-4)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        # Hidden states
        self.hidden = None

    def select_action(self, state, evaluate=False):
        if evaluate:
            with torch.no_grad():
                # For evaluation, we process one state at a time
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1, 1, state_dim]
                q_values, self.hidden = self.policy_net(state_tensor, self.hidden)
                return q_values.argmax().item()

        self.steps_done += 1
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.epsilon = eps_threshold

        if random.random() > eps_threshold:
            with torch.no_grad():
                # For training, we need to maintain the sequence
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1, 1, state_dim]
                q_values, self.hidden = self.policy_net(state_tensor, self.hidden)
                # Add diversity bonus
                action_probs = self.action_counts / (self.action_counts.sum() + 1e-5)
                diversity_bonus = 0.1 * (1 - action_probs)
                q_values = q_values.cpu().numpy() + diversity_bonus
                action = np.argmax(q_values)
        else:
            action = random.randrange(self.action_dim)

        self.action_counts[action] += 1
        return action

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        # Initialize hidden states
        hidden = self.policy_net.init_hidden(BATCH_SIZE)

        # Get current Q values
        current_q, _ = self.policy_net(states, hidden)
        # Select the Q-values for the taken actions
        current_q = current_q.gather(1, actions[:, -1].unsqueeze(1))  # Only use last action in sequence

        # Get next Q values from target network
        with torch.no_grad():
            next_q, _ = self.target_net(next_states, hidden)
            next_q = next_q.max(1)[0].unsqueeze(1)

        # Only use the last reward and done in the sequence
        expected_q = rewards[:, -1].unsqueeze(1) + (1 - dones[:, -1].unsqueeze(1)) * GAMMA * next_q

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
            checkpoint = torch.load(path, map_location=DEVICE)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.steps_done = checkpoint['steps_done']
            self.epsilon = checkpoint['epsilon']
            self.action_counts = checkpoint.get('action_counts', np.zeros(self.action_dim))

# Reward function
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

    # Initialize a dictionary to store the results
    episode_results = {
        "episode": [],
        "reward": [],
        "epsilon": [],
        "action_distribution": {
            "Drop data": [],
            "Transmit raw": [],
            "Transmit semantic summary": [],
            "Compress and transmit": [],
            "Delay transmission": []
        }
    }

    # Training parameters
    num_episodes = 50
    num_samples = min(200, len(prompts))  # Use up to 200 samples

    for episode in range(num_episodes):
        episode_rewards = []
        action_distribution = np.zeros(len(ACTIONS))
        agent.hidden = agent.policy_net.init_hidden()  # Reset hidden state at start of episode

        for i in tqdm(range(num_samples), desc=f"Episode {episode+1}/{num_episodes}"):
            try:
                emb = extract_semantic_embedding(prompts[i], tokenizer, model)
                if emb is not None:
                    action = agent.select_action(emb)
                    reward = calculate_reward(action, emb)

                    # Add to replay buffer
                    agent.replay_buffer.add(emb, action, reward, emb, False)

                    episode_rewards.append(reward)
                    action_distribution[action] += 1

                    if i % UPDATE_EVERY == 0:
                        agent.update()

            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue

        # Store results for the episode
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        action_percent = action_distribution / action_distribution.sum() * 100

        episode_results["episode"].append(episode + 1)
        episode_results["reward"].append(avg_reward)
        episode_results["epsilon"].append(agent.epsilon)

        # Store action distribution for each action
        episode_results["action_distribution"]["Drop data"].append(action_percent[0])
        episode_results["action_distribution"]["Transmit raw"].append(action_percent[1])
        episode_results["action_distribution"]["Transmit semantic summary"].append(action_percent[2])
        episode_results["action_distribution"]["Compress and transmit"].append(action_percent[3])
        episode_results["action_distribution"]["Delay transmission"].append(action_percent[4])

        # Print statistics
        print(f"\nEpisode {episode+1}:")
        print(f"Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.2f}")
        print("Action Distribution:")
        for act, percent in zip(ACTIONS.values(), action_percent):
            print(f"{act}: {percent:.1f}%")

        if (episode+1) % 10 == 0:
            agent.save_model(MODEL_SAVE_PATH)

    # Plotting: Episode vs Reward
    plt.figure(figsize=(10, 6))
    plt.plot(episode_results["episode"], episode_results["reward"], label="Avg Reward", color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Episode vs Avg Reward')
    plt.legend()
    # save as pdf
    plt.savefig("episode_vs_avg_reward.pdf")
    plt.show()

    # Plotting: Episode vs Epsilon
    plt.figure(figsize=(10, 6))
    plt.plot(episode_results["episode"], episode_results["epsilon"], label="Epsilon", color='red')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Episode vs Epsilon')
    plt.legend()
    # save as pdf
    plt.savefig("episode_vs_epsilon.pdf")
    # plt.grid(True)
    plt.show()

    # Plotting: Episode vs Action Distribution (all actions)
    actions = list(ACTIONS.values())
    plt.figure(figsize=(12, 8))

    for action in actions:
        plt.plot(episode_results["episode"], episode_results["action_distribution"][action], label=action)

    plt.xlabel('Episode')
    plt.ylabel('Action Distribution (%)')
    plt.title('Episode vs Action Distribution')
    plt.legend()
    # save as pdf
    plt.savefig("episode_vs_action_distribution.pdf")
    # plt.grid(True)
    plt.show()
