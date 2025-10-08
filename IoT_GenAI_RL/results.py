import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from IoT import*

def load_test_data(csv_path=r"processed_wisdm/test.csv"):
    df = pd.read_csv(csv_path)
    prompts = df["prompt"].tolist()
    labels = df["label"].tolist()
    return prompts, labels

def evaluate_agent(agent, prompts, labels, tokenizer, model, max_samples=100):
    test_rewards = []
    test_actions = []
    action_distribution = np.zeros(len(ACTIONS))

    for i in range(min(max_samples, len(prompts))):
        emb = extract_semantic_embedding(prompts[i], tokenizer, model)
        action = agent.select_action(emb, evaluate=True)
        reward = calculate_reward(action, emb)

        test_rewards.append(reward)
        test_actions.append(action)
        action_distribution[action] += 1

        print(f"Sample {i+1} ({labels[i]}): {ACTIONS[action]} | Reward: {reward:.3f}")

    avg_reward = np.mean(test_rewards)
    print(f"\n✅ Average Reward: {avg_reward:.3f}")

    print("\n📊 Action Distribution:")
    total = action_distribution.sum()
    for idx, count in enumerate(action_distribution):
        percent = (count / total) * 100 if total > 0 else 0
        print(f"{ACTIONS[idx]}: {int(count)} samples ({percent:.1f}%)")

    return test_rewards, test_actions

def plot_action_distribution(test_actions):
    counts = np.bincount(test_actions, minlength=len(ACTIONS))
    labels = [ACTIONS[i] for i in range(len(ACTIONS))]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts, color='skyblue')
    plt.title("Action Distribution")
    plt.xlabel("Actions")
    plt.ylabel("Count")
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_reward_trend(rewards):
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, marker='o', linestyle='-', color='green')
    plt.title("Reward per Sample")
    plt.xlabel("Sample Index")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🧠 Evaluating trained DQN agent on test.csv...\n")

    # Load test prompts & labels
    prompts, labels = load_test_data(r"processed_wisdm/test.csv")

    # Load Mistral tokenizer & model
    tokenizer, model = load_model()

    # Initialize and load agent
    test_embedding = extract_semantic_embedding(prompts[0], tokenizer, model)
    agent = DQNAgent(test_embedding.shape[0], len(ACTIONS))
    agent.load_model(MODEL_SAVE_PATH)

    # Evaluate
    test_rewards, test_actions = evaluate_agent(agent, prompts, labels, tokenizer, model, max_samples=100)

    # Plot results
    plot_action_distribution(test_actions)
    plot_reward_trend(test_rewards)
