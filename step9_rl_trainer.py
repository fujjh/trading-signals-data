#!/usr/bin/env python3
"""
================================================================================
STEP 9: SPY Deep Reinforcement Learning (PPO)
================================================================================

Proximal Policy Optimization (PPO) for trading SPY.
Uses actor-critic architecture to learn optimal trading policy through
interaction with market environment.

Author: SignalsAlpha
Version: 1.0 (RL Edition)
Date: 2026-04-29

================================================================================
APPROACH:
================================================================================

1. Define trading environment (OpenAI Gym style)
2. PPO agent learns policy: state -> action (buy/sell/hold)
3. Reward function based on returns and risk-adjusted metrics
4. Compare against GA, XGBoost, and LSTM strategies
================================================================================
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Install with: pip install torch")

# Configuration
TICKER = "SPY"
TECHNICAL_DIR = Path("data/technical_analysis") / TICKER
OUTPUT_DIR = Path("data/optimizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RL Parameters
INITIAL_CAPITAL = 100000
MAX_POSITION_PCT = 0.2
TRANSACTION_COST = 0.001  # 0.1%

# PPO Parameters
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # Generalized Advantage Estimation
CLIP_EPSILON = 0.2
CRITIC_COEF = 0.5
ENTROPY_COEF = 0.01
BATCH_SIZE = 64
N_EPOCHS = 10
N_EPISODES = 1000


def load_spy_data():
    """Load SPY technical data."""
    tech_file = TECHNICAL_DIR / f"{TICKER}_1d_technical.csv"
    if not tech_file.exists():
        print(f"ERROR: Technical file not found: {tech_file}")
        return None
    
    df = pd.read_csv(tech_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df


def create_features(df: pd.DataFrame) -> np.ndarray:
    """Create feature set for RL agent."""
    features = pd.DataFrame(index=df.index)
    
    # Price features (normalized)
    features['returns_1d'] = df['close'].pct_change()
    features['returns_5d'] = df['close'].pct_change(5)
    features['volatility'] = features['returns_1d'].rolling(20).std()
    
    # Technical indicators
    if 'rsi' in df.columns:
        features['rsi'] = df['rsi'] / 100.0
    
    if 'macd' in df.columns:
        features['macd'] = df['macd'] / df['close']
    
    if 'sma_20' in df.columns:
        features['price_vs_sma20'] = df['close'] / df['sma_20'] - 1
    if 'sma_200' in df.columns:
        features['price_vs_sma200'] = df['close'] / df['sma_200'] - 1
    
    if 'pct_from_52w_high' in df.columns:
        features['pct_from_52w_high'] = df['pct_from_52w_high'] / 100.0
    
    # Volume
    if 'volume' in df.columns:
        features['volume_norm'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Fill NaN
    features = features.fillna(0)
    
    return features.values


class TradingEnvironment:
    """Trading environment for RL."""
    
    def __init__(self, df: pd.DataFrame, features: np.ndarray, initial_capital: float = 100000):
        self.df = df.reset_index(drop=True)
        self.features = features
        self.initial_capital = initial_capital
        self.max_position = initial_capital * MAX_POSITION_PCT
        
        self.n_features = features.shape[1]
        self.n_actions = 3  # 0=hold, 1=buy, 2=sell
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.cash = self.initial_capital
        self.position = 0  # Number of shares
        self.portfolio_value = self.initial_capital
        self.done = False
        self.trades = []
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current state observation."""
        obs = self.features[self.current_step].copy()
        # Add portfolio state
        obs = np.append(obs, [
            self.position / (self.max_position / self.df['close'].iloc[self.current_step]),  # Position ratio
            self.cash / self.initial_capital,  # Cash ratio
        ])
        return obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info."""
        if self.done:
            return self._get_observation(), 0, True, {}
        
        current_price = self.df['close'].iloc[self.current_step]
        reward = 0
        
        # Execute action
        if action == 1:  # Buy
            max_shares = min(self.cash * (1 - TRANSACTION_COST) / current_price, 
                           self.max_position / current_price)
            if max_shares > 0:
                cost = max_shares * current_price * (1 + TRANSACTION_COST)
                self.position += max_shares
                self.cash -= cost
                self.trades.append(('buy', self.current_step, current_price, max_shares))
        
        elif action == 2:  # Sell
            if self.position > 0:
                proceeds = self.position * current_price * (1 - TRANSACTION_COST)
                self.cash += proceeds
                self.trades.append(('sell', self.current_step, current_price, self.position))
                self.position = 0
        
        # Move to next step
        self.current_step += 1
        
        # Calculate portfolio value
        if self.current_step < len(self.df):
            next_price = self.df['close'].iloc[self.current_step]
            self.portfolio_value = self.cash + self.position * next_price
            
            # Calculate reward (daily return)
            prev_value = self.portfolio_value / (1 + self.df['close'].pct_change().iloc[self.current_step])
            daily_return = (self.portfolio_value - prev_value) / prev_value
            
            # Risk-adjusted reward (Sharpe-like)
            reward = daily_return * 100  # Scale up
            
            # Penalty for being flat too long
            if len(self.trades) == 0 and self.current_step > 100:
                reward -= 0.1
        
        # Check if done
        if self.current_step >= len(self.df) - 1:
            self.done = True
            # Final reward based on total return
            total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
            reward += total_return * 10  # Bonus for total performance
        
        return self._get_observation(), reward, self.done, {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'cash': self.cash
        }


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int = 128):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_probs, value


def run_ppo_training():
    """Main PPO training loop."""
    print("=" * 70)
    print("STEP 9: SPY Deep Reinforcement Learning (PPO)")
    print("=" * 70)
    
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not installed. Install with: pip install torch")
        return
    
    df = load_spy_data()
    if df is None:
        return
    
    print(f"Loaded {len(df)} rows")
    print(f"Initial capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Max position: {MAX_POSITION_PCT*100:.0f}%")
    print("=" * 70)
    
    # Create features
    features = create_features(df)
    print(f"Features: {features.shape[1]}")
    
    # Split train/test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    train_features = features[:split_idx]
    test_features = features[split_idx:]
    
    print(f"\nTrain: {len(train_df)} days")
    print(f"Test: {len(test_df)} days")
    
    # Initialize
    env = TradingEnvironment(train_df, train_features, INITIAL_CAPITAL)
    input_dim = env.n_features + 2  # +2 for position and cash ratios
    
    model = ActorCritic(input_dim, env.n_actions)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    
    print("\n" + "=" * 70)
    print("Training PPO...")
    print("=" * 70)
    
    for episode in range(N_EPISODES):
        state = env.reset()
        states, actions, rewards, values, dones = [], [], [], [], []
        
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs, value = model(state_tensor)
            
            dist = Categorical(action_probs)
            action = dist.sample()
            
            next_state, reward, done, info = env.step(action.item())
            
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)
            
            state = next_state
            
            if len(states) >= BATCH_SIZE:
                break
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t] if not dones[t] else 0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + GAMMA * next_value * next_non_terminal - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # PPO update
        for _ in range(N_EPOCHS):
            action_probs, values = model(states)
            dist = Categorical(action_probs)
            
            # Calculate ratio
            old_action_probs = action_probs.detach().gather(1, actions.unsqueeze(1)).squeeze()
            new_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
            ratio = new_action_probs / (old_action_probs + 1e-10)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = actor_loss + CRITIC_COEF * critic_loss - ENTROPY_COEF * entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Track metrics
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        episode_lengths.append(len(rewards))
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{N_EPISODES} | Avg Reward: {avg_reward:.2f} | "
                  f"Portfolio: ${info['portfolio_value']:,.0f}")
    
    # Test on held-out data
    print("\n" + "=" * 70)
    print("Testing on Held-Out Data")
    print("=" * 70)
    
    test_env = TradingEnvironment(test_df, test_features, INITIAL_CAPITAL)
    state = test_env.reset()
    
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = model(state_tensor)
        action = torch.argmax(action_probs).item()
        state, _, done, info = test_env.step(action)
    
    # Calculate metrics
    final_value = info['portfolio_value']
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL
    bh_return = (test_df['close'].iloc[-1] / test_df['close'].iloc[0] - 1)
    
    print(f"Final Portfolio: ${final_value:,.2f}")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Buy & Hold Return: {bh_return*100:.2f}%")
    print(f"Outperformance: {(total_return - bh_return)*100:.2f}%")
    print(f"Number of trades: {len(test_env.trades)}")
    
    # Save model
    model_file = OUTPUT_DIR / f"spy_ppo_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode_rewards': episode_rewards,
    }, str(model_file))
    
    # Save config
    config_file = OUTPUT_DIR / f"spy_ppo_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_file, 'w') as f:
        json.dump({
            'ticker': TICKER,
            'initial_capital': INITIAL_CAPITAL,
            'max_position_pct': MAX_POSITION_PCT,
            'transaction_cost': TRANSACTION_COST,
            'n_episodes': N_EPISODES,
            'final_return': float(total_return),
            'buy_hold_return': float(bh_return),
            'outperformance': float(total_return - bh_return),
            'n_trades': len(test_env.trades),
            'avg_episode_reward': float(np.mean(episode_rewards[-100:])),
            'generated_at': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nSaved to: {config_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_ppo_training()
