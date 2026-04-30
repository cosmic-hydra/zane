import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Any

class LeadRLOptimizer:
    """
    Lead Optimization using Reinforcement Learning (Policy Gradient).
    """
    
    def __init__(self, policy_network: nn.Module, property_predictor: Any):
        self.policy = policy_network
        self.predictor = property_predictor
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        
    def train_step(self, states: torch.Tensor, actions: torch.Tensor, smiles_list: List[str]):
        """
        Single RL training step.
        """
        self.optimizer.zero_grad()
        
        # Calculate rewards based on predicted properties
        rewards = []
        for smiles in smiles_list:
            reward = self.predictor.predict(smiles)
            rewards.append(reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
        
        # Policy gradient update (Simplified)
        log_probs = self.policy(states).gather(1, actions.unsqueeze(1))
        loss = -(log_probs * rewards.unsqueeze(1)).mean()
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
