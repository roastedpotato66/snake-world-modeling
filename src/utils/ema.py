"""Exponential Moving Average for model weights."""

import torch


class EMA:
    """Exponential Moving Average of model weights."""
    
    def __init__(self, model, decay=0.9999, warmup=1000):
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        """Update EMA weights."""
        self.step += 1
        # Linear warmup of decay
        decay = min(self.decay, (1 + self.step) / (self.warmup + self.step))
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].lerp_(param.data, 1 - decay)
    
    def apply(self, model):
        """Apply EMA weights to model (for eval/inference)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
    
    def state_dict(self):
        """Get EMA state dict."""
        return {'shadow': self.shadow, 'step': self.step}
    
    def load_state_dict(self, state):
        """Load EMA state dict."""
        self.shadow = state['shadow']
        self.step = state['step']

