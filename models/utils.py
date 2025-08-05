import torch
import torch.nn.functional as F
import numpy as np

def compute_action_token_entropy(action_tokens, temperature=1.0):
    """
    calculate entropy of action tokens
    
    Args:
        action_tokens: torch.Tensor of shape [..., hidden_dim]
                      action token embeddings from transformer output
        temperature: float, temperature for softmax (default=1.0)
    
    Returns:
        entropy: torch.Tensor of shape [...] - entropy for each token
        mean_entropy: float - average entropy across all tokens
    """
    with torch.no_grad():
        # Get the original shape (excluding last dimension)
        original_shape = action_tokens.shape[:-1]
        hidden_dim = action_tokens.shape[-1]
        
        # Flatten to [..., hidden_dim] -> [N, hidden_dim]
        # Use reshape instead of view to handle non-contiguous tensors
        flat_tokens = action_tokens.reshape(-1, hidden_dim)
        
        # Apply temperature scaling
        scaled_tokens = flat_tokens / temperature
        
        # Convert to probability distribution via softmax along hidden_dim
        probs = F.softmax(scaled_tokens, dim=-1)
        
        # Compute entropy: -sum(p * log(p))
        # Use log_softmax for numerical stability
        log_probs = F.log_softmax(scaled_tokens, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
         
        entropy = entropy.reshape(original_shape)
        mean_entropy = entropy.mean().item()
        
        return entropy.detach(), mean_entropy