import torch
import torch.nn.functional as F
from ovam.utils.attention_ops import (
    ActivationTypeVar,
    AggregationTypeVar,
    apply_activation,
    apply_aggregation,
)

def forward(
    self, x, remove_special_tokens: bool = False, weight=None
) -> "torch.Tensor":
    """Compute the attention for a given input x.

    Arguments
    ---------
    x : torch.Tensor or str
        The input to compute the attention for. If a string, it is encoded
        using the text encoder.

    Returns
    -------
    torch.Tensor
        The attention heatmaps. Shape: (n_images, n_tokens, block_latent_size[0], block_latent_size[1])
    """
    if isinstance(x, str):
        # Encode text
        x = self.encode_text(x, remove_special_tokens=remove_special_tokens)

    # Fix dimension mismatch: ovam expects (seq_len, dim) but may receive (batch_size, seq_len, dim)
    # If x is 3D, take the first element (single batch) and squeeze to 2D
    if x.dim() == 3:
        if x.shape[0] == 1:
            x = x.squeeze(0)  # (1, seq_len, dim) -> (seq_len, dim)
        else:
            # If batch_size > 1, use the first batch element
            x = x[0]  # (batch_size, seq_len, dim) -> (seq_len, dim)

    attention = []
    for block in self.blocks.values():
        if block.hidden_states:
            attention.append(block.forward(x))
            

    if self.block_latent_size is None:
        # Get the latent size from the first block
        a, b = 0, 0
        for att in attention:
            a = max(a, att.shape[-2])
            b = max(b, att.shape[-1])
        block_latent_size = (a, b)
    else:
        block_latent_size = self.block_latent_size

    # Interpolate all attentions to the same size
    attentions = []
    for att in attention:
        if att.shape[-2:] == block_latent_size:
            # If the attention has the same size as the latent size, do nothing
            attentions.append(att)
            continue
        att = F.interpolate(
            att,
            size=block_latent_size,
            mode=self.block_interpolation_mode,
        )
        attentions.append(att)
    # Remove reference to attention without interpolation
    del attention
    attentions = torch.stack(attentions, dim=0)
    
    
    if weight is not None:
        weight = weight[:,None, None, None, None]
        def weighted_aggregation(tensor):
            # print('stacked attn:',tensor.shape)
            # print('weight for different layers:',weight.shape)
            
            tensor = weight * tensor
            aggregation = self.heatmaps_aggregation
            
            if aggregation == "mean":
                return torch.mean(tensor, dim=0)
            elif aggregation == "sum":
                return torch.sum(tensor, dim=0)
            elif aggregation == "max":
                return torch.max(tensor, dim=0)
            else:
                raise ValueError(f"Unknown activation function: {aggregation}")
                

        attentions = apply_aggregation(
            attentions, weighted_aggregation
        )  # Collapse dim 0
    else:
        attentions = apply_aggregation(
            attentions, self.heatmaps_aggregation
        )  # Collapse dim 0
            

    # Shape (n_images, n_tokens, block_latent_size[0], block_latent_size[1])
    attentions = apply_activation(attentions, self.heatmaps_activation)

    if self.expand_size is not None:
        attentions = F.interpolate(
            attentions,
            size=self.expand_size,
            mode=self.expand_interpolation_mode,
        )
    
    return attentions
