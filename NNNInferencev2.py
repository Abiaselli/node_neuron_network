import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from tkinter import Tk, filedialog, Label, Entry, Button, Text, END, messagebox
import os
import threading
from torch.autograd import Function
import logging
import json

def load_model_parameters(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Saved for reference or later implementation as an option
class MiniTransformerNode(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = nn.Embedding(max_seq_length, embed_size)
        self.vocab_size=vocab_size
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.cross_node_attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)

    def forward(self, x, seq_lengths, prev_node_output=None, src_mask=None, is_final_node=False):
        # Ensure x is within the correct token index range
        x = torch.clamp(x, min=0, max=self.vocab_size - 1)  # Make sure indices are within range

        if x.dim() == 2:  
            embeddings = self.embedding(x)  # Shape: (batch_size, seq_length, embed_size)
        else:  
            embeddings = x  # If already embeddings, use them

        batch_size, seq_length = embeddings.size(0), embeddings.size(1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        pos_encodings = self.pos_encoder(positions)
        pos_encodings = pos_encodings.expand(batch_size, seq_length, -1)

        # Add positional encodings to embeddings
        src = embeddings + pos_encodings

        # Forward through transformer encoder (self-attention)
        output = self.transformer_encoder(src, src_mask)

        # Cross-node attention (global attention) - apply only if there is a previous node
        if prev_node_output is not None:
            output, attention_weights = self.cross_node_attention(output, prev_node_output, prev_node_output, attn_mask=src_mask)
        else:
            # Set attention_weights to None if there's no previous node output
            attention_weights = None

        # Skip connection: add previous node output to current output
        if prev_node_output is not None:
            output = output + prev_node_output

        # Final token prediction layer in the final node
        if is_final_node:
            output = self.fc_out(output)

        return output, attention_weights

#Standard Node network
class CascadeTransformer(nn.Module):
    def __init__(self, num_nodes, vocab_size, embed_size, num_heads, num_layers, hidden_size, max_seq_length):
        super().__init__()
        self.nodes = nn.ModuleList([
            MiniTransformerNode(vocab_size, embed_size, num_heads, num_layers, hidden_size, max_seq_length) 
            for _ in range(num_nodes)
        ])
        self.num_heads=num_heads

    
    def forward(self, x, seq_lengths,  mask=None):
        prev_node_output = None
        attention_weights_all_nodes = []  # To store attention weights from all nodes
        for i, node in enumerate(self.nodes):
            is_final_node = (i == len(self.nodes) - 1)
            x, attention_weights = node(x, seq_lengths, prev_node_output=prev_node_output, src_mask=mask, is_final_node=is_final_node)
            prev_node_output = x  # Store the current output for use in the next node
            attention_weights_all_nodes.append(attention_weights)  # Append attention weights from each node
        return x, attention_weights_all_nodes

#MatMul-Free Node Network
class CascadeTransformerMM(nn.Module):
    def __init__(self, num_nodes, vocab_size, embed_size, hidden_size, num_heads, max_seq_length, eps=1e-8):
        super().__init__()
        self.nodes = nn.ModuleList([
            MatMulFreeLanguageModel(vocab_size, embed_size, hidden_size, num_heads, max_seq_length, eps) 
            for _ in range(num_nodes)
        ])
        self.num_heads=num_heads
        
    
    def forward(self, x, seq_lengths, mask=None):
        prev_node_output = None
        attention_weights_all_nodes = []  # To store attention weights from all nodes
        for i, node in enumerate(self.nodes):
            is_final_node = (i == len(self.nodes) - 1)
            x, attention_weights = node(x, seq_lengths, prev_node_output=prev_node_output, src_mask=mask, is_final_node=is_final_node)
            prev_node_output = x  # Store the current output for use in the next node
            attention_weights_all_nodes.append(attention_weights)  # Append attention weights from each node
        return x, attention_weights_all_nodes

# MatMul-Free Language Model Components

class TernaryWeightFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scaling_factor=None):
        if scaling_factor is None:
            # Compute scaling factor based on mean absolute value
            scaling_factor = 1.0 / (weight.abs().mean() + 1e-5)
            # Clamp scaling factor to prevent numerical instability
            scaling_factor = torch.clamp(scaling_factor, 1e-4, 1e4)

        # Scale and ternarize weights
        scaled_weight = weight * scaling_factor
        # Ensure no NaN or Inf values
        if torch.isnan(scaled_weight).any() or torch.isinf(scaled_weight).any():
            logging.error("Quantized weights contain NaN or Inf.")
            raise ValueError("Quantized weights contain NaN or Inf.")
        
        ternary_weight = torch.sign(scaled_weight)
        
        # Save context for backward pass
        ctx.save_for_backward(weight, ternary_weight, scaling_factor.clone().detach().requires_grad_(True))
        return ternary_weight

    @staticmethod
    def backward(ctx, grad_output):
        weight, ternary_weight, scaling_factor = ctx.saved_tensors
        # Straight-through estimator with scaling factor
        grad_input = grad_output * scaling_factor
        return grad_input, None

def ternarize_weight(weight):
    ternary_weight = TernaryWeightFunction.apply(weight)
    if torch.isnan(ternary_weight).any() or torch.isinf(ternary_weight).any():
        logging.error("Ternarized weights contain NaN or Inf.")
        raise ValueError("Ternarized weights contain NaN or Inf.")
    return ternary_weight


# Updated Weight Quantization
class WeightQuantFunction(Function):
    @staticmethod
    def forward(ctx, weight, num_bits=8):
        """Improved weight quantization with better numerical stability"""
        # Calculate scale based on weight statistics
        max_val = weight.abs().max()
        scale = (2 ** (num_bits - 1) - 1) / (max_val + 1e-5)
        
        # Scale and quantize
        weight_scaled = weight * scale
        weight_clipped = torch.clamp(weight_scaled, -2**(num_bits-1), 2**(num_bits-1)-1)
        weight_rounded = torch.round(weight_clipped)
        
        # Rescale back
        weight_quant = weight_rounded / scale
        
        # Save for backward
        ctx.save_for_backward(weight, weight_quant)
        return weight_quant

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through estimator with gradient clipping"""
        weight, weight_quant = ctx.saved_tensors
        # Clip gradients to stabilize training
        grad_input = torch.clamp(grad_output, -1.0, 1.0)
        return grad_input, None


class MatMulFreeLinearFunction(Function):
    """
    Custom autograd function for a BitNet-style matrix multiplication-free linear layer.
    """
    @staticmethod
    def forward(ctx, input, weight):
        """
        Forward pass using BitNet logic.
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            weight (torch.Tensor): Weight tensor of shape (input_dim, output_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        logging.debug(f"Input to MatMulFreeLinear: mean={input.mean().item():.4f}, std={input.std().item():.4f}, shape={input.shape}")
        if torch.isnan(input).any() or torch.isinf(input).any():
            logging.error("Input contains NaN or Inf.")
            raise ValueError("Input contains NaN or Inf.")

        logging.debug(f"Sample input for MatMulFreeLinearFunction: {input[:10]}")
        # Convert weights to binary representation (sign-based quantization)
        binary_weight = torch.sign(weight)

        # Compute the linear operation without traditional multiplication
        pos_mask = (binary_weight > 0).float()
        neg_mask = (binary_weight < 0).float()
        logging.debug(f"pos_mask: mean={pos_mask.mean().item():.4f}, std={pos_mask.std().item():.4f}, shape={pos_mask.shape}")
        logging.debug(f"neg_mask: mean={neg_mask.mean().item():.4f}, std={neg_mask.std().item():.4f}, shape={neg_mask.shape}")
        logging.debug(f"Sample pos_mask values: {pos_mask.flatten()[:10]}")
        logging.debug(f"Sample neg_mask values: {neg_mask.flatten()[:10]}")
        pos_contrib = torch.matmul(input, pos_mask)
        neg_contrib = torch.matmul(input, neg_mask)
        # Log a sample of predictions and targets
        logging.debug(f"pos_contrib: mean={pos_contrib.mean().item():.4f}, shape={pos_contrib.shape}")
        logging.debug(f"neg_contrib: mean={neg_contrib.mean().item():.4f}, shape={neg_contrib.shape}")
        logging.debug(f"Sample pos_contrib values: {pos_contrib.flatten()[:10]}")
        logging.debug(f"Sample neg_contrib values: {neg_contrib.flatten()[:10]}")
        output = pos_contrib - neg_contrib
        logging.debug(f"output before skip connection: mean={output.mean().item():.4f}, shape={output.shape}")
        logging.debug(f"Sample output values: {output.flatten()[:10]}")

        # Save tensors for backward pass
        ctx.save_for_backward(input, binary_weight.t())
        logging.debug(f"Saved tensors: input shape={input.shape}, pos_mask shape={pos_mask.shape}, neg_mask shape={neg_mask.shape}")


        # Log details of the input tensor being saved
        logging.debug(f"Input to MatMulFreeLinear saved: mean={input.mean().item():.4f}, std={input.std().item():.4f}")
        logging.debug(f"Sample output values: {output.flatten()[:10]}")
        return output


    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for BitNet-style linear operation.
        Args:
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Gradients with respect to input and weight.
        """
        input, binary_weight_t = ctx.saved_tensors  # binary_weight was saved transposed
        logging.debug(f"binary_weight_t shape: {binary_weight_t.shape}")
        # Handle 2D and 3D binary_weight cases
        if binary_weight_t.ndim == 2:
            binary_weight = binary_weight_t  # Shape: [embed_size, vocab_size]
        elif binary_weight_t.ndim == 3:
            binary_weight = binary_weight_t.transpose(-2, -1)  # Swap last two dimensions for 3D
        else:
            raise ValueError(f"Unsupported binary_weight_t dimensions: {binary_weight_t.ndim}")

        logging.debug(f"Gradient input (dO) shape: {grad_output.shape}")
        logging.debug(f"Sample gradient input (dO): {grad_output.flatten()[:10]}")
        logging.debug(f"binary_weight mean: {binary_weight.mean().item():.4f}, std: {binary_weight.std().item():.4f}")
        logging.debug(f"Sample binary_weight values: {binary_weight.flatten()[:10]}")
        logging.debug(f"binary_weight shape: {binary_weight.shape}")

        # Compute gradients
        if grad_output.ndim == 2:  # Standard case
            grad_input = grad_output.matmul(binary_weight)  # Shape: [batch_size * seq_len, embed_size]
            logging.debug(f"Grad_input  shape: {grad_input.shape}")

            grad_weight = grad_output.t().matmul(input)  # Shape: [embed_size, vocab_size]
        elif grad_output.ndim == 3:  # Case for batch processing with 3D tensors
            grad_input = grad_output.matmul(binary_weight.transpose(-2, -1))  # Adjust for 3D
            logging.debug(f"Grad_input  shape: {grad_input.shape}")

            grad_weight = grad_output.transpose(-2, -1).matmul(input)  # Adjust for 3D
        else:
            raise ValueError(f"Unsupported grad_output dimensions: {grad_output.ndim}")
        # Log gradients for debugging

        logging.debug(f"grad_input mean: {grad_input.mean().item():.4f}, std: {grad_input.std().item():.4f}")
        logging.debug(f"Gradient weight shape: {grad_weight.shape}")
        logging.debug(f"grad_weight mean: {grad_weight.mean().item():.4f}, std: {grad_weight.std().item():.4f}")
        # Transpose grad_weight back if needed
        if grad_weight.ndim == 2:
            grad_weight = grad_weight.t()  # Ensure it matches the original weight shape
        elif grad_weight.ndim == 3:
            grad_weight = grad_weight.transpose(-2, -1)  # Adjust for 3D if needed

        logging.debug(f"Adjusted grad_weight shape: {grad_weight.shape}")
        return grad_input, grad_weight

class ActivationQuantFunction(Function):
    @staticmethod
    def forward(ctx, X, max_scale=1e3, min_scale=1e-3):
        """
        Forward pass for Activation Quantization using STE.
        """
        # Compute scaling factor s
        max_val = torch.max(torch.abs(X), dim=-1, keepdim=True)[0]
        s = 127.0 / (max_val + 1e-5)  # Prevent division by zero
        
        # Clamp scaling factor to prevent extreme values
        s = torch.clamp(s, min=min_scale, max=max_scale)
        
        # Quantize
        X_scaled = s * X
        X_quant = torch.clamp(torch.round(X_scaled), -128, 127) / s
        
        # Save tensors for backward
        ctx.save_for_backward(X, s)
        ctx.min_scale = min_scale
        ctx.max_scale = max_scale
        
        return X_quant

    @staticmethod
    def backward(ctx, dX_quant):
        """
        Backward pass for Activation Quantization using STE.
        """
        X, s = ctx.saved_tensors
        # Gradient is passed directly through STE
        dX = dX_quant.clone()
        return dX, None, None

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.scale * (x * rms)

pad_token_id=1

class MLGRULayer(nn.Module):
    """Dirty MLGRUlayer for Matmulfreemodel"""
    def __init__(self, embed_size, hidden_size, eps=1e-5):
        super(MLGRULayer, self).__init__()
        self.cell = MLGRUCell(embed_size, hidden_size, eps)
        self.hidden_size = hidden_size
            
    def forward(self, x, seq_lengths):
        logging.debug(f"Input to MLGRULayer: {x.shape}")
        logging.debug(f"Output of Attention Layer: {x.shape}")

        batch_size, max_seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)
        outputs = []
        try:
            for t in range(max_seq_len):
                x_t = x[:, t, :]
                o_t, h_t = self.cell(x_t, h_t)
                outputs.append(o_t.unsqueeze(1))

            output = torch.cat(outputs, dim=1)
        except Exception as e:
            raise ValueError("Gru kayer failed") 

        outputs = torch.cat(outputs, dim=1)
        logging.debug(f"Shape of outputs after cat: {outputs.shape}")

        # Log statistics after MLGRULayer
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            logging.error("Outputs from MLGRULayer contain NaN or Inf.")
            raise ValueError("Outputs from MLGRULayer contain NaN or Inf.")

        return outputs

class MLGRUCell(nn.Module):
    """MLGRUCell implementation for matmulfreemodel"""
    def __init__(self, input_size, hidden_size, eps=1e-5):
        super(MLGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.rms_norm = RMSNorm(hidden_size, eps=self.eps)

        # Initialize weights
        self.W_f = nn.Parameter(self.ternary_initialize((input_size, hidden_size)))
        self.W_c = nn.Parameter(self.ternary_initialize((input_size, hidden_size)))
        self.W_g = nn.Parameter(self.ternary_initialize((hidden_size, input_size)))

        self.b_f = nn.Parameter(torch.ones(hidden_size))  # Initialize forget gate bias to 1
        self.b_c = nn.Parameter(torch.zeros(hidden_size))
        self.b_g = nn.Parameter(torch.zeros(hidden_size))

        self.initialize_copying_weights()

    def initialize_copying_weights(self):
        """Initialize copying weights for mirror neuron behavior"""
        for name, param in self.named_parameters():
            if "attention" in name:
                param.data.fill_(0.5)  # Bias attention to input

    def ternary_initialize(self, size):
        """Generate a tensor of ternary weights {-1, 0, +1}."""
        # Randomly assign -1, 0, or +1 with equal probability
        rand_tensor = torch.rand(size)  # Random values in [0, 1)
        ternary_tensor = torch.zeros_like(rand_tensor)
        ternary_tensor[rand_tensor < 1/3] = -1  # ~33% probability for -1
        ternary_tensor[rand_tensor > 2/3] = 1   # ~33% probability for +1
        # Remaining ~33% remain 0
        return ternary_tensor


    def forward(self, x_t, h_t_minus_1):
        # RMS Normalization and Activation Quantization
        x_t = self.rms_norm(x_t)
        x_t = ActivationQuantFunction.apply(x_t)

        # Quantize and ternarize weights
        W_f = WeightQuantFunction.apply(self.W_f)
        W_c = WeightQuantFunction.apply(self.W_c)
        W_g = WeightQuantFunction.apply(self.W_g)
        
        W_f_ternary = ternarize_weight(W_f)
        W_c_ternary = ternarize_weight(W_c)
        W_g_ternary = ternarize_weight(W_g)

        # MatMul-Free Linear Operations
        logging.debug(f"Shape of w_f_ternary passed to MMLF: {W_f_ternary.shape}")
        logging.debug(f"Shape of x_t passed to MMLF: {x_t.shape}")

        f_t_linear = MatMulFreeLinearFunction.apply(x_t, W_f_ternary) + self.b_f
        
        logging.debug(f"Shape of w_c_ternary passed to MMLF: {W_c_ternary.shape}")

        c_t_linear = MatMulFreeLinearFunction.apply(x_t, W_c_ternary) + self.b_c
        
        logging.debug(f"Shape of w_g_ternary passed to MMLF: {W_g_ternary.shape}")

        g_t_linear = MatMulFreeLinearFunction.apply(x_t, W_g_ternary) + self.b_g

        # Activation Functions
        f_t = torch.sigmoid(f_t_linear)
        if torch.isnan(f_t).any() or torch.isinf(f_t).any():
            logging.error("f_t contains NaN or Inf after sigmoid in MLGRUCell.")
            raise ValueError("f_t contains NaN or Inf after sigmoid in MLGRUCell.")

        c_t = F.silu(c_t_linear)
        g_t = torch.sigmoid(g_t_linear)
        if torch.isnan(g_t).any() or torch.isinf(g_t).any():
            logging.error("g_t contains NaN or Inf after sigmoid in MLGRUCell.")
            raise ValueError("g_t contains NaN or Inf after sigmoid in MLGRUCell.")

        h_t = f_t * h_t_minus_1 + (1 - f_t) * c_t
        o_t = g_t * h_t

        return o_t, h_t


class MatMulFreeGLU(nn.Module):
    """MatmulfreeGLU mechanism"""
    def __init__(self, input_size, hidden_size, eps=1e-5):
        super(MatMulFreeGLU, self).__init__()
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_g = nn.Parameter(self.ternary_initialize((input_size, hidden_size)))
        self.W_u = nn.Parameter(self.ternary_initialize((input_size, hidden_size)))
        self.W_d = nn.Parameter(self.ternary_initialize((hidden_size, input_size)))
        self.b_g = nn.Parameter(torch.randn(hidden_size))
        self.b_u = nn.Parameter(torch.randn(hidden_size))
        self.b_d = nn.Parameter(torch.randn(hidden_size))
        
        self.rms_norm = RMSNorm(hidden_size, eps=self.eps)
        self.initialize_copying_weights()

    def initialize_copying_weights(self):
        for name, param in self.named_parameters():
            if "attention" in name:
                param.data.fill_(0.5)  # Bias attention to input

    def ternary_initialize(self, size):
        """Generate a tensor of ternary weights {-1, 0, +1}."""
        # Randomly assign -1, 0, or +1 with equal probability
        rand_tensor = torch.rand(size)  # Random values in [0, 1)
        ternary_tensor = torch.zeros_like(rand_tensor)
        ternary_tensor[rand_tensor < 1/3] = -1  # ~33% probability for -1
        ternary_tensor[rand_tensor > 2/3] = 1   # ~33% probability for +1
        # Remaining ~33% remain 0
        return ternary_tensor
    
    def forward(self, x):
        # Apply RMS normalization using custom Function
        x_norm = self.rms_norm(x)
        if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
            logging.error("x_norm contains NaN or Inf after rms_norm in MatMulFreeGLU.")
            raise ValueError("x_norm contains NaN or Inf after rms_norm in MatMulFreeGLU.")

        # Activation Quantization using custom Function
        x_quant = ActivationQuantFunction.apply(x_norm)
        if torch.isnan(x_quant).any() or torch.isinf(x_quant).any():
            logging.error("x_quant contains NaN or Inf after activation_quant in MatMulFreeGLU.")
            raise ValueError("x_quant contains NaN or Inf after activation_quant in MatMulFreeGLU.")

        # Weight Quantization
        W_g_bar = WeightQuantFunction.apply(self.W_g)
        W_u_bar = WeightQuantFunction.apply(self.W_u)
        W_d_bar = WeightQuantFunction.apply(self.W_d)

        # MatMul-Free Linear Operations
        logging.debug(f"Shape of W_G_bar passed to MMLF: {W_g_bar.shape}")
        logging.debug(f"Shape of x passed to MMLF: {x.shape}")

        g_t = MatMulFreeLinearFunction.apply(x, W_g_bar)+self.b_g
        logging.debug(f"Input to MatMulFreeLinear: mean={g_t.mean().item():.4f}, std={g_t.std().item():.4f}, shape={g_t.shape}")
        logging.debug(f"Shape of W_u_bar passed to MMLF: {W_u_bar.shape}")

        u_t = MatMulFreeLinearFunction.apply(x, W_u_bar)+self.b_u
        logging.debug(f"Input to MatMulFreeLinear: mean={u_t.mean().item():.4f}, std={u_t.std().item():.4f}, shape={u_t.shape}")

        p_t = F.silu(g_t) * u_t
        logging.debug(f"Input to MatMulFreeLinear: mean={p_t.mean().item():.4f}, std={p_t.std().item():.4f}, shape={p_t.shape}")
        logging.debug(f"Shape of W_d_bar passed to MMLF: {W_d_bar.shape}")
        logging.debug(f"Shape of p_t passed to MMLF: {p_t.shape}")

        d_t = MatMulFreeLinearFunction.apply(p_t, W_d_bar)+self.b_d
        logging.debug(f"Input to MatMulFreeLinear: mean={d_t.mean().item():.4f}, std={d_t.std().item():.4f}, shape={d_t.shape}")


        # Check for NaN or Inf in output
        if torch.isnan(d_t).any() or torch.isinf(d_t).any():
            logging.error("Output of MatMulFreeGLU contains NaN or Inf.")
            raise ValueError("Output of MatMulFreeGLU contains NaN or Inf.")


        return d_t


class MatMulFreeLanguageModel(nn.Module):
    """MatmukFreeLangiuagemodel concept with multihead attention"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_heads, max_seq_length, eps=1e-5):
        super(MatMulFreeLanguageModel, self).__init__()
        self.eps=eps
        self.seq_length=max_seq_length
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.cross_node_attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        self.mlgru_layer = MLGRULayer(embed_size, hidden_size, eps)
        self.glu = MatMulFreeGLU(hidden_size, hidden_size, eps)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.rms_norm = RMSNorm(hidden_size, eps=eps)
        self.vocab_size=vocab_size
        self.initialize_ternarized_outputs()
        self.initialize_copying_weights()

    def initialize_copying_weights(self):
        for name, param in self.named_parameters():
            if "attention" in name:
                param.data.fill_(0.5)  # Bias attention to input

    def initialize_ternarized_outputs(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'W_' in name:
                    # ternary_initialize expects a shape tuple
                    param.data = self.ternary_initialize(param.data.shape)
                elif 'b_f' in name:
                    nn.init.ones_(param)
                elif 'b_' in name:
                    nn.init.zeros_(param)
                logging.debug(f"Initialized {name} with shape {param.shape}")


    def ternary_initialize(self, size):
        """Generate a tensor of ternary weights {-1, 0, +1}."""
        # Randomly assign -1, 0, or +1 with equal probability
        rand_tensor = torch.rand(size)  # Random values in [0, 1)
        ternary_tensor = torch.zeros_like(rand_tensor)
        ternary_tensor[rand_tensor < 1/3] = -1  # ~33% probability for -1
        ternary_tensor[rand_tensor > 2/3] = 1   # ~33% probability for +1
        # Remaining ~33% remain 0
        return ternary_tensor


    def forward(self, input_ids, seq_length, prev_node_output=None, src_mask=None, is_final_node=False):
        logging.debug(f"Input IDs shape: {input_ids.shape}")
        input_ids = torch.clamp(input_ids, min=0, max=self.vocab_size - 1)  # Make sure indices are within range

        # Embedding Layer
        if input_ids.dim() == 2:  
            x = self.embedding(input_ids.long())  # Shape: (batch_size, seq_length, embed_size)
        else:  
            x = input_ids  # If already embeddings, use them
        logging.debug(f"X passed to attention: mean={x.mean().item():.4f}, std={x.std().item():.4f}, shape={x.shape}")

        # MLGRULayer
        x = self.mlgru_layer(x, seq_length)
        logging.debug(f"X passed to GLU: mean={x.mean().item():.4f}, std={x.std().item():.4f}, shape={x.shape}")

        # MatMulFreeGLU
        x = self.glu(x)
        logging.debug(f"X passed to RMSNORM: mean={x.mean().item():.4f}, std={x.std().item():.4f}, shape={x.shape}")

        # Check if x is finite before RMS normalization
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.error("x contains NaN or Inf before rms_norm.")
            raise ValueError("x contains NaN or Inf before rms_norm.")

        # RMS Normalization using custom autograd Function
        x = self.rms_norm(x)
        logging.debug(f"X passed to Activationquant: mean={x.mean().item():.4f}, std={x.std().item():.4f}, shape={x.shape}")

        # Check for NaN or Inf after RMS normalization
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.error("x contains NaN or Inf after rms_norm.")
            raise ValueError("x contains NaN or Inf after rms_norm.")

        # Activation Quantization using custom autograd Function
        x = ActivationQuantFunction.apply(x)
        logging.debug(f"X passed to W_bar: mean={x.mean().item():.4f}, std={x.std().item():.4f}, shape={x.shape}")
        batch_size, seq_len, embed_size = x.shape

        # Check for NaN or Inf after activation quantization
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.error("x contains NaN or Inf after activation_quant.")
            raise ValueError("x contains NaN or Inf after activation_quant.")

        # Weight Quantization using custom autograd Function
        W_bar = WeightQuantFunction.apply(self.output_layer.weight)
        logging.debug(f"W_bar passed to Logits: mean={x.mean().item():.4f}, std={x.std().item():.4f}, shape={x.shape}")
        output=x

        # Cross-node attention (global attention) - apply only if there is a previous node
        if prev_node_output is not None:
            output, attention_weights = self.cross_node_attention(output, prev_node_output, prev_node_output, attn_mask=src_mask)
            logging.debug(f"Shape of output: {output.shape}")
            logging.debug(f"Shape of attention_weights: {attention_weights.shape}")

        else:
            # Set attention_weights to None if there's no previous node output
            attention_weights = None

        # Skip connection: add previous node output to current output
        if prev_node_output is not None:
            output = output + prev_node_output

        # Final token prediction layer in the final node
        if is_final_node:
            output = self.output_layer(output)

            
        logits = output


        # Check for NaN or Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logging.error("Logits contain NaN or Inf after matmul_free_linear.")
            raise ValueError("Logits contain NaN or Inf after matmul_free_linear.")

        return logits, attention_weights

# Generate src mask function
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# Top-K and Top-P Filtering
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    batch_size, vocab_size = logits.size()
    
    # Apply top-k filtering
    if top_k > 0:
        top_k = min(max(top_k, 1), vocab_size)
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.tensor(filter_value, device=logits.device), logits)
    
    # Apply top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # Scatter to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    
    return logits

# Tokenizer Validation and Loading
def validate_tokenizer_folder(tokenizer_path):
    required_files = ["tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(tokenizer_path, f))]
    if missing_files:
        raise FileNotFoundError(f"Missing files in tokenizer folder: {missing_files}")

def load_tokenizer(tokenizer_path):
    validate_tokenizer_folder(tokenizer_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    print(f"Special tokens loaded: {tokenizer.special_tokens_map}")
    return tokenizer

def ensure_special_tokens(tokenizer):
    special_tokens = {}
    if tokenizer.eos_token is None:
        special_tokens['eos_token'] = '<eos>'
    if tokenizer.pad_token is None:
        special_tokens['pad_token'] = '<pad>'
    
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print(f"Added special tokens: {special_tokens}")
    else:
        print("All special tokens are already present.")
    
    print(f"EOS Token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    print(f"PAD Token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
    return tokenizer

# Model Loading Function
def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if 'state_dict' in checkpoint and 'model_parameters' in checkpoint:
        # New model format with parameters included
        state_dict = checkpoint['state_dict']
        model_parameters = checkpoint['model_parameters']
    else:
        # Old model format without parameters
        state_dict = checkpoint
        model_parameters = None

    return state_dict, model_parameters




# Text Generation Function
def generate_text_gui(model, tokenizer, input_text, max_length=50, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0):
    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    # Remove or adjust the truncation as needed
    generated = input_ids.clone()
    seq_length = input_ids.size(1)  # Infer sequence length dynamically from input
    print(f"Input IDs: {input_ids}, Shape: {input_ids.shape}, Type: {input_ids.dtype}")
    print(f"Seq Length: {seq_length}")
    with torch.no_grad():
        for _ in range(max_length):
            logits, attention_weights = model(generated, seq_length)  # Unpack the tuple
            next_token_logits = logits[:, -1, :]  # ✅ Now this will work because logits is a tensor


            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Repetition Penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated.view(-1).tolist()):
                    next_token_logits[:, token_id] /= repetition_penalty

            # Filter logits using top-k and/or top-p sampling
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

            # Sample from the filtered distribution
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)

            # Append to generated tokens
            generated = torch.cat((generated, next_token_id), dim=1)

            # Stop if the EOS token is generated
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    # Move generated tokens back to CPU before decoding
    generated = generated.cpu()
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output_text


#GUI Implementation
class LanguageModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MatMul-Free Language Model")

        # Initialize model and tokenizer as None
        self.model = None
        self.tokenizer = None

        # Define Entry widgets for model and tokenizer paths
        Label(root, text="Model Path:").pack(pady=(10, 0))
        self.model_path_entry = Entry(root, width=60)
        self.model_path_entry.pack(pady=(0, 10))

        Label(root, text="Tokenizer Path:").pack(pady=(0, 0))
        self.tokenizer_path_entry = Entry(root, width=60)
        self.tokenizer_path_entry.pack(pady=(0, 10))

        # Select Folder Button
        self.select_button = Button(root, text="Select Model Folder", command=self.select_folder)
        self.select_button.pack(pady=(0, 10))

        # Model Parameters
        Label(root, text="Vocabulary Size:").pack(pady=(10, 0))
        self.vocab_size_entry = Entry(root, width=60)
        self.vocab_size_entry.pack(pady=(0, 10))
        self.vocab_size_entry.insert(0, "30000")  # Default value

        Label(root, text="Embedding Size:").pack(pady=(0, 0))
        self.embed_size_entry = Entry(root, width=60)
        self.embed_size_entry.pack(pady=(0, 10))
        self.embed_size_entry.insert(0, "60")  # Default value

        Label(root, text="Hidden Size:").pack(pady=(0, 0))
        self.hidden_size_entry = Entry(root, width=60)
        self.hidden_size_entry.pack(pady=(0, 10))
        self.hidden_size_entry.insert(0, "60")  # Default value

        Label(root, text="Nodes:").pack(pady=(0, 0))
        self.num_nodes_entry = Entry(root, width=60)
        self.num_nodes_entry.pack(pady=(0, 10))
        self.num_nodes_entry.insert(0, "4")  # Default value
        
        Label(root, text="Heads:").pack(pady=(0, 0))
        self.num_heads_entry = Entry(root, width=60)
        self.num_heads_entry.pack(pady=(0, 10))
        self.num_heads_entry.insert(0, "6")  # Default value

        # Input Text
        Label(root, text="Input Text:").pack(pady=(10, 0))
        self.input_box = Text(root, height=5, width=60)
        self.input_box.pack(pady=(0, 10))

        # Generation Parameters
        Label(root, text="Max Length:").pack(pady=(10, 0))
        self.max_length_entry = Entry(root, width=60)
        self.max_length_entry.pack(pady=(0, 10))
        self.max_length_entry.insert(0, "50")

        Label(root, text="Temperature:").pack(pady=(0, 0))
        self.temperature_entry = Entry(root, width=60)
        self.temperature_entry.pack(pady=(0, 10))
        self.temperature_entry.insert(0, "1.0")

        Label(root, text="Top-K:").pack(pady=(0, 0))
        self.top_k_entry = Entry(root, width=60)
        self.top_k_entry.pack(pady=(0, 10))
        self.top_k_entry.insert(0, "0")

        Label(root, text="Top-P:").pack(pady=(0, 0))
        self.top_p_entry = Entry(root, width=60)
        self.top_p_entry.pack(pady=(0, 10))
        self.top_p_entry.insert(0, "0.0")

        Label(root, text="Repetition Penalty:").pack(pady=(0, 0))
        self.repetition_penalty_entry = Entry(root, width=60)
        self.repetition_penalty_entry.pack(pady=(0, 10))
        self.repetition_penalty_entry.insert(0, "1.0")

        # Generate Button
        self.generate_button = Button(root, text="Generate Text", command=self.generate_text_callback)
        self.generate_button.pack(pady=(0, 10))

        # Output Box
        Label(root, text="Generated Output:").pack(pady=(10, 0))
        self.output_box = Text(root, height=10, width=60)
        self.output_box.pack(pady=(0, 10))
        # Select log file path
        self.select_log_file()

        # Setup logging
        logging.basicConfig(filename=self.log_file_path, level=logging.ERROR,
                            format='%(asctime)s - %(levelname)s - %(message)s')


    def select_log_file(self):
        self.log_file_path = filedialog.asksaveasfilename(
            title="Select Log File Location",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        if self.log_file_path:
            print(f"Log file will be saved to: {self.log_file_path}")
        else:
            self.log_file_path = 'training_debug.log'  # Default log file
            print(f"No log file selected. Using default: {self.log_file_path}")
            
    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            # Set model and tokenizer paths
            model_path = os.path.join(folder_path, "matmul_free_lm.pth")
            tokenizer_path = folder_path  # Assuming tokenizer files are in the same folder

            # Update Entry widgets
            self.model_path_entry.delete(0, END)
            self.model_path_entry.insert(0, model_path)

            self.tokenizer_path_entry.delete(0, END)
            self.tokenizer_path_entry.insert(0, tokenizer_path)

            # Load model and tokenizer
            try:
                self.load_model_and_tokenizer(model_path, tokenizer_path)
                messagebox.showinfo("Success", "Model and Tokenizer loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model/tokenizer:\n{e}")

    def load_model_and_tokenizer(self, model_path, tokenizer_path):
        # Load tokenizer
        tokenizer = load_tokenizer(tokenizer_path)
        tokenizer = ensure_special_tokens(tokenizer)

        # Load model parameters from model_config.json
        config_path = os.path.join(os.path.dirname(model_path), 'model_config.json')
        if not os.path.exists(config_path):
            messagebox.showerror("Error", "model_config.json not found.")
            return

        model_parameters = load_model_parameters(config_path)

        # Update Entry widgets with loaded parameters
        self.vocab_size_entry.config(state='normal')
        self.vocab_size_entry.delete(0, END)
        self.vocab_size_entry.insert(0, str(model_parameters['vocab_size']))
        self.vocab_size_entry.config(state='readonly')

        self.embed_size_entry.config(state='normal')
        self.embed_size_entry.delete(0, END)
        self.embed_size_entry.insert(0, str(model_parameters['embed_size']))
        self.embed_size_entry.config(state='readonly')

        self.hidden_size_entry.config(state='normal')
        self.hidden_size_entry.delete(0, END)
        self.hidden_size_entry.insert(0, str(model_parameters['hidden_size']))
        self.hidden_size_entry.config(state='readonly')

        self.num_nodes_entry.config(state='normal')
        self.num_nodes_entry.delete(0, END)
        self.num_nodes_entry.insert(0, str(model_parameters['num_nodes']))
        self.num_nodes_entry.config(state='readonly')
        
        self.num_heads_entry.config(state='normal')
        self.num_heads_entry.delete(0, END)
        self.num_heads_entry.insert(0, str(model_parameters['num_heads']))
        self.num_heads_entry.config(state='readonly')
        
        # Create the appropriate model based on the architecture
        architecture = model_parameters.get('architecture', "Cascade Transformer" or 'Cascade MatMul-Free LM')

        if architecture == 'Cascade Transformer':
            model = CascadeTransformer(
                vocab_size=model_parameters['vocab_size'],
                embed_size=model_parameters['embed_size'],
                hidden_size=model_parameters['hidden_size'],
                num_nodes=model_parameters['num_nodes'],
                num_heads=model_parameters['num_heads'],
                num_layers=model_parameters['num_layers'],
                max_seq_length=1024
            )
            # Adjust model path if needed
            model_path = os.path.join(os.path.dirname(model_path), 'cascade_transformer_model.pth')
        elif architecture =='Cascade MatMul-Free LM':
            model = CascadeTransformerMM(
                vocab_size=model_parameters['vocab_size'],
                embed_size=model_parameters['embed_size'],
                hidden_size=model_parameters['hidden_size'],
                num_nodes=model_parameters['num_nodes'],
                num_heads=model_parameters['num_heads'],
                max_seq_length=1024
            )

            # Adjust model path if needed
            model_path = os.path.join(os.path.dirname(model_path), 'cascade_matmul_free_lm.pth')
        else:
            messagebox.showerror("Error", f"Unsupported architecture: {architecture}")
            return

        print(f"Model Parameters:")
        print(f"  Vocab Size: {model_parameters['vocab_size']}")
        print(f"  Embed Size: {model_parameters['embed_size']}")
        print(f"  Hidden Size: {model_parameters['hidden_size']}")
        print(f"  Num Heads: {model_parameters['num_heads']}")
        print(f"  Num Layers: {model_parameters['num_layers']}")
        # Load state_dict
        state_dict, _ = load_model(model_path, device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        # Update class attributes
        self.tokenizer = tokenizer
        self.model = model





    def generate_text_callback(self):
        if self.model is None or self.tokenizer is None:
            messagebox.showwarning("Warning", "Please load a model and tokenizer first.")
            return

        input_text = self.input_box.get("1.0", END).strip()
        if not input_text:
            messagebox.showwarning("Warning", "Please enter some input text.")
            return

        # Retrieve generation parameters
        try:
            max_length = int(self.max_length_entry.get())
            temperature = float(self.temperature_entry.get())
            top_k = int(self.top_k_entry.get())
            top_p = float(self.top_p_entry.get())
            repetition_penalty = float(self.repetition_penalty_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid generation parameters.")
            return

        # Start generation in a separate thread to keep GUI responsive
        threading.Thread(
            target=self.generate_and_display,
            args=(input_text, max_length, temperature, top_k, top_p, repetition_penalty)
        ).start()

    def generate_and_display(self, input_text, max_length, temperature, top_k, top_p, repetition_penalty):
        try:
            output = generate_text_gui(
                model=self.model,
                tokenizer=self.tokenizer,
                input_text=input_text,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            self.output_box.delete("1.0", END)
            self.output_box.insert(END, output)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate text:\n{e}")

def main():
    root = Tk()
    gui = LanguageModelGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
