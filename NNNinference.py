import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from tkinter import Tk, filedialog, Label, Entry, Button, Text, END, messagebox
import os
import threading
from torch.quantization import QuantStub, DeQuantStub, fuse_modules

import json

def load_model_parameters(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# RMS Normalization Function
class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, unbiased=False, keepdim=True)
        r = 1 / torch.sqrt(torch.clamp(variance + eps, min=1e-10))  # Prevent division by zero
        y = r * (x - mean)
        ctx.save_for_backward(x, mean, variance, r)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, mean, variance, r = ctx.saved_tensors
        eps = ctx.eps
        N = x.shape[-1]
        denom = variance + eps
        denom = torch.clamp(denom, min=1e-8)  # Ensure denom is not too small
        grad_input = (1 / N) * r * (
            N * grad_output
            - grad_output.sum(dim=-1, keepdim=True)
            - (x - mean) * ((grad_output * (x - mean)).sum(dim=-1, keepdim=True) / denom)
        )
        return grad_input, None


def rms_norm(x, eps=1e-8):
    return RMSNormFunction.apply(x, eps)

# Activation quantization function
def activation_quant(x, bits=8):
    if torch.isnan(x).any():
        x = torch.nan_to_num(x, nan=0.0)
    qmin = -2**(bits - 1)
    qmax = 2**(bits - 1) - 1
    x_abs_max = x.abs().max()
    if x_abs_max == 0 or torch.isnan(x_abs_max):
        scale = 1.0  # Avoid division by zero
    else:
        scale = x_abs_max / qmax
    x_quant = torch.clamp((x / scale).round(), qmin, qmax)
    x_dequant = x_quant * scale
    return x_dequant

# Custom Ternary Weight Function
class TernaryWeightFunction(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, weight):
        # Ternarize weights to -1, 0, or +1
        ternary_weight = torch.sign(weight)
        return ternary_weight

    @staticmethod
    def backward(_ctx, grad_output):
        # Gradient is passed through unchanged
        grad_input = grad_output.clone()
        return grad_input

def ternarize_weight(weight):
    return TernaryWeightFunction.apply(weight)

# Matmul-free linear function with quantization
def matmul_free_linear(input, weight):
    # Quantize input and weight
    input_q = activation_quant(input)
    weight_q = ternarize_weight(weight)
    
    # Perform matrix multiplication
    output = input_q.matmul(weight_q.t())
    return output

# MatMul-free Linear Gated Recurrent Unit (MLGRU) Cell
class MLGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        # Weights and biases
        self.W_f = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_c = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_g = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_f = nn.Parameter(torch.randn(hidden_size))
        self.b_c = nn.Parameter(torch.randn(hidden_size))
        self.b_g = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x_t, h_t_minus_1):
        # Apply RMS normalization
        x_t = rms_norm(x_t, self.eps)

        # Linear operations
        f_t_linear = matmul_free_linear(x_t, self.W_f) + self.b_f
        c_t_linear = matmul_free_linear(x_t, self.W_c) + self.b_c
        g_t_linear = matmul_free_linear(x_t, self.W_g) + self.b_g

        # Activation functions
        sig_f_t = torch.sigmoid(f_t_linear)
        silu_c_t = F.silu(c_t_linear)
        sig_g_t = torch.sigmoid(g_t_linear)

        # Hidden state computations
        h_t = sig_f_t * h_t_minus_1 + (1 - sig_f_t) * silu_c_t
        o_t = h_t * sig_g_t

        return o_t, h_t


# MLGRU Layer
class MLGRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super().__init__()
        self.cell = MLGRUCell(input_size, hidden_size, eps)
        self.hidden_size = hidden_size

    def forward(self, x):

        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            o_t, h_t = self.cell(x_t, h_t)
            outputs.append(o_t.unsqueeze(1))

        output = torch.cat(outputs, dim=1)
        return output


# MatMul-free GLU
class MatMulFreeGLU(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super().__init__()
        self.eps = eps

        self.W_g = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_u = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_d = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_g = nn.Parameter(torch.randn(hidden_size))
        self.b_u = nn.Parameter(torch.randn(hidden_size))
        self.b_d = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        # Apply RMS normalization
        x = rms_norm(x, self.eps)
        # Quantize activations
        x = activation_quant(x)

        # Linear operations
        g_t = matmul_free_linear(x, self.W_g) + self.b_g
        u_t = matmul_free_linear(x, self.W_u) + self.b_u

        # Activation functions
        g_t = F.silu(g_t)
        p_t = g_t * u_t  # Assuming linear activation

        # Output layer
        d_t = matmul_free_linear(p_t, self.W_d) + self.b_d

        return d_t

#Saved for reference or later implementation as an option
class MiniTransformerNode(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, hidden_size, max_seq_length=128):
        super().__init__()
        self.embedding = nn.Embedding(30000, embed_size)
        self.pos_encoder = nn.Embedding(max_seq_length, embed_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_size, 30000)
        self.cross_node_attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)

    def forward(self, x, prev_node_output=None, src_mask=None, is_final_node=False):
        # Ensure x is within the correct token index range
        x = torch.clamp(x.long(), min=0, max=29999)

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
            output, attention_weights = self.cross_node_attention(output, prev_node_output, prev_node_output)
        else:
            # Set attention_weights to None if there's no previous node output
            attention_weights = None

        # Skip connection: add previous node output to current output
        if prev_node_output is not None:
            output = output + prev_node_output

        # Final token prediction layer in the final node
        if is_final_node:
            output = self.fc_out(output) + self.fc_out.bias

        return output, attention_weights


class CascadeTransformer(nn.Module):
    def __init__(self, num_nodes, vocab_size, embed_size, hidden_size, num_heads, eps=1e-8):
        super().__init__()
        self.nodes = nn.ModuleList([
            MatMulFreeLanguageModel(vocab_size, embed_size, hidden_size, num_heads, eps) 
            for _ in range(num_nodes)
        ])
        self.num_heads=num_heads
        
    
    def forward(self, x, mask=None):
        prev_node_output = None
        attention_weights_all_nodes = []  # To store attention weights from all nodes
        for i, node in enumerate(self.nodes):
            is_final_node = (i == len(self.nodes) - 1)
            x, attention_weights = node(x, prev_node_output=prev_node_output, src_mask=mask, is_final_node=is_final_node)
            prev_node_output = x  # Store the current output for use in the next node
            attention_weights_all_nodes.append(attention_weights)  # Append attention weights from each node
        return x, attention_weights_all_nodes

# MatMul-Free Language Model
class MatMulFreeLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_heads, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.mlgru_layer = MLGRULayer(embed_size, hidden_size, eps)
        self.glu = MatMulFreeGLU(hidden_size, hidden_size, eps)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.cross_node_attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)

    def forward(self, input_ids, prev_node_output=None, src_mask=None, is_final_node=False):
        
        if input_ids.dim() == 2:  
            x = self.embedding(input_ids.long())  # Shape: (batch_size, seq_length, embed_size)
        else:  
            x = input_ids  # If already embeddings, use them
            
        x = self.mlgru_layer(x)

        x = self.glu(x)

        # Apply RMS normalization and activation quantization before output layer
        x = rms_norm(x, self.eps)
        x = activation_quant(x)

        # Output layer
        output = x

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
            output = self.output_layer(output) + self.output_layer.bias

        logits = output
        return logits, attention_weights




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
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    # Remove or adjust the truncation as needed
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_length):
            logits, attention_weights = model(generated)  # Unpack the tuple
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
        architecture = model_parameters.get('architecture', 'Cascade MatMul-Free LM')

        if architecture in ['CascadeTransformer', 'Cascade MatMul-Free LM']:
            model = CascadeTransformer(
                vocab_size=model_parameters['vocab_size'],
                embed_size=model_parameters['embed_size'],
                hidden_size=model_parameters['hidden_size'],
                num_nodes=model_parameters['num_nodes'],
                num_heads=model_parameters['num_heads']
            )
        else:
            messagebox.showerror("Error", f"Unsupported architecture: {architecture}")
            return

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
