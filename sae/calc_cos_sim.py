import argparse
import sys
import os
import torch
from tqdm import tqdm

# Ensure the sae_model can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sae_model import TopKSAE

def calc_average_cos_sim(ckpt_path: str, device: str = "auto", chunk_size: int = 2048):
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
    print(f"Loading SAE from {ckpt_path} to {device}...")
    model = TopKSAE.load(ckpt_path, device=device)
    
    # Decoder weight shape is (input_dim, dict_size)
    # The vectors are columns (dim=0 is input_dim, dim=1 is dict_size)
    # They are already normalized in dim=0 during _normalize_decoder(), 
    # but we can normalize again just to be safe.
    W = model.decoder.weight.data.to(torch.float32)
    W = torch.nn.functional.normalize(W, dim=0)
    
    dict_size = W.shape[1]
    input_dim = W.shape[0]
    print(f"Decoder weight shape: {W.shape}. Dictionary size: {dict_size}, Input dim: {input_dim}")
    
    # We want to compute the average off-diagonal cosine similarity
    # Calculate W^T @ W
    # To save memory and prevent OOM on standard devices, we compute it in chunks
    total_sim = 0.0
    total_abs_sim = 0.0
    max_sim = -1.0
    count = 0
    
    print("Computing pairwise cosine similarities...")
    W_T = W.T # shape: (dict_size, input_dim)
    
    for i in tqdm(range(0, dict_size, chunk_size)):
        end_i = min(i + chunk_size, dict_size)
        W_chunk = W_T[i:end_i] # (chunk, input_dim)
        
        # sim_chunk shape: (chunk, dict_size)
        sim_chunk = torch.matmul(W_chunk, W)
        
        # Remove self-similarity (diagonal elements)
        for j in range(end_i - i):
            global_idx = i + j
            sim_chunk[j, global_idx] = 0.0 # Set diagonal to 0 so it doesn't affect sum
            
        total_sim += sim_chunk.sum().item()
        total_abs_sim += sim_chunk.abs().sum().item()
        max_sim = max(max_sim, sim_chunk.max().item())
        
        # The number of off-diagonal elements processed in this chunk
        count += (end_i - i) * dict_size - (end_i - i)

    avg_sim = total_sim / count
    avg_abs_sim = total_abs_sim / count
    
    print("-" * 50)
    print("Results:")
    print(f"Average Cosine Similarity:          {avg_sim:.6f}")
    print(f"Average Absolute Cosine Similarity: {avg_abs_sim:.6f}")
    print(f"Maximum Off-diagonal Similarity:    {max_sim:.6f}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average cosine similarity of SAE decoder vectors.")
    parser.add_argument("ckpt_path", type=str, help="Path to the SAE checkpoint file")
    parser.add_argument("--device", type=str, default="auto", help="Device to load the model on (cuda/cpu/mps/auto)")
    parser.add_argument("--chunk_size", type=int, default=2048, help="Chunk size for matrix multiplication to save memory")
    args = parser.parse_args()
    
    calc_average_cos_sim(args.ckpt_path, device=args.device, chunk_size=args.chunk_size)
