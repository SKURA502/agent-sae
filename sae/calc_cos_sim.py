import argparse
import sys
import os

import torch
from tqdm import tqdm

# Ensure the project root is importable so torch.load can resolve
# checkpointed objects such as `sae.sae_model.SAEConfig`.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sae.sae_model import TopKSAE


def load_normalized_decoder(ckpt_path: str, device: str = "auto") -> torch.Tensor:
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
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

    return W


def update_global_topk(
    top_values: torch.Tensor,
    top_i: torch.Tensor,
    top_j: torch.Tensor,
    candidate_values: torch.Tensor,
    candidate_i: torch.Tensor,
    candidate_j: torch.Tensor,
    k: int,
):
    merged_values = torch.cat([top_values, candidate_values])
    merged_i = torch.cat([top_i, candidate_i])
    merged_j = torch.cat([top_j, candidate_j])

    keep = min(k, merged_values.numel())
    new_values, indices = torch.topk(merged_values, k=keep)
    return new_values, merged_i[indices], merged_j[indices]


def calc_average_cos_sim(
    ckpt_path: str,
    device: str = "auto",
    chunk_size: int = 2048,
    topk: int = 10,
):
    W = load_normalized_decoder(ckpt_path, device=device)
    dict_size = W.shape[1]
    num_pairs = dict_size * (dict_size - 1) // 2
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    topk = min(topk, num_pairs)

    # We want to compute the average off-diagonal cosine similarity
    # Calculate W^T @ W
    # To save memory and prevent OOM on standard devices, we compute it in chunks
    total_sim = 0.0
    total_abs_sim = 0.0
    max_sim = -1.0
    count = 0
    top_values = torch.empty(0, dtype=torch.float32)
    top_i = torch.empty(0, dtype=torch.long)
    top_j = torch.empty(0, dtype=torch.long)
    
    print("Computing pairwise cosine similarities...")
    W_T = W.T  # shape: (dict_size, input_dim)
    
    for i in tqdm(range(0, dict_size, chunk_size)):
        end_i = min(i + chunk_size, dict_size)
        W_chunk = W_T[i:end_i]  # (chunk, input_dim)
        
        # sim_chunk shape: (chunk, dict_size)
        sim_chunk = torch.matmul(W_chunk, W)
        row_ids = torch.arange(i, end_i, device=sim_chunk.device)
        
        # Keep only the strict upper triangle for top-k so each pair appears once.
        upper_mask = torch.arange(dict_size, device=sim_chunk.device).unsqueeze(0) > row_ids.unsqueeze(1)
        upper_values = sim_chunk[upper_mask]
        if upper_values.numel() > 0:
            upper_indices = upper_mask.nonzero(as_tuple=False)
            candidate_k = min(topk, upper_values.numel())
            candidate_values, candidate_pos = torch.topk(upper_values, k=candidate_k)
            candidate_rows = upper_indices[candidate_pos, 0]
            candidate_cols = upper_indices[candidate_pos, 1]
            candidate_i = row_ids[candidate_rows].to(device="cpu")
            candidate_j = candidate_cols.to(device="cpu")
            top_values, top_i, top_j = update_global_topk(
                top_values,
                top_i,
                top_j,
                candidate_values.to(device="cpu"),
                candidate_i,
                candidate_j,
                topk,
            )

        # Remove self-similarity (diagonal elements)
        for j in range(end_i - i):
            global_idx = i + j
            sim_chunk[j, global_idx] = 0.0  # Set diagonal to 0 so it doesn't affect sum
            
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
    print(f"Top-{topk} Most Similar Feature Pairs:")
    for rank, (value, feat_i, feat_j) in enumerate(zip(top_values.tolist(), top_i.tolist(), top_j.tolist()), start=1):
        print(f"  {rank:>2}. feature {feat_i} <-> feature {feat_j}: {value:.6f}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average cosine similarity of SAE decoder vectors.")
    parser.add_argument("--ckpt_path", type=str, help="Path to the SAE checkpoint file")
    parser.add_argument("--device", type=str, default="auto", help="Device to load the model on (cuda/cpu/mps/auto)")
    parser.add_argument("--chunk_size", type=int, default=2048, help="Chunk size for matrix multiplication to save memory")
    parser.add_argument("--topk", type=int, default=5, help="Number of most similar feature pairs to report")
    args = parser.parse_args()

    calc_average_cos_sim(
        args.ckpt_path,
        device=args.device,
        chunk_size=args.chunk_size,
        topk=args.topk,
    )
