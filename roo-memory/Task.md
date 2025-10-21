# Task: Identify CPU/RAM-bound operations that could affect token generation speed significantly, especially for tensor parallelism, and find a way to use p2p between GPUs (For example I have 8 GPUs with fast p2p access between each other).

## Objectives:
1. Identify CPU/RAM-bound operations that affect token generation speed
2. Focus on tensor parallelism operations
3. Implement P2P communication between GPUs for faster data transfer
4. Optimize performance for multi-GPU setups (8 GPUs with fast P2P access)

## Expected Outcomes:
- Improved token generation speed through optimized CPU/RAM operations
- Efficient GPU-to-GPU communication using P2P
- Better utilization of multi-GPU hardware capabilities