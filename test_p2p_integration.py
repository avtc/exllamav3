#!/usr/bin/env python3
"""
Integration test for P2P backend with tensor parallelism in ExLlamaV3.

This test validates that the P2P backend works correctly with the existing
tensor parallelism infrastructure and provides performance benefits.
"""

import os
import sys
import torch
import time
import argparse
from typing import Dict, List, Tuple

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exllamav3 import Model, Config, Cache, Tokenizer
from exllamav3.model.model_tp_backend import TPBackendP2P, TPBackendNative
from exllamav3.model.model_tp_p2p import P2PTopology


def test_p2p_topology_detection():
    """Test P2P topology detection and analysis."""
    print("Testing P2P topology detection...")
    
    # Get available CUDA devices
    if torch.cuda.device_count() < 2:
        print("  Skipping P2P topology test - need at least 2 GPUs")
        return True
    
    active_devices = list(range(torch.cuda.device_count()))
    topology = P2PTopology(active_devices)
    
    # Get topology summary
    summary = topology.get_topology_summary()
    print(f"  Topology summary: {summary}")
    
    # Test connectivity checks
    for i in active_devices:
        for j in active_devices:
            if i != j:
                can_access = topology.can_access_peer(i, j)
                print(f"  Device {i} -> Device {j}: {'P2P available' if can_access else 'P2P not available'}")
    
    # Test algorithm selection
    for size in [1024, 1024*1024, 10*1024*1024]:
        algorithm = topology.select_reduce_algorithm(size)
        print(f"  Tensor size {size//1024}KB: Selected algorithm {algorithm}")
    
    return True


def test_p2p_backend_initialization():
    """Test P2P backend initialization and basic operations."""
    print("\nTesting P2P backend initialization...")
    
    if torch.cuda.device_count() < 2:
        print("  Skipping P2P backend test - need at least 2 GPUs")
        return True
    
    active_devices = list(range(torch.cuda.device_count()))
    output_device = active_devices[0]
    
    # Test P2P backend initialization
    try:
        backend = TPBackendP2P(
            device=output_device,
            active_devices=active_devices,
            output_device=output_device,
            init_method="tcp://127.0.0.1:12345",
            master=True,
            uuid="test-uuid"
        )
        print(f"  P2P backend initialized successfully")
        print(f"  P2P available: {backend.use_p2p}")
        
        # Test memory pool stats
        if backend.use_p2p:
            stats = backend.get_memory_pool_stats()
            print(f"  Memory pool stats: {stats}")
        
        # Test direct memory copy
        if backend.use_p2p and len(active_devices) >= 2:
            test_tensor = torch.randn(100, 100, device=output_device)
            try:
                copied_tensor = backend.copy_tensor_direct(output_device, active_devices[1], test_tensor)
                print(f"  Direct memory copy successful")
            except Exception as e:
                print(f"  Direct memory copy failed: {e}")
        
        backend.close()
        return True
    except Exception as e:
        print(f"  P2P backend initialization failed: {e}")
        return False


def test_p2p_with_model_loading(model_dir: str, tp_backend: str = "p2p"):
    """Test loading a model with P2P backend."""
    print(f"\nTesting model loading with {tp_backend} backend...")
    
    if not os.path.exists(model_dir):
        print(f"  Model directory not found: {model_dir}")
        return True
    
    try:
        # Load config
        config = Config.from_directory(model_dir)
        
        # Create model
        model = Model.from_config(config)
        
        # Load with tensor parallelism
        model.load(
            tensor_p=True,
            tp_backend=tp_backend,
            progressbar=True,
            verbose=True
        )
        
        print(f"  Model loaded successfully with {tp_backend} backend")
        
        # Test forward pass
        tokenizer = Tokenizer.from_config(config)
        cache = Cache(model, max_num_tokens=2048)
        
        # Simple test prompt
        prompt = "The quick brown fox"
        input_ids = tokenizer.encode(prompt)
        
        # Test prefill
        start_time = time.time()
        output = model.prefill(input_ids, {"cache": cache})
        prefill_time = time.time() - start_time
        
        # Test forward pass
        start_time = time.time()
        output = model.forward(input_ids, {"cache": cache})
        forward_time = time.time() - start_time
        
        print(f"  Prefill time: {prefill_time:.3f}s")
        print(f"  Forward time: {forward_time:.3f}s")
        
        # Unload model
        model.unload()
        return True
    except Exception as e:
        print(f"  Model loading failed: {e}")
        return False


def test_p2p_vs_native_performance(model_dir: str):
    """Compare performance between P2P and native backends."""
    print("\nTesting P2P vs Native backend performance...")
    
    if not os.path.exists(model_dir):
        print(f"  Model directory not found: {model_dir}")
        return True
    
    if torch.cuda.device_count() < 2:
        print("  Skipping performance test - need at least 2 GPUs")
        return True
    
    results = {}
    
    for backend in ["native", "p2p"]:
        print(f"  Testing {backend} backend...")
        
        try:
            # Load config
            config = Config.from_directory(model_dir)
            
            # Create model
            model = Model.from_config(config)
            
            # Load with tensor parallelism
            model.load(
                tensor_p=True,
                tp_backend=backend,
                progressbar=False,
                verbose=False
            )
            
            # Test forward pass
            tokenizer = Tokenizer.from_config(config)
            cache = Cache(model, max_num_tokens=2048)
            
            # Simple test prompt
            prompt = "The quick brown fox jumps over the lazy dog"
            input_ids = tokenizer.encode(prompt)
            
            # Warm up
            for _ in range(3):
                model.forward(input_ids, {"cache": cache})
            
            # Benchmark
            num_iterations = 10
            start_time = time.time()
            
            for _ in range(num_iterations):
                output = model.forward(input_ids, {"cache": cache})
            
            total_time = time.time() - start_time
            avg_time = total_time / num_iterations
            
            results[backend] = {
                "total_time": total_time,
                "avg_time": avg_time,
                "tokens_per_second": len(input_ids) / avg_time
            }
            
            print(f"    Average time: {avg_time:.3f}s")
            print(f"    Tokens/sec: {len(input_ids) / avg_time:.1f}")
            
            # Unload model
            model.unload()
        except Exception as e:
            print(f"    {backend} backend test failed: {e}")
            results[backend] = None
    
    # Compare results
    if results.get("native") and results.get("p2p"):
        native_time = results["native"]["avg_time"]
        p2p_time = results["p2p"]["avg_time"]
        speedup = native_time / p2p_time
        
        print(f"  Performance comparison:")
        print(f"    Native: {native_time:.3f}s")
        print(f"    P2P: {p2p_time:.3f}s")
        print(f"    Speedup: {speedup:.2f}x")
    
    return True


def test_p2p_fallback():
    """Test P2P backend fallback to native when P2P is not available."""
    print("\nTesting P2P backend fallback...")
    
    # Create a mock scenario where P2P is not available
    active_devices = [0]  # Single device
    output_device = 0
    
    try:
        backend = TPBackendP2P(
            device=output_device,
            active_devices=active_devices,
            output_device=output_device,
            init_method="tcp://127.0.0.1:12345",
            master=True,
            uuid="test-uuid"
        )
        
        # Should fallback to native backend
        print(f"  P2P available: {backend.use_p2p}")
        
        # Test operations - should work through fallback
        test_tensor = torch.randn(100, 100, device=output_device)
        backend.broadcast(test_tensor, output_device)
        backend.all_reduce(test_tensor)
        
        print("  Fallback operations successful")
        backend.close()
        return True
    except Exception as e:
        print(f"  Fallback test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test P2P backend integration")
    parser.add_argument("--model_dir", type=str, help="Path to model directory")
    parser.add_argument("--test_topology", action="store_true", help="Test P2P topology detection")
    parser.add_argument("--test_backend", action="store_true", help="Test P2P backend initialization")
    parser.add_argument("--test_model", action="store_true", help="Test model loading with P2P")
    parser.add_argument("--test_performance", action="store_true", help="Test P2P vs Native performance")
    parser.add_argument("--test_fallback", action="store_true", help="Test P2P fallback")
    
    args = parser.parse_args()
    
    # Run all tests if no specific test is requested
    run_all = not any([
        args.test_topology,
        args.test_backend,
        args.test_model,
        args.test_performance,
        args.test_fallback
    ])
    
    print("P2P Backend Integration Test")
    print("=" * 50)
    
    success = True
    
    # Test P2P topology detection
    if run_all or args.test_topology:
        success &= test_p2p_topology_detection()
    
    # Test P2P backend initialization
    if run_all or args.test_backend:
        success &= test_p2p_backend_initialization()
    
    # Test model loading with P2P
    if run_all or args.test_model:
        if args.model_dir:
            success &= test_p2p_with_model_loading(args.model_dir)
        else:
            print("\nSkipping model test - no model directory provided")
            print("  Use --model_dir to specify a model directory")
    
    # Test performance comparison
    if run_all or args.test_performance:
        if args.model_dir:
            success &= test_p2p_vs_native_performance(args.model_dir)
        else:
            print("\nSkipping performance test - no model directory provided")
            print("  Use --model_dir to specify a model directory")
    
    # Test P2P fallback
    if run_all or args.test_fallback:
        success &= test_p2p_fallback()
    
    print("\n" + "=" * 50)
    if success:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())