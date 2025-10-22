#!/usr/bin/env python3
"""
P2P Backend Example

This example demonstrates how to use the P2P backend for tensor parallel inference
with ExLlamaV3. It shows automatic detection, manual specification, performance
comparison, and error handling scenarios.
"""

import sys
import os
import time
import torch
import argparse
from typing import Dict, List, Tuple, Optional

# Add parent directory to path to import exllamav3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3 import Generator, Job, model_init
from exllamav3.model.model_tp_cuda import check_p2p_connectivity
from exllamav3.model.model_tp_backend import get_available_backends, create_tp_backend


def check_system_p2p_support() -> bool:
    """
    Check if the current system supports P2P connectivity.
    
    Returns:
        bool: True if P2P is supported, False otherwise
    """
    print("\n=== Checking P2P System Support ===")
    
    # Get available CUDA devices
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. P2P requires CUDA.")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"✅ Found {num_gpus} CUDA device(s)")
    
    if num_gpus < 2:
        print("❌ P2P requires at least 2 GPUs. Found only {num_gpus}.")
        return False
    
    # Check P2P connectivity for all devices
    devices = list(range(num_gpus))
    print(f"🔄 Checking P2P connectivity for devices: {devices}")
    
    try:
        p2p_supported = check_p2p_connectivity(devices)
        if p2p_supported:
            print("✅ Full P2P connectivity detected between all GPUs")
            return True
        else:
            print("❌ P2P connectivity not available between all GPUs")
            return False
    except Exception as e:
        print(f"❌ Error checking P2P connectivity: {e}")
        return False


def demonstrate_automatic_backend_selection(model_dir: str, devices: List[int]):
    """
    Demonstrate automatic backend selection with P2P detection.
    
    Args:
        model_dir: Directory containing the model
        devices: List of device IDs to use
    """
    print("\n=== Automatic Backend Selection Demo ===")
    print(f"Using devices: {devices}")
    
    # Check available backends
    available_backends = get_available_backends()
    print(f"Available backends: {available_backends}")
    
    # Load model with automatic backend selection
    try:
        print("\n🔄 Loading model with automatic backend selection (tp_backend='auto')...")
        
        model_init_args = {
            'model_dir': model_dir,
            'devices': devices,
            'tp_backend': 'auto',  # This will auto-detect P2P if available
            'max_batch_size': 1,
            'max_seq_len': 2048,
            'max_input_len': 1024,
            'max_output_len': 1024
        }
        
        # Initialize model, cache, and tokenizer
        model, config, cache, tokenizer = model_init.init_from_dict(model_init_args)
        
        # Check which backend was selected
        if hasattr(model, 'tp_backend'):
            backend_type = type(model.tp_backend).__name__
            print(f"✅ Model loaded successfully with backend: {backend_type}")
        else:
            print("✅ Model loaded successfully")
            
        # Test inference
        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=tokenizer
        )
        
        # Simple inference test
        test_prompt = "Hello, I'm a language model and"
        input_ids = tokenizer.encode(test_prompt, add_bos=True)
        
        print(f"\n🔄 Testing inference with prompt: '{test_prompt}'")
        job = Job(
            input_ids=input_ids,
            max_new_tokens=50,
            temperature=0.7
        )
        generator.enqueue(job)
        
        # Get results
        results = list(generator.iterate())
        if results:
            generated_text = results[-1].get("text", "")
            print(f"✅ Generated: '{generated_text}'")
        
        # Clean up
        generator.close()
        model.unload()
        
    except Exception as e:
        print(f"❌ Error during automatic backend selection: {e}")


def demonstrate_manual_p2p_backend(model_dir: str, devices: List[int]):
    """
    Demonstrate manual P2P backend specification.
    
    Args:
        model_dir: Directory containing the model
        devices: List of device IDs to use
    """
    print("\n=== Manual P2P Backend Demo ===")
    print(f"Using devices: {devices}")
    
    try:
        print("\n🔄 Loading model with explicit P2P backend (tp_backend='p2p')...")
        
        model_init_args = {
            'model_dir': model_dir,
            'devices': devices,
            'tp_backend': 'p2p',  # Force P2P backend
            'max_batch_size': 1,
            'max_seq_len': 2048,
            'max_input_len': 1024,
            'max_output_len': 1024
        }
        
        # Initialize model, cache, and tokenizer
        model, config, cache, tokenizer = model_init.init_from_dict(model_init_args)
        print("✅ Model loaded successfully with P2P backend")
        
        # Test inference
        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=tokenizer
        )
        
        # Test with a longer prompt
        test_prompt = "In this demonstration, we'll show how the P2P backend improves performance for tensor parallel inference. The key advantages include:"
        input_ids = tokenizer.encode(test_prompt, add_bos=True)
        
        print(f"\n🔄 Testing inference with longer prompt")
        
        start_time = time.time()
        job = Job(
            input_ids=input_ids,
            max_new_tokens=100,
            temperature=0.7
        )
        generator.enqueue(job)
        
        # Get results
        results = list(generator.iterate())
        end_time = time.time()
        
        if results:
            generated_text = results[-1].get("text", "")
            tokens_generated = results[-1].get("new_tokens", 0)
            inference_time = end_time - start_time
            tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
            
            print(f"✅ Generated {tokens_generated} tokens in {inference_time:.2f}s")
            print(f"✅ Performance: {tokens_per_second:.2f} tokens/second")
            print(f"✅ Generated text: '{generated_text[:100]}...'")
        
        # Clean up
        generator.close()
        model.unload()
        
    except Exception as e:
        print(f"❌ Error with manual P2P backend: {e}")


def demonstrate_backend_comparison(model_dir: str, devices: List[int]):
    """
    Compare performance between P2P, NCCL, and Native backends.
    
    Args:
        model_dir: Directory containing the model
        devices: List of device IDs to use
    """
    print("\n=== Backend Performance Comparison ===")
    print(f"Using devices: {devices}")
    
    backends_to_test = ["p2p", "nccl", "native"]
    results = {}
    
    # Test prompt for comparison
    test_prompt = "The quick brown fox jumps over the lazy dog. " * 5  # Make it longer for better comparison
    
    for backend in backends_to_test:
        print(f"\n🔄 Testing backend: {backend}")
        
        try:
            model_init_args = {
                'model_dir': model_dir,
                'devices': devices,
                'tp_backend': backend,
                'max_batch_size': 1,
                'max_seq_len': 2048,
                'max_input_len': 1024,
                'max_output_len': 512
            }
            
            # Initialize model, cache, and tokenizer
            model, config, cache, tokenizer = model_init.init_from_dict(model_init_args)
            print(f"✅ Model loaded with {backend} backend")
            
            # Create generator
            generator = Generator(
                model=model,
                cache=cache,
                tokenizer=tokenizer
            )
            
            # Encode test prompt
            input_ids = tokenizer.encode(test_prompt, add_bos=True)
            
            # Measure inference time
            start_time = time.time()
            job = Job(
                input_ids=input_ids,
                max_new_tokens=200,
                temperature=0.7
            )
            generator.enqueue(job)
            
            # Get results
            results_list = list(generator.iterate())
            end_time = time.time()
            
            if results_list:
                result = results_list[-1]
                tokens_generated = result.get("new_tokens", 0)
                inference_time = end_time - start_time
                tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
                
                results[backend] = {
                    'tokens': tokens_generated,
                    'time': inference_time,
                    'tokens_per_second': tokens_per_second,
                    'success': True
                }
                
                print(f"✅ Generated {tokens_generated} tokens in {inference_time:.2f}s")
                print(f"✅ Performance: {tokens_per_second:.2f} tokens/second")
            else:
                results[backend] = {
                    'success': False,
                    'error': 'No results generated'
                }
                print(f"❌ No results generated")
            
            # Clean up
            generator.close()
            model.unload()
            
        except Exception as e:
            results[backend] = {
                'success': False,
                'error': str(e)
            }
            print(f"❌ Error with {backend} backend: {e}")
    
    # Print comparison summary
    print("\n=== Performance Comparison Summary ===")
    print(f"{'Backend':<10} {'Tokens/s':<12} {'Time (s)':<12} {'Status':<10}")
    print("-" * 45)
    
    for backend, data in results.items():
        if data['success']:
            print(f"{backend:<10} {data['tokens_per_second']:<12.2f} {data['time']:<12.2f} {'✅ Success':<10}")
        else:
            print(f"{backend:<10} {'N/A':<12} {'N/A':<12} {'❌ Failed':<10}")
    
    # Find the best performing backend
    successful_results = {k: v for k, v in results.items() if v['success']}
    if successful_results:
        best_backend = max(successful_results.items(), key=lambda x: x[1]['tokens_per_second'])
        print(f"\n🏆 Best performing backend: {best_backend[0]} ({best_backend[1]['tokens_per_second']:.2f} tokens/s)")


def demonstrate_error_handling(model_dir: str, devices: List[int]):
    """
    Demonstrate error handling and fallback scenarios.
    
    Args:
        model_dir: Directory containing the model
        devices: List of device IDs to use
    """
    print("\n=== Error Handling and Fallback Demo ===")
    
    # Test 1: Non-existent backend
    print("\n1. Testing non-existent backend...")
    try:
        model_init_args = {
            'model_dir': model_dir,
            'devices': devices,
            'tp_backend': 'nonexistent',
            'max_batch_size': 1,
        }
        
        model, config, cache, tokenizer = model_init.init_from_dict(model_init_args)
        print("❌ Should have failed with non-existent backend")
        model.unload()
        
    except Exception as e:
        print(f"✅ Correctly caught error for non-existent backend: {e}")
    
    # Test 2: P2P backend on single GPU (should fail gracefully)
    print("\n2. Testing P2P backend on single GPU...")
    try:
        single_device = [devices[0]] if devices else [0]
        model_init_args = {
            'model_dir': model_dir,
            'devices': single_device,
            'tp_backend': 'p2p',
            'max_batch_size': 1,
        }
        
        model, config, cache, tokenizer = model_init.init_from_dict(model_init_args)
        print("❌ Should have failed with P2P on single GPU")
        model.unload()
        
    except Exception as e:
        print(f"✅ Correctly caught error for P2P on single GPU: {e}")
    
    # Test 3: Auto backend with no P2P support (should fallback to NCCL)
    print("\n3. Testing auto backend fallback...")
    try:
        # Force P2P to be unavailable by using a single device
        single_device = [devices[0]] if devices else [0]
        model_init_args = {
            'model_dir': model_dir,
            'devices': single_device,
            'tp_backend': 'auto',  # Should fallback to NCCL
            'max_batch_size': 1,
        }
        
        model, config, cache, tokenizer = model_init.init_from_dict(model_init_args)
        print("✅ Auto backend correctly fell back to NCCL for single GPU")
        model.unload()
        
    except Exception as e:
        print(f"❌ Unexpected error with auto backend fallback: {e}")


def main():
    parser = argparse.ArgumentParser(description="P2P Backend Example")
    model_init.add_args(parser, cache=True)
    parser.add_argument("--demo", type=str, choices=["auto", "manual", "compare", "error", "all"], 
                       default="all", help="Which demo to run")
    parser.add_argument("--devices", type=str, default="0,1", 
                       help="Comma-separated list of device IDs to use (default: 0,1)")
    
    args = parser.parse_args()
    
    # Parse devices
    devices = [int(d.strip()) for d in args.devices.split(",")]
    
    # Get model directory from args
    model_dir = args.model_dir
    if not model_dir:
        print("❌ Model directory is required. Use -m or --model_dir to specify.")
        return
    
    print("🚀 P2P Backend Example")
    print("=" * 50)
    
    # Check system support first
    p2p_supported = check_system_p2p_support()
    
    # Run selected demos
    if args.demo in ["auto", "all"]:
        demonstrate_automatic_backend_selection(model_dir, devices)
    
    if args.demo in ["manual", "all"]:
        if p2p_supported:
            demonstrate_manual_p2p_backend(model_dir, devices)
        else:
            print("⚠️  Skipping manual P2P demo - P2P not supported")
    
    if args.demo in ["compare", "all"]:
        demonstrate_backend_comparison(model_dir, devices)
    
    if args.demo in ["error", "all"]:
        demonstrate_error_handling(model_dir, devices)
    
    print("\n✅ P2P Backend Example completed!")


if __name__ == "__main__":
    main()