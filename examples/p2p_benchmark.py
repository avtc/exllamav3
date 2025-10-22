#!/usr/bin/env python3
"""
P2P Performance Benchmark

This example provides comprehensive benchmarking capabilities for the P2P backend,
including performance comparisons between backends, scalability testing with
different tensor sizes, and throughput/latency analysis.
"""

import sys
import os
import time
import torch
import argparse
import json
import gc
import psutil
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Add parent directory to path to import exllamav3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3 import Generator, Job, model_init
from exllamav3.model.model_tp_cuda import check_p2p_connectivity
from exllamav3.model.model_tp_backend import get_available_backends, create_tp_backend


@dataclass
class BenchmarkResult:
    """Data class to store benchmark results."""
    
    backend: str
    devices: List[int]
    tensor_size: int
    batch_size: int
    tokens_generated: int
    inference_time: float
    tokens_per_second: float
    memory_usage: Dict[str, float]
    gpu_utilization: Dict[int, float]
    success: bool
    error: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class P2PBenchmark:
    """Comprehensive benchmarking suite for P2P backend."""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.results = []
        self.device_stats = {}
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        return {
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "gpu_memory": [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())],
            "cpu_cores": psutil.cpu_count(),
            "cpu_memory": psutil.virtual_memory().total,
            "python_version": sys.version
        }
    
    def warm_up_cuda(self, devices: List[int], warmup_steps: int = 10):
        """Warm up CUDA devices to ensure consistent measurements."""
        print(f"\n🔥 Warming up CUDA devices: {devices}")
        
        # Create dummy tensors for warmup
        for device in devices:
            torch.cuda.set_device(device)
            for _ in range(warmup_steps):
                # Create and process dummy tensors
                dummy_tensor = torch.randn(1024, 1024, device=device, dtype=torch.float16)
                result = dummy_tensor @ dummy_tensor
                del dummy_tensor, result
                torch.cuda.synchronize()
        
        print("✅ CUDA warmup completed")
    
    def measure_gpu_memory(self, device: int) -> Dict[str, float]:
        """Measure GPU memory usage."""
        torch.cuda.set_device(device)
        return {
            "allocated": torch.cuda.memory_allocated(device) / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved(device) / 1024**3,     # GB
            "max_allocated": torch.cuda.max_memory_allocated(device) / 1024**3,  # GB
        }
    
    def measure_gpu_utilization(self, device: int) -> float:
        """Measure GPU utilization (simulated)."""
        # Note: In practice, you might use nvidia-ml-py or similar tools
        # For this example, we'll simulate it
        torch.cuda.set_device(device)
        torch.cuda.synchronize()
        
        # Create a small workload to get utilization
        start_time = time.time()
        dummy_tensor = torch.randn(512, 512, device=device, dtype=torch.float16)
        for _ in range(100):
            result = dummy_tensor @ dummy_tensor
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Simple utilization estimate based on time
        elapsed = end_time - start_time
        utilization = min(100.0, elapsed * 10)  # Very rough estimate
        
        del dummy_tensor, result
        return utilization
    
    def generate_test_prompts(self, count: int, min_length: int = 50, max_length: int = 200) -> List[str]:
        """Generate test prompts of varying lengths."""
        base_prompts = [
            "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
            "In the realm of artificial intelligence, large language models have revolutionized how we interact with technology.",
            "Quantum computing represents a paradigm shift in computational capabilities, offering exponential speedups for certain problems.",
            "Climate change is one of the most pressing challenges facing humanity in the 21st century.",
            "The exploration of space has always captivated human imagination and driven technological innovation.",
            "Machine learning algorithms are becoming increasingly sophisticated, enabling breakthroughs in various fields.",
            "Renewable energy sources are essential for sustainable development and environmental preservation.",
            "The human brain remains the most complex and powerful computing device known to science.",
            "Blockchain technology offers new possibilities for secure and transparent transactions.",
            "The future of transportation lies in autonomous vehicles and smart infrastructure systems."
        ]
        
        prompts = []
        for i in range(count):
            # Vary prompt length
            if i % 3 == 0:
                # Short prompt
                prompt = base_prompts[i % len(base_prompts)][:min_length]
            elif i % 3 == 1:
                # Medium prompt
                mid_point = (min_length + max_length) // 2
                prompt = base_prompts[i % len(base_prompts)][:mid_point]
            else:
                # Long prompt
                prompt = base_prompts[i % len(base_prompts)][:max_length]
            
            prompts.append(prompt)
        
        return prompts
    
    def benchmark_throughput(self, backend: str, devices: List[int], 
                           tensor_sizes: List[int], batch_sizes: List[int],
                           warmup_runs: int = 3, test_runs: int = 5) -> List[BenchmarkResult]:
        """Benchmark throughput for different tensor sizes and batch sizes."""
        print(f"\n🚀 Throughput Benchmark: {backend}")
        print(f"Devices: {devices}")
        print(f"Tensor sizes: {tensor_sizes}")
        print(f"Batch sizes: {batch_sizes}")
        
        results = []
        
        try:
            # Load model with specified backend
            model_init_args = {
                'model_dir': self.model_dir,
                'devices': devices,
                'tp_backend': backend,
                'max_batch_size': max(batch_sizes),
                'max_seq_len': max(tensor_sizes) + 512,
                'max_input_len': max(tensor_sizes),
                'max_output_len': 512,
                'cache': True
            }
            
            model, config, cache, tokenizer = model_init.init_from_dict(model_init_args)
            print(f"✅ Model loaded with {backend} backend")
            
            generator = Generator(
                model=model,
                cache=cache,
                tokenizer=tokenizer
            )
            
            # Warm up
            self.warm_up_cuda(devices)
            
            # Generate test prompts
            test_prompts = self.generate_test_prompts(len(batch_sizes) * len(tensor_sizes))
            
            # Test different tensor sizes and batch sizes
            for tensor_size in tensor_sizes:
                for batch_size in batch_sizes:
                    print(f"\n📊 Testing: tensor_size={tensor_size}, batch_size={batch_size}")
                    
                    batch_results = []
                    
                    for run in range(warmup_runs + test_runs):
                        try:
                            # Select appropriate prompt
                            prompt_idx = (run % len(test_prompts))
                            test_prompt = test_prompts[prompt_idx]
                            
                            # Encode prompt
                            input_ids = tokenizer.encode(test_prompt, add_bos=True)
                            
                            # Pad/truncate to tensor size
                            if len(input_ids) > tensor_size:
                                input_ids = input_ids[:tensor_size]
                            else:
                                # Pad with zeros
                                padding = tensor_size - len(input_ids)
                                input_ids = torch.cat([input_ids, torch.zeros(padding, dtype=torch.long)])
                            
                            # Create batch
                            batch_input_ids = input_ids.unsqueeze(0).repeat(batch_size, 1)
                            
                            # Measure memory before
                            memory_before = {}
                            for device in devices:
                                memory_before[device] = self.measure_gpu_memory(device)
                            
                            # Measure GPU utilization before
                            gpu_util_before = {}
                            for device in devices:
                                gpu_util_before[device] = self.measure_gpu_utilization(device)
                            
                            # Run inference
                            start_time = time.perf_counter()
                            
                            job = Job(
                                input_ids=batch_input_ids,
                                max_new_tokens=min(100, tensor_size // 4),
                                temperature=0.7,
                                stop_conditions=[tokenizer.eos_token_id]
                            )
                            generator.enqueue(job)
                            
                            # Get results
                            results_list = list(generator.iterate())
                            
                            end_time = time.perf_counter()
                            
                            # Measure memory after
                            memory_after = {}
                            for device in devices:
                                memory_after[device] = self.measure_gpu_memory(device)
                            
                            # Measure GPU utilization after
                            gpu_util_after = {}
                            for device in devices:
                                gpu_util_after[device] = self.measure_gpu_utilization(device)
                            
                            # Calculate metrics
                            inference_time = end_time - start_time
                            tokens_generated = 0
                            
                            if results_list:
                                for result in results_list:
                                    tokens_generated += result.get("new_tokens", 0)
                            
                            throughput = tokens_generated / inference_time if inference_time > 0 else 0
                            
                            # Calculate memory usage
                            memory_usage = {}
                            for device in devices:
                                memory_usage[device] = {
                                    "allocated": memory_after[device]["allocated"] - memory_before[device]["allocated"],
                                    "peak": memory_after[device]["max_allocated"]
                                }
                            
                            # Calculate average GPU utilization
                            avg_gpu_util = {}
                            for device in devices:
                                avg_gpu_util[device] = (gpu_util_before[device] + gpu_util_after[device]) / 2
                            
                            if run >= warmup_runs:  # Skip warmup runs
                                result = BenchmarkResult(
                                    backend=backend,
                                    devices=devices,
                                    tensor_size=tensor_size,
                                    batch_size=batch_size,
                                    tokens_generated=tokens_generated,
                                    inference_time=inference_time,
                                    tokens_per_second=throughput,
                                    memory_usage=memory_usage,
                                    gpu_utilization=avg_gpu_util,
                                    success=True
                                )
                                batch_results.append(result)
                                print(f"  Run {run - warmup_runs + 1}: {throughput:.2f} tokens/s")
                        
                        except Exception as e:
                            print(f"  Run {run} failed: {e}")
                            if run >= warmup_runs:
                                result = BenchmarkResult(
                                    backend=backend,
                                    devices=devices,
                                    tensor_size=tensor_size,
                                    batch_size=batch_size,
                                    tokens_generated=0,
                                    inference_time=0,
                                    tokens_per_second=0,
                                    memory_usage={},
                                    gpu_utilization={},
                                    success=False,
                                    error=str(e)
                                )
                                batch_results.append(result)
                    
                    # Calculate average for this configuration
                    if batch_results:
                        avg_throughput = np.mean([r.tokens_per_second for r in batch_results if r.success])
                        std_throughput = np.std([r.tokens_per_second for r in batch_results if r.success])
                        
                        print(f"  Average: {avg_throughput:.2f} ± {std_throughput:.2f} tokens/s")
                        results.extend(batch_results)
            
            # Clean up
            generator.close()
            model.unload()
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            # Create error result
            result = BenchmarkResult(
                backend=backend,
                devices=devices,
                tensor_size=0,
                batch_size=0,
                tokens_generated=0,
                inference_time=0,
                tokens_per_second=0,
                memory_usage={},
                gpu_utilization={},
                success=False,
                error=str(e)
            )
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def benchmark_scalability(self, backends: List[str], device_counts: List[int],
                            tensor_size: int = 1024, warmup_runs: int = 2, test_runs: int = 3) -> List[BenchmarkResult]:
        """Benchmark scalability across different numbers of devices."""
        print(f"\n📈 Scalability Benchmark")
        print(f"Backends: {backends}")
        print(f"Device counts: {device_counts}")
        print(f"Tensor size: {tensor_size}")
        
        results = []
        
        for backend in backends:
            for device_count in device_counts:
                devices = list(range(device_count))
                print(f"\n🔬 Testing: {backend} with {device_count} devices")
                
                try:
                    # Run throughput benchmark for single configuration
                    backend_results = self.benchmark_throughput(
                        backend=backend,
                        devices=devices,
                        tensor_sizes=[tensor_size],
                        batch_sizes=[1],
                        warmup_runs=warmup_runs,
                        test_runs=test_runs
                    )
                    
                    # Calculate scalability metrics
                    if backend_results:
                        successful_results = [r for r in backend_results if r.success]
                        if successful_results:
                            avg_throughput = np.mean([r.tokens_per_second for r in successful_results])
                            
                            # Calculate speedup relative to single device
                            if device_count > 1:
                                single_device_results = [r for r in self.results 
                                                       if r.backend == backend and len(r.devices) == 1]
                                if single_device_results:
                                    single_device_throughput = np.mean([r.tokens_per_second for r in single_device_results if r.success])
                                    speedup = avg_throughput / single_device_throughput if single_device_throughput > 0 else 0
                                    efficiency = speedup / device_count * 100
                                else:
                                    speedup = efficiency = 0
                            else:
                                speedup = efficiency = 100
                            
                            print(f"  Throughput: {avg_throughput:.2f} tokens/s")
                            print(f"  Speedup: {speedup:.2f}x")
                            print(f"  Efficiency: {efficiency:.1f}%")
                    
                    results.extend(backend_results)
                
                except Exception as e:
                    print(f"❌ Scalability test failed: {e}")
        
        return results
    
    def benchmark_latency(self, backend: str, devices: List[int], 
                         test_sizes: List[int], num_tests: int = 10) -> List[BenchmarkResult]:
        """Benchmark latency for different tensor sizes."""
        print(f"\n⚡ Latency Benchmark: {backend}")
        print(f"Devices: {devices}")
        print(f"Test sizes: {test_sizes}")
        print(f"Number of tests: {num_tests}")
        
        results = []
        
        try:
            # Load model
            model_init_args = {
                'model_dir': self.model_dir,
                'devices': devices,
                'tp_backend': backend,
                'max_batch_size': 1,
                'max_seq_len': max(test_sizes) + 512,
                'max_input_len': max(test_sizes),
                'max_output_len': 50,
                'cache': True
            }
            
            model, config, cache, tokenizer = model_init.init_from_dict(model_init_args)
            print(f"✅ Model loaded with {backend} backend")
            
            generator = Generator(
                model=model,
                cache=cache,
                tokenizer=tokenizer
            )
            
            # Warm up
            self.warm_up_cuda(devices)
            
            for tensor_size in test_sizes:
                print(f"\n📏 Testing latency for tensor size: {tensor_size}")
                
                latencies = []
                
                for test in range(num_tests):
                    try:
                        # Create test prompt
                        test_prompt = "Hello world. " * (tensor_size // 12)
                        input_ids = tokenizer.encode(test_prompt, add_bos=True)
                        
                        # Pad/truncate to exact size
                        if len(input_ids) > tensor_size:
                            input_ids = input_ids[:tensor_size]
                        else:
                            padding = tensor_size - len(input_ids)
                            input_ids = torch.cat([input_ids, torch.zeros(padding, dtype=torch.long)])
                        
                        # Measure latency
                        start_time = time.perf_counter()
                        
                        job = Job(
                            input_ids=input_ids,
                            max_new_tokens=10,  # Small number for latency test
                            temperature=0.7
                        )
                        generator.enqueue(job)
                        
                        results_list = list(generator.iterate())
                        
                        end_time = time.perf_counter()
                        
                        latency = end_time - start_time
                        latencies.append(latency)
                        
                        print(f"  Test {test + 1}: {latency * 1000:.2f} ms")
                    
                    except Exception as e:
                        print(f"  Test {test + 1} failed: {e}")
                
                # Calculate latency statistics
                if latencies:
                    avg_latency = np.mean(latencies)
                    std_latency = np.std(latencies)
                    min_latency = np.min(latencies)
                    max_latency = np.max(latencies)
                    
                    print(f"  Average latency: {avg_latency * 1000:.2f} ± {std_latency * 1000:.2f} ms")
                    print(f"  Min latency: {min_latency * 1000:.2f} ms")
                    print(f"  Max latency: {max_latency * 1000:.2f} ms")
                    
                    # Create result
                    result = BenchmarkResult(
                        backend=backend,
                        devices=devices,
                        tensor_size=tensor_size,
                        batch_size=1,
                        tokens_generated=10,  # Fixed for latency test
                        inference_time=avg_latency,
                        tokens_per_second=10 / avg_latency if avg_latency > 0 else 0,
                        memory_usage={},
                        gpu_utilization={},
                        success=True
                    )
                    results.append(result)
            
            # Clean up
            generator.close()
            model.unload()
            
        except Exception as e:
            print(f"❌ Latency benchmark failed: {e}")
            result = BenchmarkResult(
                backend=backend,
                devices=devices,
                tensor_size=0,
                batch_size=1,
                tokens_generated=0,
                inference_time=0,
                tokens_per_second=0,
                memory_usage={},
                gpu_utilization={},
                success=False,
                error=str(e)
            )
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def generate_report(self, output_file: str = "p2p_benchmark_report.json"):
        """Generate comprehensive benchmark report."""
        print(f"\n📋 Generating benchmark report...")
        
        # Get system info
        system_info = self.get_system_info()
        
        # Process results
        report = {
            "system_info": system_info,
            "benchmark_timestamp": time.time(),
            "total_benchmarks": len(self.results),
            "successful_benchmarks": len([r for r in self.results if r.success]),
            "failed_benchmarks": len([r for r in self.results if not r.success]),
            "results": []
        }
        
        # Convert results to dictionaries
        for result in self.results:
            result_dict = asdict(result)
            # Convert numpy types to native Python types
            for key, value in result_dict.items():
                if isinstance(value, np.floating):
                    result_dict[key] = float(value)
                elif isinstance(value, np.integer):
                    result_dict[key] = int(value)
            report["results"].append(result_dict)
        
        # Calculate summary statistics
        successful_results = [r for r in self.results if r.success]
        if successful_results:
            report["summary"] = {
                "average_throughput": np.mean([r.tokens_per_second for r in successful_results]),
                "max_throughput": np.max([r.tokens_per_second for r in successful_results]),
                "min_throughput": np.min([r.tokens_per_second for r in successful_results]),
                "throughput_std": np.std([r.tokens_per_second for r in successful_results])
            }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Report saved to: {output_file}")
        return report
    
    def print_summary(self):
        """Print a human-readable summary of benchmark results."""
        print("\n" + "=" * 60)
        print("🏆 BENCHMARK SUMMARY")
        print("=" * 60)
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        print(f"Total benchmarks: {len(self.results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        
        if successful_results:
            print(f"\nPerformance Metrics:")
            print(f"  Average throughput: {np.mean([r.tokens_per_second for r in successful_results]):.2f} tokens/s")
            print(f"  Maximum throughput: {np.max([r.tokens_per_second for r in successful_results]):.2f} tokens/s")
            print(f"  Minimum throughput: {np.min([r.tokens_per_second for r in successful_results]):.2f} tokens/s")
            
            # Group by backend
            backends = set(r.backend for r in successful_results)
            print(f"\nBackend Performance:")
            for backend in backends:
                backend_results = [r for r in successful_results if r.backend == backend]
                avg_throughput = np.mean([r.tokens_per_second for r in backend_results])
                print(f"  {backend}: {avg_throughput:.2f} tokens/s")
            
            # Best performing configuration
            best_result = max(successful_results, key=lambda x: x.tokens_per_second)
            print(f"\n🥇 Best Performance:")
            print(f"  Backend: {best_result.backend}")
            print(f"  Devices: {best_result.devices}")
            print(f"  Tensor size: {best_result.tensor_size}")
            print(f"  Batch size: {best_result.batch_size}")
            print(f"  Throughput: {best_result.tokens_per_second:.2f} tokens/s")
        
        if failed_results:
            print(f"\n❌ Failed Benchmarks:")
            for result in failed_results:
                print(f"  {result.backend}: {result.error}")


def main():
    parser = argparse.ArgumentParser(description="P2P Performance Benchmark")
    model_init.add_args(parser, cache=True)
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing the model")
    parser.add_argument("--devices", type=str, default="0,1", 
                       help="Comma-separated list of device IDs to use")
    parser.add_argument("--benchmark-type", type=str, choices=["throughput", "scalability", "latency", "all"],
                       default="all", help="Type of benchmark to run")
    parser.add_argument("--output", type=str, default="p2p_benchmark_report.json",
                       help="Output file for benchmark report")
    parser.add_argument("--tensor-sizes", type=str, default="256,512,1024,2048",
                       help="Comma-separated list of tensor sizes to test")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                       help="Comma-separated list of batch sizes to test")
    parser.add_argument("--device-counts", type=str, default="1,2,4",
                       help="Comma-separated list of device counts for scalability test")
    parser.add_argument("--warmup-runs", type=int, default=3,
                       help="Number of warmup runs for each test")
    parser.add_argument("--test-runs", type=int, default=5,
                       help="Number of test runs for each configuration")
    
    args = parser.parse_args()
    
    # Parse arguments
    devices = [int(d.strip()) for d in args.devices.split(",")]
    tensor_sizes = [int(s.strip()) for s in args.tensor_sizes.split(",")]
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
    device_counts = [int(c.strip()) for c in args.device_counts.split(",")]
    
    print("🚀 P2P Performance Benchmark")
    print("=" * 50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. P2P backend requires CUDA.")
        return
    
    # Check P2P connectivity
    if not check_p2p_connectivity(devices):
        print("⚠️  P2P connectivity not available between all devices")
        print("Some benchmarks may fail or use fallback backends")
    
    # Get available backends
    available_backends = get_available_backends()
    backends_to_test = [b for b in available_backends if b in ["p2p", "nccl", "native"]]
    
    print(f"Available backends: {available_backends}")
    print(f"Backends to test: {backends_to_test}")
    
    # Create benchmark suite
    benchmark = P2PBenchmark(args.model_dir)
    
    # Run benchmarks
    if args.benchmark_type in ["throughput", "all"]:
        print(f"\n{'='*20} THROUGHPUT BENCHMARK {'='*20}")
        benchmark.benchmark_throughput(
            backend=backends_to_test,
            devices=devices,
            tensor_sizes=tensor_sizes,
            batch_sizes=batch_sizes,
            warmup_runs=args.warmup_runs,
            test_runs=args.test_runs
        )
    
    if args.benchmark_type in ["scalability", "all"]:
        print(f"\n{'='*20} SCALABILITY BENCHMARK {'='*20}")
        benchmark.benchmark_scalability(
            backends=backends_to_test,
            device_counts=device_counts,
            tensor_size=1024,  # Fixed size for scalability test
            warmup_runs=args.warmup_runs,
            test_runs=args.test_runs
        )
    
    if args.benchmark_type in ["latency", "all"]:
        print(f"\n{'='*20} LATENCY BENCHMARK {'='*20}")
        benchmark.benchmark_latency(
            backend=backends_to_test[0],  # Test with first available backend
            devices=devices,
            test_sizes=[128, 256, 512, 1024],
            num_tests=10
        )
    
    # Generate report and print summary
    benchmark.generate_report(args.output)
    benchmark.print_summary()
    
    print("\n✅ P2P Performance Benchmark completed!")


if __name__ == "__main__":
    main()