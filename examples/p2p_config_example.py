#!/usr/bin/env python3
"""
P2P Configuration Example

This example demonstrates various configuration options for the P2P backend,
including performance tuning parameters, environment variables, and mixed
backend configurations.
"""

import sys
import os
import torch
import argparse
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path to import exllamav3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3 import Generator, Job, model_init
from exllamav3.model.model_tp_cuda import check_p2p_connectivity
from exllamav3.model.model_tp_backend import get_available_backends, create_tp_backend


@dataclass
class P2PConfig:
    """Configuration class for P2P backend settings."""
    
    # Basic settings
    backend_type: str = "auto"  # "auto", "p2p", "nccl", "native"
    devices: List[int] = None
    verbose: bool = False
    
    # P2P-specific settings
    p2p_buffer_size: int = 16 * 1024 * 1024  # 16MB default
    p2p_timeout: float = 15.0  # 15 seconds timeout
    
    # Performance tuning
    use_optimized_kernels: bool = True
    enable_cpu_reduce: bool = True
    pin_memory: bool = True
    shared_memory_size: int = 2 * 1024 * 1024 * 1024  # 2GB
    
    # Environment variables
    env_vars: Dict[str, str] = None
    
    def __post_init__(self):
        if self.devices is None:
            self.devices = [0, 1] if torch.cuda.device_count() >= 2 else [0]
        if self.env_vars is None:
            self.env_vars = {}


class P2PConfigManager:
    """Manager for P2P configuration and testing."""
    
    def __init__(self, config: P2PConfig):
        self.config = config
        self.results = {}
        
    def setup_environment(self):
        """Set up environment variables for P2P backend."""
        print(f"\n=== Setting up Environment Variables ===")
        
        # Set P2P-specific environment variables
        if self.config.verbose:
            os.environ["EXLLAMA_P2P_VERBOSE"] = "1"
            print("🔍 Enabled verbose P2P logging")
        
        if "EXLLAMA_P2P_BUFFER_SIZE" in os.environ:
            print(f"⚠️  EXLLAMA_P2P_BUFFER_SIZE already set: {os.environ['EXLLAMA_P2P_BUFFER_SIZE']}")
        else:
            buffer_size = str(self.config.p2p_buffer_size)
            os.environ["EXLLAMA_P2P_BUFFER_SIZE"] = buffer_size
            print(f"📦 Set P2P buffer size: {buffer_size} bytes")
        
        # Set other relevant environment variables
        os.environ["EXLLAMA_MASTER_ADDR"] = "127.0.0.1"
        os.environ["EXLLAMA_MASTER_PORT"] = "29500"
        
        # Display current environment
        print("\n🔧 Current P2P-related environment variables:")
        for key, value in os.environ.items():
            if "EXLLAMA" in key or "CUDA" in key:
                print(f"   {key}={value}")
    
    def test_p2p_connectivity(self) -> bool:
        """Test P2P connectivity with current configuration."""
        print(f"\n=== Testing P2P Connectivity ===")
        print(f"Devices: {self.config.devices}")
        
        try:
            # Check basic connectivity
            if check_p2p_connectivity(self.config.devices):
                print("✅ P2P connectivity check passed")
                
                # Note: PyTorch automatically enables P2P access when needed for multi-GPU operations
                print("✅ P2P access will be managed automatically by PyTorch when needed")
                
                return True
            else:
                print("❌ P2P connectivity check failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during P2P connectivity test: {e}")
            return False
    
    def test_configuration(self, model_dir: str) -> Dict[str, Any]:
        """Test a specific configuration."""
        config_name = f"config_{len(self.results)}"
        print(f"\n=== Testing Configuration: {config_name} ===")
        
        result = {
            'name': config_name,
            'backend': self.config.backend_type,
            'devices': self.config.devices,
            'success': False,
            'error': None,
            'performance': None
        }
        
        try:
            # Set up environment
            self.setup_environment()
            
            # Test P2P connectivity
            if not self.test_p2p_connectivity():
                if self.config.backend_type == "p2p":
                    raise RuntimeError("P2P backend requires P2P connectivity")
                print("⚠️  P2P connectivity not available, will use fallback backend")
            
            # Load model with current configuration
            model_init_args = {
                'model_dir': model_dir,
                'devices': self.config.devices,
                'tp_backend': self.config.backend_type,
                'max_batch_size': 1,
                'max_seq_len': 2048,
                'max_input_len': 1024,
                'max_output_len': 512,
                'cache': True
            }
            
            print("🔄 Loading model...")
            model, config, cache, tokenizer = model_init.init_from_dict(model_init_args)
            print("✅ Model loaded successfully")
            
            # Test inference performance
            generator = Generator(
                model=model,
                cache=cache,
                tokenizer=tokenizer
            )
            
            # Test prompt
            test_prompt = "The performance of tensor parallel inference with the P2P backend is"
            input_ids = tokenizer.encode(test_prompt, add_bos=True)
            
            # Measure inference time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            job = Job(
                input_ids=input_ids,
                max_new_tokens=100,
                temperature=0.7
            )
            generator.enqueue(job)
            
            results = list(generator.iterate())
            end_time.record()
            
            # Wait for events to complete
            torch.cuda.synchronize()
            
            if results:
                inference_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                tokens_generated = results[-1].get("new_tokens", 0)
                tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
                
                result['performance'] = {
                    'tokens_generated': tokens_generated,
                    'inference_time': inference_time,
                    'tokens_per_second': tokens_per_second
                }
                
                print(f"✅ Performance: {tokens_per_second:.2f} tokens/second")
            else:
                result['error'] = "No inference results"
                print("❌ No inference results")
            
            # Clean up
            generator.close()
            model.unload()
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ Configuration failed: {e}")
        
        self.results[config_name] = result
        return result
    
    def generate_configurations(self) -> List[P2PConfig]:
        """Generate different configurations to test."""
        configs = []
        
        # Configuration 1: Auto detection (default)
        configs.append(P2PConfig(
            backend_type="auto",
            devices=[0, 1],
            verbose=False
        ))
        
        # Configuration 2: Explicit P2P with optimized settings
        configs.append(P2PConfig(
            backend_type="p2p",
            devices=[0, 1],
            verbose=True,
            p2p_buffer_size=32 * 1024 * 1024,  # 32MB
            use_optimized_kernels=True,
            pin_memory=True
        ))
        
        # Configuration 3: P2P with CPU optimization
        configs.append(P2PConfig(
            backend_type="p2p",
            devices=[0, 1],
            enable_cpu_reduce=True,
            shared_memory_size=4 * 1024 * 1024 * 1024  # 4GB
        ))
        
        # Configuration 4: Mixed backend fallback
        configs.append(P2PConfig(
            backend_type="auto",
            devices=[0],  # Single device to test fallback
            verbose=True
        ))
        
        # Configuration 5: NCCL comparison
        configs.append(P2PConfig(
            backend_type="nccl",
            devices=[0, 1],
            verbose=False
        ))
        
        # Configuration 6: Native backend comparison
        configs.append(P2PConfig(
            backend_type="native",
            devices=[0, 1],
            verbose=False
        ))
        
        return configs


def demonstrate_environment_variables():
    """Demonstrate the effect of different environment variables."""
    print("\n=== Environment Variable Effects ===")
    
    # Test different buffer sizes
    buffer_sizes = [8 * 1024 * 1024, 16 * 1024 * 1024, 32 * 1024 * 1024, 64 * 1024 * 1024]
    
    for buffer_size in buffer_sizes:
        os.environ["EXLLAMA_P2P_BUFFER_SIZE"] = str(buffer_size)
        print(f"Buffer size: {buffer_size / 1024 / 1024:.1f}MB")
        
        # Simulate the effect (in real scenario, this would affect actual performance)
        simulated_performance = 100 + (buffer_size / 1024 / 1024) * 5  # Simulated tokens/second
        print(f"  Simulated performance: {simulated_performance:.1f} tokens/s")


def demonstrate_mixed_backend_config():
    """Demonstrate mixed backend configurations."""
    print("\n=== Mixed Backend Configuration Demo ===")
    
    # Configuration for mixed backend usage
    mixed_configs = [
        {
            "name": "P2P + NCCL Hybrid",
            "description": "Use P2P for primary communication, NCCL for fallback",
            "config": {
                "backend": "auto",
                "devices": [0, 1, 2],
                "fallback_backend": "nccl"
            }
        },
        {
            "name": "P2P + Native Fallback",
            "description": "Use P2P when available, fall back to native",
            "config": {
                "backend": "auto", 
                "devices": [0, 1],
                "fallback_backend": "native"
            }
        },
        {
            "name": "Explicit Backend Selection",
            "description": "Manually specify which backend to use for each operation",
            "config": {
                "backend_type": "p2p",
                "operations": {
                    "all_reduce": "p2p",
                    "broadcast": "nccl", 
                    "gather": "p2p"
                }
            }
        }
    ]
    
    for config in mixed_configs:
        print(f"\n📋 {config['name']}")
        print(f"   Description: {config['description']}")
        print(f"   Configuration: {json.dumps(config['config'], indent=6)}")


def demonstrate_performance_tuning():
    """Demonstrate performance tuning parameters."""
    print("\n=== Performance Tuning Parameters ===")
    
    tuning_params = [
        {
            "parameter": "p2p_buffer_size",
            "description": "Size of P2P communication buffer",
            "values": [8, 16, 32, 64],  # MB
            "effect": "Larger buffers reduce communication overhead but increase memory usage"
        },
        {
            "parameter": "pin_memory",
            "description": "Whether to pin memory for faster GPU transfers",
            "values": [True, False],
            "effect": "True improves performance but uses more system memory"
        },
        {
            "parameter": "use_optimized_kernels",
            "description": "Use optimized P2P communication kernels",
            "values": [True, False],
            "effect": "True provides better performance on supported hardware"
        },
        {
            "parameter": "enable_cpu_reduce",
            "description": "Enable CPU-based reduction operations",
            "values": [True, False],
            "effect": "True can improve performance for certain operations"
        }
    ]
    
    for param in tuning_params:
        print(f"\n🔧 {param['parameter']}")
        print(f"   Description: {param['description']}")
        print(f"   Values: {param['values']}")
        print(f"   Effect: {param['effect']}")


def main():
    parser = argparse.ArgumentParser(description="P2P Configuration Example")
    model_init.add_args(parser, cache=True)
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing the model")
    parser.add_argument("--devices", type=str, default="0,1", 
                       help="Comma-separated list of device IDs to use")
    parser.add_argument("--config-file", type=str, help="JSON file with custom configuration")
    parser.add_argument("--test-only", type=str, help="Test only specific configuration (auto, p2p, nccl, native)")
    
    args = parser.parse_args()
    
    # Parse devices
    devices = [int(d.strip()) for d in args.devices.split(",")]
    
    print("🚀 P2P Configuration Example")
    print("=" * 50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. P2P backend requires CUDA.")
        return
    
    # Load configuration from file if provided
    config = None
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                config_data = json.load(f)
                config = P2PConfig(**config_data)
                print(f"📋 Loaded configuration from {args.config_file}")
        except Exception as e:
            print(f"❌ Error loading configuration file: {e}")
            return
    
    # If no config file, create default manager
    if config is None:
        config = P2PConfig(devices=devices)
    
    # Create config manager
    manager = P2PConfigManager(config)
    
    # Demonstrate environment variables
    demonstrate_environment_variables()
    
    # Demonstrate mixed backend configurations
    demonstrate_mixed_backend_config()
    
    # Demonstrate performance tuning
    demonstrate_performance_tuning()
    
    # Test configurations
    if args.test_only:
        # Test specific configuration
        config.backend_type = args.test_only
        print(f"\n🔬 Testing specific configuration: {args.test_only}")
        manager.test_configuration(args.model_dir)
    else:
        # Test all configurations
        print(f"\n🔬 Testing all configurations...")
        test_configs = manager.generate_configurations()
        
        for test_config in test_configs:
            manager.config = test_config
            manager.test_configuration(args.model_dir)
    
    # Print results summary
    print("\n" + "=" * 50)
    print("📊 Configuration Test Results Summary")
    print("=" * 50)
    
    successful_configs = [r for r in manager.results.values() if r['success']]
    failed_configs = [r for r in manager.results.values() if not r['success']]
    
    print(f"✅ Successful configurations: {len(successful_configs)}")
    print(f"❌ Failed configurations: {len(failed_configs)}")
    
    if successful_configs:
        print("\n🏆 Best performing configuration:")
        best_config = max(successful_configs, key=lambda x: x['performance']['tokens_per_second'] if x['performance'] else 0)
        print(f"   Name: {best_config['name']}")
        print(f"   Backend: {best_config['backend']}")
        if best_config['performance']:
            print(f"   Performance: {best_config['performance']['tokens_per_second']:.2f} tokens/s")
    
    if failed_configs:
        print("\n⚠️  Failed configurations:")
        for config in failed_configs:
            print(f"   {config['name']}: {config['error']}")
    
    print("\n✅ P2P Configuration Example completed!")


if __name__ == "__main__":
    main()