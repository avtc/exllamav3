"""
P2P Monitoring Configuration for ExLlamaV3

This module provides configuration options for P2P monitoring, profiling,
and debugging tools.
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class P2PMonitoringConfig:
    """Configuration for P2P monitoring."""
    # Basic monitoring settings
    enable_monitoring: bool = True
    monitoring_level: str = "basic"  # "basic", "detailed", "comprehensive"
    max_history_size: int = 10000
    enable_real_time: bool = True
    output_dir: str = "./p2p_monitoring"
    
    # Performance tracking
    track_bandwidth: bool = True
    track_latency: bool = True
    track_memory_usage: bool = True
    track_gpu_utilization: bool = True
    track_success_rates: bool = True
    
    # Operation filtering
    monitor_operations: List[str] = None  # None means all operations
    exclude_operations: List[str] = None
    
    # Device filtering
    monitor_devices: List[int] = None  # None means all devices
    exclude_devices: List[int] = None
    
    # Alerting
    enable_alerts: bool = False
    alert_thresholds: Dict[str, float] = None
    
    # Export settings
    auto_export: bool = True
    export_interval: int = 300  # seconds
    export_format: str = "json"  # "json", "csv"


@dataclass
class P2PProfilingConfig:
    """Configuration for P2P profiling."""
    # Basic profiling settings
    enable_profiling: bool = True
    profiling_level: str = "basic"  # "basic", "detailed", "comprehensive"
    max_sessions: int = 100
    output_dir: str = "./p2p_profiling"
    
    # Profiling options
    profile_all_operations: bool = False
    profile_operations: List[str] = None  # None means all operations
    min_iterations: int = 5
    warmup_iterations: int = 3
    
    # Performance analysis
    enable_algorithm_comparison: bool = True
    enable_bottleneck_detection: bool = True
    enable_optimization_suggestions: bool = True
    
    # Export settings
    auto_export: bool = True
    export_format: str = "json"


@dataclass
class P2PDebuggingConfig:
    """Configuration for P2P debugging."""
    # Basic debugging settings
    enable_debugging: bool = False
    debug_level: str = "basic"  # "basic", "detailed", "verbose"
    max_events: int = 10000
    output_dir: str = "./p2p_debugging"
    
    # Debugging options
    trace_operations: bool = True
    trace_memory_operations: bool = False
    trace_topology_changes: bool = True
    
    # Error handling
    log_all_errors: bool = True
    log_warnings: bool = True
    max_error_history: int = 1000
    
    # Communication tracing
    enable_communication_traces: bool = False
    max_trace_depth: int = 100
    
    # Export settings
    auto_export: bool = True
    export_format: str = "json"


@dataclass
class P2PDashboardConfig:
    """Configuration for P2P performance dashboard."""
    # Dashboard settings
    enable_dashboard: bool = False
    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = False
    
    # Visualization options
    enable_charts: bool = True
    chart_refresh_interval: int = 5  # seconds
    max_data_points: int = 1000
    
    # Dashboard features
    show_topology: bool = True
    show_performance_metrics: bool = True
    show_debug_info: bool = False
    show_alerts: bool = True


@dataclass
class P2PBenchmarkConfig:
    """Configuration for P2P benchmark suite."""
    # Benchmark settings
    enable_benchmarks: bool = False
    output_dir: str = "./p2p_benchmarks"
    
    # Test configurations
    test_tensor_sizes: List[int] = None
    test_dtypes: List[str] = None
    test_algorithms: List[str] = None
    test_operations: List[str] = None
    
    # Benchmark parameters
    num_iterations: int = 10
    warmup_iterations: int = 3
    test_all_configs: bool = False
    
    # Benchmark types
    run_comprehensive: bool = False
    run_scalability: bool = False
    run_algorithm_comparison: bool = False
    
    # Export settings
    auto_export: bool = True
    export_format: str = "json"


@dataclass
class P2PConfig:
    """Complete P2P monitoring configuration."""
    monitoring: P2PMonitoringConfig
    profiling: P2PProfilingConfig
    debugging: P2PDebuggingConfig
    dashboard: P2PDashboardConfig
    benchmark: P2PBenchmarkConfig
    
    def __post_init__(self):
        """Initialize default values for lists and dicts."""
        if self.monitoring.monitor_operations is None:
            self.monitoring.monitor_operations = ["broadcast", "all_reduce", "gather", "direct_copy"]
        
        if self.monitoring.exclude_operations is None:
            self.monitoring.exclude_operations = []
        
        if self.monitoring.monitor_devices is None:
            self.monitoring.monitor_devices = []
        
        if self.monitoring.exclude_devices is None:
            self.monitoring.exclude_devices = []
        
        if self.monitoring.alert_thresholds is None:
            self.monitoring.alert_thresholds = {
                "min_bandwidth_gbps": 1.0,
                "max_latency_ms": 100.0,
                "min_success_rate": 0.95,
                "max_memory_usage_percent": 90.0
            }
        
        if self.profiling.profile_operations is None:
            self.profiling.profile_operations = ["broadcast", "all_reduce", "gather", "direct_copy"]
        
        if self.benchmark.test_tensor_sizes is None:
            self.benchmark.test_tensor_sizes = [1024, 1024*1024, 1024*1024*4]
        
        if self.benchmark.test_dtypes is None:
            self.benchmark.test_dtypes = ["float32", "float16", "int32"]
        
        if self.benchmark.test_algorithms is None:
            self.benchmark.test_algorithms = ["ring", "binary_tree", "kary_tree", "balanced_tree"]
        
        if self.benchmark.test_operations is None:
            self.benchmark.test_operations = ["broadcast", "all_reduce", "gather", "direct_copy"]


class P2PConfigManager:
    """Manager for P2P configuration."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or "p2p_config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> P2PConfig:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_dict = json.load(f)
                
                # Create config from dict
                return P2PConfig(
                    monitoring=P2PMonitoringConfig(**config_dict.get("monitoring", {})),
                    profiling=P2PProfilingConfig(**config_dict.get("profiling", {})),
                    debugging=P2PDebuggingConfig(**config_dict.get("debugging", {})),
                    dashboard=P2PDashboardConfig(**config_dict.get("dashboard", {})),
                    benchmark=P2PBenchmarkConfig(**config_dict.get("benchmark", {}))
                )
            except Exception as e:
                print(f"Failed to load P2P config from {self.config_file}: {e}")
                print("Using default configuration.")
        
        # Return default configuration
        return P2PConfig(
            monitoring=P2PMonitoringConfig(),
            profiling=P2PProfilingConfig(),
            debugging=P2PDebuggingConfig(),
            dashboard=P2PDashboardConfig(),
            benchmark=P2PBenchmarkConfig()
        )
    
    def save_config(self, config_file: Optional[str] = None):
        """
        Save configuration to file.
        
        Args:
            config_file: Path to save configuration (uses default if None)
        """
        save_path = config_file or self.config_file
        
        try:
            with open(save_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            print(f"P2P configuration saved to {save_path}")
        except Exception as e:
            print(f"Failed to save P2P config to {save_path}: {e}")
    
    def get_config(self) -> P2PConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs):
        """
        Update configuration with new values.
        
        Args:
            **kwargs: Configuration updates
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Unknown configuration key: {key}")
    
    def update_monitoring_config(self, **kwargs):
        """Update monitoring configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config.monitoring, key):
                setattr(self.config.monitoring, key, value)
            else:
                print(f"Unknown monitoring configuration key: {key}")
    
    def update_profiling_config(self, **kwargs):
        """Update profiling configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config.profiling, key):
                setattr(self.config.profiling, key, value)
            else:
                print(f"Unknown profiling configuration key: {key}")
    
    def update_debugging_config(self, **kwargs):
        """Update debugging configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config.debugging, key):
                setattr(self.config.debugging, key, value)
            else:
                print(f"Unknown debugging configuration key: {key}")
    
    def update_dashboard_config(self, **kwargs):
        """Update dashboard configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config.dashboard, key):
                setattr(self.config.dashboard, key, value)
            else:
                print(f"Unknown dashboard configuration key: {key}")
    
    def update_benchmark_config(self, **kwargs):
        """Update benchmark configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config.benchmark, key):
                setattr(self.config.benchmark, key, value)
            else:
                print(f"Unknown benchmark configuration key: {key}")
    
    def create_default_config_file(self, config_file: Optional[str] = None):
        """Create a default configuration file."""
        save_path = config_file or self.config_file
        
        default_config = P2PConfig(
            monitoring=P2PMonitoringConfig(),
            profiling=P2PProfilingConfig(),
            debugging=P2PDebuggingConfig(),
            dashboard=P2PDashboardConfig(),
            benchmark=P2PBenchmarkConfig()
        )
        
        try:
            with open(save_path, 'w') as f:
                json.dump(asdict(default_config), f, indent=2)
            print(f"Default P2P configuration created at {save_path}")
        except Exception as e:
            print(f"Failed to create default config file: {e}")
    
    def print_config(self):
        """Print current configuration."""
        print("P2P Configuration:")
        print("=" * 50)
        print(json.dumps(asdict(self.config), indent=2))


# Global configuration manager
_global_config_manager: Optional[P2PConfigManager] = None


def get_global_config_manager() -> P2PConfigManager:
    """Get the global configuration manager."""
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = P2PConfigManager()
    
    return _global_config_manager


def get_global_config() -> P2PConfig:
    """Get the global configuration."""
    return get_global_config_manager().get_config()


def initialize_global_config(config_file: Optional[str] = None) -> P2PConfigManager:
    """
    Initialize the global configuration manager.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Global configuration manager
    """
    global _global_config_manager
    
    _global_config_manager = P2PConfigManager(config_file)
    return _global_config_manager


def load_config_from_env():
    """Load configuration from environment variables."""
    config_manager = get_global_config_manager()
    config = config_manager.get_config()
    
    # Load monitoring config from environment
    if "P2P_MONITORING_LEVEL" in os.environ:
        config.monitoring.monitoring_level = os.environ["P2P_MONITORING_LEVEL"]
    
    if "P2P_MONITORING_OUTPUT_DIR" in os.environ:
        config.monitoring.output_dir = os.environ["P2P_MONITORING_OUTPUT_DIR"]
    
    if "P2P_ENABLE_MONITORING" in os.environ:
        config.monitoring.enable_monitoring = os.environ["P2P_ENABLE_MONITORING"].lower() == "true"
    
    # Load profiling config from environment
    if "P2P_PROFILING_LEVEL" in os.environ:
        config.profiling.profiling_level = os.environ["P2P_PROFILING_LEVEL"]
    
    if "P2P_PROFILING_OUTPUT_DIR" in os.environ:
        config.profiling.output_dir = os.environ["P2P_PROFILING_OUTPUT_DIR"]
    
    if "P2P_ENABLE_PROFILING" in os.environ:
        config.profiling.enable_profiling = os.environ["P2P_ENABLE_PROFILING"].lower() == "true"
    
    # Load debugging config from environment
    if "P2P_DEBUG_LEVEL" in os.environ:
        config.debugging.debug_level = os.environ["P2P_DEBUG_LEVEL"]
    
    if "P2P_DEBUG_OUTPUT_DIR" in os.environ:
        config.debugging.output_dir = os.environ["P2P_DEBUG_OUTPUT_DIR"]
    
    if "P2P_ENABLE_DEBUGGING" in os.environ:
        config.debugging.enable_debugging = os.environ["P2P_ENABLE_DEBUGGING"].lower() == "true"
    
    # Load dashboard config from environment
    if "P2P_DASHBOARD_HOST" in os.environ:
        config.dashboard.host = os.environ["P2P_DASHBOARD_HOST"]
    
    if "P2P_DASHBOARD_PORT" in os.environ:
        config.dashboard.port = int(os.environ["P2P_DASHBOARD_PORT"])
    
    if "P2P_ENABLE_DASHBOARD" in os.environ:
        config.dashboard.enable_dashboard = os.environ["P2P_ENABLE_DASHBOARD"].lower() == "true"
    
    return config_manager


def create_example_config_file(filename: str = "p2p_config_example.json"):
    """Create an example configuration file with comments."""
    example_config = {
        "monitoring": {
            "enable_monitoring": True,
            "monitoring_level": "basic",
            "max_history_size": 10000,
            "enable_real_time": True,
            "output_dir": "./p2p_monitoring",
            "track_bandwidth": True,
            "track_latency": True,
            "track_memory_usage": True,
            "track_gpu_utilization": True,
            "track_success_rates": True,
            "monitor_operations": ["broadcast", "all_reduce", "gather", "direct_copy"],
            "exclude_operations": [],
            "monitor_devices": [],
            "exclude_devices": [],
            "enable_alerts": False,
            "alert_thresholds": {
                "min_bandwidth_gbps": 1.0,
                "max_latency_ms": 100.0,
                "min_success_rate": 0.95,
                "max_memory_usage_percent": 90.0
            },
            "auto_export": True,
            "export_interval": 300,
            "export_format": "json"
        },
        "profiling": {
            "enable_profiling": True,
            "profiling_level": "basic",
            "max_sessions": 100,
            "output_dir": "./p2p_profiling",
            "profile_all_operations": False,
            "profile_operations": ["broadcast", "all_reduce", "gather", "direct_copy"],
            "min_iterations": 5,
            "warmup_iterations": 3,
            "enable_algorithm_comparison": True,
            "enable_bottleneck_detection": True,
            "enable_optimization_suggestions": True,
            "auto_export": True,
            "export_format": "json"
        },
        "debugging": {
            "enable_debugging": False,
            "debug_level": "basic",
            "max_events": 10000,
            "output_dir": "./p2p_debugging",
            "trace_operations": True,
            "trace_memory_operations": False,
            "trace_topology_changes": True,
            "log_all_errors": True,
            "log_warnings": True,
            "max_error_history": 1000,
            "enable_communication_traces": False,
            "max_trace_depth": 100,
            "auto_export": True,
            "export_format": "json"
        },
        "dashboard": {
            "enable_dashboard": False,
            "host": "127.0.0.1",
            "port": 8080,
            "debug": False,
            "enable_charts": True,
            "chart_refresh_interval": 5,
            "max_data_points": 1000,
            "show_topology": True,
            "show_performance_metrics": True,
            "show_debug_info": False,
            "show_alerts": True
        },
        "benchmark": {
            "enable_benchmarks": False,
            "output_dir": "./p2p_benchmarks",
            "test_tensor_sizes": [1024, 1048576, 4194304],
            "test_dtypes": ["float32", "float16", "int32"],
            "test_algorithms": ["ring", "binary_tree", "kary_tree", "balanced_tree"],
            "test_operations": ["broadcast", "all_reduce", "gather", "direct_copy"],
            "num_iterations": 10,
            "warmup_iterations": 3,
            "test_all_configs": False,
            "run_comprehensive": False,
            "run_scalability": False,
            "run_algorithm_comparison": False,
            "auto_export": True,
            "export_format": "json"
        }
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(example_config, f, indent=2)
        print(f"Example P2P configuration created at {filename}")
    except Exception as e:
        print(f"Failed to create example config file: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="P2P Configuration Manager")
    parser.add_argument("--create-example", action="store_true",
                        help="Create example configuration file")
    parser.add_argument("--config-file", default="p2p_config.json",
                        help="Configuration file path")
    parser.add_argument("--print-config", action="store_true",
                        help="Print current configuration")
    parser.add_argument("--load-env", action="store_true",
                        help="Load configuration from environment variables")
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_config_file()
    
    config_manager = get_global_config_manager()
    
    if args.load_env:
        load_config_from_env()
    
    if args.print_config:
        config_manager.print_config()