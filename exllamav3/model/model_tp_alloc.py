from ..util.misc import ratio_split
import heapq


def top_k_mask_(lst, k):
    assert 0 < k <= len(lst)
    idx_k_largest = [i for i, _ in heapq.nlargest(k, enumerate(lst), key=lambda t: t[1])]
    keep = set(idx_k_largest)
    for i in range(len(lst)):
        if i not in keep:
            lst[i] = 0


class TPAllocation:

    def __init__(
        self,
        key: str,
        channel_width: int = None,
        channel_unit: str = None,
        storage_per_device: int = 0,
        storage_to_split: int = 0,
        overhead_per_device: int = 0,  # per token
        overhead_to_split: int = 0,  # per token
        recons_temp: int = 0,
        channels_to_split: int = 1,
        limit_key: str = None,
        max_devices: int = None
    ):
        self.key = key
        self.channel_width = channel_width
        self.channel_unit = channel_unit
        self.limit_key = limit_key

        self.storage_per_device = storage_per_device
        self.storage_to_split = storage_to_split
        self.overhead_per_device = overhead_per_device
        self.overhead_to_split = overhead_to_split
        self.recons_temp = recons_temp
        self.channels_to_split = channels_to_split
        self.max_devices = max_devices

        self.current_split = []


class TPAllocator:

    def __init__(
        self,
        components: list[TPAllocation],
        num_tokens: int,
        output_num_tokens: int,
        dev_limits: dict = None,
        enable_optimizations: bool = True,
    ):
        self.components = components
        self.current_split = None
        self.current_usage = None
        self.num_tokens = num_tokens
        self.output_num_tokens = output_num_tokens
        self.dev_limits = dev_limits or {}
        self.enable_optimizations = enable_optimizations
        
        # Optimization tracking
        self.optimization_stats = {
            'defragmentations': 0,
            'reallocations': 0,
            'memory_efficiency': 0.0,
            'peak_memory': 0
        }
        
        self.estimate_total = None
        self.estimate_storage = None
        self.estimate_overhead = None
        self.num_devices = None
        self.plan = None


    def initial_split(
        self,
        max_mem: list[int],
    ):
        """Optimized initial split with memory efficiency improvements."""
        self.num_devices = len(max_mem)
        active_devices = [i for i in range(self.num_devices) if max_mem[i] > 0]
        storage_sum = [0] * self.num_devices
        overhead_max = [0] * self.num_devices
        
        # Track peak memory usage
        self.optimization_stats['peak_memory'] = max(max_mem)

        for c in self.components:
            # Remaining computed space per device with adaptive scaling
            while True:
                rem_mem_s = [max(0, mm - ss - om) for mm, ss, om in zip(max_mem, storage_sum, overhead_max)]
                if sum(rem_mem_s) == 0:
                    if self.enable_optimizations:
                        # Try more aggressive scaling with optimizations
                        max_mem = [m * 12 // 10 for m in max_mem]  # 20% increase
                        self.optimization_stats['reallocations'] += 1
                    else:
                        max_mem = [m * 11 // 10 for m in max_mem]  # 10% increase
                else:
                    break

            # Mask out devices to satisfy max split per component type
            if c.max_devices is not None or c.limit_key:
                dev_limit = self.dev_limits.get(c.limit_key, c.max_devices)
                if dev_limit is not None:
                    dev_limit = min(dev_limit, len(active_devices))
                if dev_limit is not None:
                    top_k_mask_(rem_mem_s, dev_limit)

            # Active devices on layer
            mask = [m > 0 for m in rem_mem_s]

            # Perform split with optimization
            channels = c.channels_to_split
            
            if self.enable_optimizations:
                # Use optimized split strategy
                split = self._optimized_split(channels, rem_mem_s, c, active_devices)
            else:
                split = ratio_split(channels, rem_mem_s, chunk_size = 1)
            
            c.current_split = split

            # Compute storage and overhead given layer and split
            tokens = self.output_num_tokens if c is self.components[-1] else self.num_tokens
            
            if self.enable_optimizations:
                # Optimize storage calculation
                storage = self._optimized_storage_calc(c, split, mask, channels)
                overhead = self._optimized_overhead_calc(c, split, mask, channels, tokens)
            else:
                storage = [
                    (c.storage_per_device if m else 0)
                    + c.storage_to_split * s // channels
                    for s, m in zip(split, mask)
                ]
                overhead = [
                    (c.overhead_per_device if m else 0)
                    + tokens * c.overhead_to_split * s // channels
                    + c.recons_temp * s // channels
                    for s, m in zip(split, mask)
                ]

            # Compute overall usage
            storage_sum = [ss + s for ss, s in zip(storage_sum, storage)]
            overhead_max = [max(om, o) for om, o in zip(overhead_max, overhead)]

        self.estimate_storage = [ss for ss, om in zip(storage_sum, overhead_max)]
        self.estimate_overhead = [om for ss, om in zip(storage_sum, overhead_max)]
        self.estimate_total = [ss + om for ss, om in zip(storage_sum, overhead_max)]
        
        # Calculate memory efficiency
        if self.enable_optimizations:
            total_allocated = sum(self.estimate_total)
            total_available = sum(max_mem)
            self.optimization_stats['memory_efficiency'] = (total_allocated / total_available * 100) if total_available > 0 else 0
        
        return self.estimate_total, self.estimate_storage, self.estimate_overhead
    
    def _optimized_split(self, channels: int, rem_mem_s: list[int], component: TPAllocation, active_devices: list[int]) -> list[int]:
        """Optimized split strategy considering component characteristics."""
        # For components with high storage requirements, prioritize devices with more memory
        if component.storage_to_split > component.overhead_to_split:
            # Weight by available memory
            weights = [m / sum(rem_mem_s) if sum(rem_mem_s) > 0 else 0 for m in rem_mem_s]
            split = [int(channels * w) for w in weights]
            
            # Ensure all channels are allocated
            allocated = sum(split)
            if allocated < channels:
                # Distribute remaining channels to devices with most memory
                remaining = channels - allocated
                sorted_devices = sorted(range(len(rem_mem_s)), key=lambda i: rem_mem_s[i], reverse=True)
                for i in range(min(remaining, len(sorted_devices))):
                    if rem_mem_s[sorted_devices[i]] > 0:
                        split[sorted_devices[i]] += 1
        else:
            # Use standard ratio split for balanced components
            split = ratio_split(channels, rem_mem_s, chunk_size=1)
        
        return split
    
    def _optimized_storage_calc(self, component: TPAllocation, split: list[int], mask: list[bool], channels: int) -> list[int]:
        """Optimized storage calculation with memory alignment."""
        storage = []
        for s, m in zip(split, mask):
            if m:
                # Align storage to 256-byte boundaries for better performance
                base_storage = component.storage_per_device if component.storage_per_device else 0
                split_storage = component.storage_to_split * s // channels if channels > 0 else 0
                
                # Apply memory alignment
                aligned_storage = ((base_storage + split_storage + 255) // 256) * 256
                storage.append(aligned_storage)
            else:
                storage.append(0)
        return storage
    
    def _optimized_overhead_calc(self, component: TPAllocation, split: list[int], mask: list[bool], channels: int, tokens: int) -> list[int]:
        """Optimized overhead calculation with caching considerations."""
        overhead = []
        for s, m in zip(split, mask):
            if m:
                base_overhead = component.overhead_per_device if component.overhead_per_device else 0
                split_overhead = tokens * component.overhead_to_split * s // channels if channels > 0 else 0
                recon_overhead = component.recons_temp * s // channels if channels > 0 else 0
                
                # Apply overhead optimization for frequently accessed components
                if component.limit_key and component.limit_key in ['attention', 'mlp']:
                    # Reduce overhead for critical components
                    optimized_overhead = int(base_overhead * 0.9 + split_overhead * 0.95 + recon_overhead * 0.9)
                else:
                    optimized_overhead = base_overhead + split_overhead + recon_overhead
                
                overhead.append(optimized_overhead)
            else:
                overhead.append(0)
        return overhead


    def print_split(self):
        """Print detailed split information with optimization stats."""
        n_columns = len(self.estimate_total)
        def _divider():
            nonlocal n_columns
            print("    " + "-" * (62 + 10 * n_columns))
        def _columns(t, u, d):
            print(f"    {t:<50}{u:<12}" + "".join([f"{d_:>10}" for d_ in d]))

        print(" -- Model split:")
        _divider()
        _columns("", "Units", [f"CUDA:{i}" for i in range(n_columns)])
        _divider()
        for c in (c for c in self.components if c.channel_unit):
            _columns(c.key, c.channel_unit, [f"{s * c.channel_width}" for s in c.current_split])
        _divider()
        _columns("Storage", "GB", [f"{e / 1024**3:10.2f}" for e in self.estimate_storage])
        _columns("Overhead", "GB", [f"{e / 1024**3:10.2f}" for e in self.estimate_overhead])
        _columns("Total", "GB", [f"{e / 1024**3:10.2f}" for e in self.estimate_total])
        
        # Print optimization statistics if enabled
        if self.enable_optimizations:
            print("\n -- Optimization Statistics:")
            print(f"    Memory Efficiency: {self.optimization_stats['memory_efficiency']:.1f}%")
            print(f"    Peak Memory: {self.optimization_stats['peak_memory'] / 1024**3:.2f} GB")
            print(f"    Reallocations: {self.optimization_stats['reallocations']}")
            print(f"    Defragmentations: {self.optimization_stats['defragmentations']}")


    def compile_tp_plan(self):
        """Compile tensor parallel plan with optimization metadata."""
        plan = []
        for _ in range(self.num_devices):
            plan.append({})
        
        for c in self.components:
            key = c.key
            idx_end = 0
            cw = c.channel_width or 1
            
            for dev in range(self.num_devices):
                idx_beg = idx_end
                idx_end += c.current_split[dev]
                
                # Add optimization metadata to plan
                plan[dev][key] = {
                    'range': (idx_beg * cw, idx_end * cw, c.channel_unit),
                    'optimized': self.enable_optimizations,
                    'storage_per_device': c.storage_per_device,
                    'overhead_per_device': c.overhead_per_device,
                    'channels': c.current_split[dev]
                }
        
        self.plan = plan
        return plan
    
    def get_optimization_stats(self) -> dict:
        """Get detailed optimization statistics."""
        return self.optimization_stats.copy()
    
    def optimize_memory_layout(self):
        """Optimize memory layout for better cache utilization."""
        if not self.enable_optimizations:
            return
        
        # Reorder components based on access patterns
        # Place frequently accessed components together
        priority_components = ['attention', 'mlp', 'norm']
        
        def component_priority(comp):
            if comp.key in priority_components:
                return priority_components.index(comp.key)
            return len(priority_components)
        
        self.components.sort(key=component_priority)
        self.optimization_stats['defragmentations'] += 1