# P2P Backend Tests

This directory contains comprehensive tests for the P2P (Peer-to-Peer) backend functionality in exllamav3. The tests validate both functionality and performance aspects of the P2P backend, ensuring it works correctly and provides the expected performance improvements.

## Test Structure

The P2P tests are organized into 5 main test files:

1. **`test_p2p_backend.py`** - Unit tests for P2P backend initialization and detection
2. **`test_p2p_integration.py`** - Integration tests for end-to-end P2P communication
3. **`test_p2p_performance.py`** - Performance benchmarking and scalability tests
4. **`test_p2p_error_handling.py`** - Error handling and recovery scenario tests
5. **`test_p2p_backend_selection.py`** - Backend selection logic and configuration tests

## Prerequisites

### System Requirements
- Python 3.8 or higher
- PyTorch 2.6.0 or higher
- CUDA 11.8 or higher (for GPU testing)
- NCCL 2.16.5 or higher
- At least 2 GPUs with P2P connectivity support

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -r requirements_test.txt
```

2. **Set up conda environment (recommended):**
```bash
conda env create -f environment_test.yml
conda activate exllamav3-p2p-test
```

3. **Build the exllamav3 extension:**
```bash
pip install -e .
```

## Running Tests

### Using the Test Runner Script

The recommended way to run P2P tests is using the `run_p2p_tests.py` script:

```bash
# Run all P2P tests
python run_p2p_tests.py

# Run specific test suites
python run_p2p_tests.py --unit        # Unit tests only
python run_p2p_tests.py --integration  # Integration tests only
python run_p2p_tests.py --performance  # Performance tests only
python run_p2p_tests.py --error        # Error handling tests only
python run_p2p_tests.py --auto        # Backend selection tests only

# Run with coverage
python run_p2p_tests.py --coverage

# Run benchmarks
python run_p2p_tests.py --benchmark

# Run in parallel
python run_p2p_tests.py --parallel

# Run specific test file
python run_p2p_tests.py --file tests/test_p2p_backend.py

# Run with specific markers
python run_p2p_tests.py --markers "p2p,unit"

# Check environment
python run_p2p_tests.py --env-check

# Get JSON output
python run_p2p_tests.py --json
```

### Using pytest directly

You can also run tests using pytest directly:

```bash
# Run all P2P tests
pytest tests/ -v -m p2p

# Run specific test file
pytest tests/test_p2p_backend.py -v

# Run with coverage
pytest tests/ --cov=exllamav3 --cov-report=html --cov-report=term-missing

# Run with benchmarks
pytest tests/ --benchmark-only

# Run with specific markers
pytest tests/ -m "p2p and unit"
```

### Test Markers

The tests use pytest markers to categorize different types of tests:

- `unit`: Unit tests for individual components
- `integration`: Integration tests for multiple components
- `performance`: Performance benchmarking tests
- `stress`: Stress tests for performance validation
- `slow`: Tests that take a long time to run
- `gpu`: Tests that require GPU (will be skipped if no GPU)
- `p2p`: P2P-specific tests
- `nccl`: NCCL backend tests
- `native`: Native backend tests
- `auto`: Auto-selection logic tests
- `error`: Error handling and recovery tests
- `mock`: Tests that use mocking
- `serial`: Tests that must run serially
- `flaky`: Tests that are known to be flaky

## Test Coverage

### Unit Tests (`test_p2p_backend.py`)
- P2P connectivity detection functions
- TPBackendP2P initialization with valid/invalid configurations
- P2P memory management utilities
- Backend selection logic and automatic detection
- Mocking for multi-GPU scenarios

### Integration Tests (`test_p2p_integration.py`)
- End-to-end P2P communication operations (all_reduce, broadcast, gather, barrier)
- Performance comparison between P2P and NCCL backends
- Memory usage and leak detection
- Error handling and recovery scenarios
- Multi-process testing scenarios

### Performance Tests (`test_p2p_performance.py`)
- All-reduce performance comparison between backends
- Broadcast performance testing
- Gather operation performance
- Throughput and latency measurements
- Scalability testing with different tensor sizes
- Performance regression detection
- Memory usage analysis

### Error Handling Tests (`test_p2p_error_handling.py`)
- P2P connectivity failure scenarios
- Resource cleanup on errors
- Graceful fallback to other backends
- Edge cases and boundary conditions
- Memory leak detection in error scenarios
- Recovery mechanisms validation

### Backend Selection Tests (`test_p2p_backend_selection.py`)
- Automatic backend selection logic
- Manual backend specification
- Mixed connectivity scenarios
- Configuration validation
- Backend priority and fallback logic

## Environment Configuration

### GPU Configuration
For optimal testing, ensure you have:
- Multiple GPUs with P2P connectivity enabled
- CUDA drivers properly installed
- NCCL library available
- Sufficient GPU memory for testing

### Environment Variables
```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set PyTorch backend preferences
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

# Set logging level for debugging
export LOG_LEVEL=DEBUG
```

### Multi-GPU Testing
The tests are designed to work with multiple GPUs. If you have fewer GPUs than expected, some tests may be skipped or use mocking.

## Performance Analysis

### Benchmark Results
The performance tests generate detailed benchmark results including:
- Throughput (GB/s)
- Latency (ms)
- Scalability across different tensor sizes
- Memory usage patterns
- Comparison with NCCL backend

### Performance Metrics
Key metrics tracked:
- All-reduce bandwidth
- Broadcast latency
- Gather operation time
- Memory allocation efficiency
- GPU utilization

### Performance Regression Detection
The tests include regression detection to ensure performance doesn't degrade over time.

## Troubleshooting

### Common Issues

1. **CUDA out of memory:**
   - Reduce batch size in tests
   - Use smaller tensor sizes
   - Check available GPU memory

2. **NCCL errors:**
   - Ensure NCCL is properly installed
   - Check GPU connectivity
   - Verify CUDA compatibility

3. **Test failures:**
   - Check environment setup
   - Verify all dependencies are installed
   - Review test output for detailed error messages

4. **Performance issues:**
   - Ensure P2P connectivity is enabled
   - Check GPU drivers are up to date
   - Verify CUDA toolkit compatibility

### Debug Mode
Run tests with verbose output for debugging:
```bash
python run_p2p_tests.py --verbose
```

### Environment Check
Before running tests, check your environment:
```bash
python run_p2p_tests.py --env-check
```

## Contributing

### Adding New Tests
When adding new P2P tests:
1. Follow the existing test structure and naming conventions
2. Use appropriate pytest markers
3. Include both positive and negative test cases
4. Add performance benchmarks for new functionality
5. Ensure tests work in both single and multi-GPU environments

### Test Guidelines
- Use descriptive test names
- Include docstrings for test functions
- Use mocking for external dependencies
- Follow the AAA pattern (Arrange, Act, Assert)
- Include performance assertions where relevant
- Handle edge cases and error conditions

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Include proper error handling
- Add comments for complex logic

## Continuous Integration

The P2P tests are designed to run in CI/CD environments:
- Tests can be run in parallel to reduce execution time
- Environment checks ensure prerequisites are met
- Performance benchmarks detect regressions
- Coverage reports validate test effectiveness

### CI Configuration
For GitHub Actions or other CI systems:
```yaml
name: P2P Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements_test.txt
      - name: Run tests
        run: python run_p2p_tests.py --coverage --parallel
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## References

- [PyTorch Distributed Computing](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/index.html)
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)