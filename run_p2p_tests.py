#!/usr/bin/env python3
"""
P2P Backend Test Runner

This script provides a convenient way to run comprehensive tests for the P2P backend functionality.
It supports different test suites and provides detailed output and reporting.

Usage:
    python run_p2p_tests.py [options]

Examples:
    # Run all P2P tests
    python run_p2p_tests.py

    # Run only unit tests
    python run_p2p_tests.py --unit

    # Run only integration tests
    python run_p2p_tests.py --integration

    # Run only performance tests
    python run_p2p_tests.py --performance

    # Run only error handling tests
    python run_p2p_tests.py --error

    # Run tests with coverage
    python run_p2p_tests.py --coverage

    # Run tests with benchmarking
    python run_p2p_tests.py --benchmark

    # Run tests in parallel
    python run_p2p_tests.py --parallel

    # Run specific test file
    python run_p2p_tests.py --file tests/test_p2p_backend.py

    # Run tests with specific markers
    python run_p2p_tests.py --markers "p2p,unit"
"""

import argparse
import sys
import os
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any


class P2PTestRunner:
    """Main test runner class for P2P backend tests."""

    def __init__(self):
        self.test_files = [
            "tests/test_p2p_backend.py",
            "tests/test_p2p_integration.py", 
            "tests/test_p2p_performance.py",
            "tests/test_p2p_error_handling.py",
            "tests/test_p2p_backend_selection.py"
        ]
        
        self.test_suites = {
            "unit": ["tests/test_p2p_backend.py"],
            "integration": ["tests/test_p2p_integration.py"],
            "performance": ["tests/test_p2p_performance.py"],
            "error": ["tests/test_p2p_error_handling.py"],
            "auto": ["tests/test_p2p_backend_selection.py"],
            "all": self.test_files
        }

    def run_tests(self, test_files: List[str], options: Dict[str, Any]) -> Dict[str, Any]:
        """Run the specified test files with the given options."""
        results = {
            "success": True,
            "files": {},
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0
            }
        }

        # Build pytest command
        cmd = self._build_pytest_command(options)
        
        # Add test files to command
        cmd.extend(test_files)

        # Run tests for each file
        for test_file in test_files:
            print(f"\n{'='*60}")
            print(f"Running tests in {test_file}")
            print(f"{'='*60}")
            
            file_cmd = cmd.copy()
            file_cmd.append(test_file)
            
            result = self._execute_command(file_cmd, test_file)
            results["files"][test_file] = result
            
            # Update summary
            results["summary"]["total"] += result["total"]
            results["summary"]["passed"] += result["passed"]
            results["summary"]["failed"] += result["failed"]
            results["summary"]["skipped"] += result["skipped"]
            results["summary"]["errors"] += result["errors"]
            
            if result["failed"] > 0 or result["errors"] > 0:
                results["success"] = False

        return results

    def _build_pytest_command(self, options: Dict[str, Any]) -> List[str]:
        """Build the pytest command based on options."""
        cmd = ["pytest"]
        
        # General options
        cmd.extend(["-v", "--tb=short"])
        
        # Coverage options
        if options.get("coverage"):
            cmd.extend([
                "--cov=exllamav3",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov_p2p",
                "--cov-report=xml:coverage_p2p.xml",
                "--cov-fail-under=80"
            ])
        
        # Benchmarking options
        if options.get("benchmark"):
            cmd.extend([
                "--benchmark-only",
                "--benchmark-sort=mean",
                "--benchmark-group-by=param:backend",
                "--benchmark-warmup=on",
                "--benchmark-warmup-iterations=3",
                "--benchmark-timeout=300"
            ])
        
        # Parallel testing
        if options.get("parallel"):
            cmd.extend([
                "--numprocesses=auto",
                "--dist=worksteal",
                "--maxfail=10"
            ])
        
        # Markers
        if options.get("markers"):
            cmd.extend([f"-m {options['markers']}"])
        
        # Additional options
        if options.get("verbose"):
            cmd.append("-vv")
        
        if options.get("stop_on_fail"):
            cmd.append("--maxfail=1")
        
        return cmd

    def _execute_command(self, cmd: List[str], test_file: str) -> Dict[str, Any]:
        """Execute a command and return the result."""
        result = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "output": "",
            "command": " ".join(cmd)
        }
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            # Stream output
            for line in process.stdout:
                print(line, end="")
                result["output"] += line
                
                # Parse test results
                if "passed" in line:
                    result["passed"] += 1
                elif "failed" in line:
                    result["failed"] += 1
                elif "skipped" in line:
                    result["skipped"] += 1
                elif "error" in line:
                    result["errors"] += 1
            
            process.wait()
            result["returncode"] = process.returncode
            
            # If we can't parse from output, use pytest's JSON output
            if result["total"] == 0:
                result = self._parse_pytest_output(result["output"])
            
        except Exception as e:
            result["errors"] = 1
            result["output"] = f"Error running command: {str(e)}"
            result["returncode"] = 1
        
        return result

    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output to extract test results."""
        result = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0
        }
        
        # Simple parsing - look for common patterns
        lines = output.split('\n')
        
        for line in lines:
            if "passed" in line and "failed" not in line:
                result["passed"] += 1
            elif "failed" in line:
                result["failed"] += 1
            elif "skipped" in line:
                result["skipped"] += 1
            elif "error" in line:
                result["errors"] += 1
        
        result["total"] = result["passed"] + result["failed"] + result["skipped"] + result["errors"]
        
        return result

    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of test results."""
        print("\n" + "="*80)
        print("P2P Backend Test Summary")
        print("="*80)
        
        summary = results["summary"]
        print(f"Total Tests: {summary['total']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Errors: {summary['errors']}")
        print(f"Success Rate: {(summary['passed']/summary['total']*100):.1f}%" if summary['total'] > 0 else "N/A")
        
        print("\nFile Results:")
        for file_path, file_result in results["files"].items():
            print(f"  {file_path}: {file_result['passed']} passed, {file_result['failed']} failed, {file_result['errors']} errors")
        
        if results["success"]:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed!")
        
        print("="*80)

    def run_environment_check(self) -> Dict[str, Any]:
        """Check if the test environment is properly configured."""
        print("Checking test environment...")
        
        results = {
            "python_version": False,
            "pytorch_available": False,
            "cuda_available": False,
            "test_dependencies": False,
            "required_files": []
        }
        
        # Check Python version
        try:
            import sys
            if sys.version_info >= (3, 8):
                results["python_version"] = True
                print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} ({sys.version_info.micro})")
            else:
                print(f"✗ Python {sys.version_info.major}.{sys.version_info.minor} is too old (requires 3.8+)")
        except Exception as e:
            print(f"✗ Error checking Python version: {e}")
        
        # Check PyTorch
        try:
            import torch
            results["pytorch_available"] = True
            print(f"✓ PyTorch {torch.__version__}")
            
            # Check CUDA
            if torch.cuda.is_available():
                results["cuda_available"] = True
                print(f"✓ CUDA available, {torch.cuda.device_count()} GPUs")
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                print("⚠ CUDA not available - some tests may be skipped")
        except Exception as e:
            print(f"✗ Error checking PyTorch: {e}")
        
        # Check test dependencies
        try:
            import pytest
            import numpy
            print(f"✓ pytest {pytest.__version__}, numpy {numpy.__version__}")
            results["test_dependencies"] = True
        except Exception as e:
            print(f"✗ Error checking test dependencies: {e}")
        
        # Check required files
        required_files = [
            "tests/test_p2p_backend.py",
            "tests/test_p2p_integration.py",
            "tests/test_p2p_performance.py",
            "tests/test_p2p_error_handling.py",
            "tests/test_p2p_backend_selection.py"
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                results["required_files"].append(file_path)
                print(f"✓ {file_path}")
            else:
                print(f"✗ {file_path} not found")
        
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run P2P Backend Tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--error", action="store_true", help="Run error handling tests only")
    parser.add_argument("--auto", action="store_true", help="Run backend selection tests only")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--markers", type=str, help="Run tests with specific markers")
    parser.add_argument("--file", type=str, help="Run specific test file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop on first failure")
    parser.add_argument("--env-check", action="store_true", help="Check environment only")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = P2PTestRunner()
    
    # Check environment if requested
    if args.env_check:
        env_results = runner.run_environment_check()
        if args.json:
            print(json.dumps(env_results, indent=2))
        return
    
    # Determine which tests to run
    if args.file:
        test_files = [args.file]
    elif args.unit:
        test_files = runner.test_suites["unit"]
    elif args.integration:
        test_files = runner.test_suites["integration"]
    elif args.performance:
        test_files = runner.test_suites["performance"]
    elif args.error:
        test_files = runner.test_suites["error"]
    elif args.auto:
        test_files = runner.test_suites["auto"]
    else:
        test_files = runner.test_suites["all"]
    
    # Build options
    options = {
        "coverage": args.coverage,
        "benchmark": args.benchmark,
        "parallel": args.parallel,
        "markers": args.markers,
        "verbose": args.verbose,
        "stop_on_fail": args.stop_on_fail
    }
    
    # Run tests
    print("Starting P2P Backend Tests...")
    results = runner.run_tests(test_files, options)
    
    # Print summary
    runner.print_summary(results)
    
    # Output JSON if requested
    if args.json:
        print(json.dumps(results, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()