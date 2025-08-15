#!/usr/bin/env python3
"""
Performance Measurement Framework for Standalone Dialect Pipeline

This script measures the performance impact of different pipeline stages:
1. Baseline (no optimizations)
2. Auto warp_reduce insertion only
3. Complete automated pipeline
4. Performance comparison and analysis
"""

import subprocess
import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

class PipelinePerformanceMeasurer:
    def __init__(self, build_dir: str = "./build/bin"):
        self.build_dir = Path(build_dir)
        self.standalone_opt = self.build_dir / "standalone-opt"
        self.results = {}
        
    def measure_execution_time(self, command: List[str], iterations: int = 5) -> Dict:
        """Measure execution time of a command over multiple iterations"""
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            try:
                result = subprocess.run(
                    command, 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                end_time = time.time()
                
                if result.returncode == 0:
                    execution_time = (end_time - start_time) * 1000  # Convert to ms
                    times.append(execution_time)
                    print(f"  Iteration {i+1}: {execution_time:.2f} ms")
                else:
                    print(f"  Iteration {i+1}: FAILED - {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print(f"  Iteration {i+1}: TIMEOUT")
            except Exception as e:
                print(f"  Iteration {i+1}: ERROR - {e}")
        
        if not times:
            return {"status": "failed", "error": "No successful executions"}
        
        return {
            "status": "success",
            "iterations": len(times),
            "min_time": min(times),
            "max_time": max(times),
            "mean_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
            "all_times": times
        }
    
    def measure_pipeline_stage(self, stage_name: str, mlir_file: str, pipeline: str) -> Dict:
        """Measure performance of a specific pipeline stage"""
        print(f"\n🔍 Measuring {stage_name}...")
        print(f"  File: {mlir_file}")
        print(f"  Pipeline: {pipeline}")
        
        command = [str(self.standalone_opt), pipeline, mlir_file]
        
        # Measure compilation time
        compilation_result = self.measure_execution_time(command)
        
        # Measure output size (IR complexity)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                output_lines = len(result.stdout.split('\n'))
                output_chars = len(result.stdout)
                compilation_result.update({
                    "output_lines": output_lines,
                    "output_chars": output_chars
                })
        except:
            pass
        
        return compilation_result
    
    def run_performance_analysis(self, test_files: List[str]) -> Dict:
        """Run complete performance analysis across all pipeline stages"""
        print("🚀 Starting Performance Analysis...")
        print("=" * 60)
        
        all_results = {}
        
        for test_file in test_files:
            if not Path(test_file).exists():
                print(f"⚠️  Test file {test_file} not found, skipping...")
                continue
                
            print(f"\n📊 Analyzing: {test_file}")
            print("-" * 40)
            
            file_results = {}
            
            # Stage 1: Baseline (no optimizations)
            baseline_result = self.measure_pipeline_stage(
                "Baseline (No Optimizations)", 
                test_file, 
                ""
            )
            file_results["baseline"] = baseline_result
            
            # Stage 2: Auto warp_reduce insertion only
            warp_reduce_result = self.measure_pipeline_stage(
                "Auto Warp-Reduce Insertion", 
                test_file, 
                "--standalone-warp-reduce-automation"
            )
            file_results["warp_reduce_only"] = warp_reduce_result
            
            # Stage 3: Complete automated pipeline
            complete_result = self.measure_pipeline_stage(
                "Complete Automated Pipeline", 
                test_file, 
                "--standalone-automated-pipeline"
            )
            file_results["complete_pipeline"] = complete_result
            
            all_results[test_file] = file_results
        
        self.results = all_results
        return all_results
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        if not self.results:
            return "No results to report"
        
        report = []
        report.append("🎯 PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        for test_file, file_results in self.results.items():
            report.append(f"📁 Test File: {test_file}")
            report.append("-" * 40)
            
            if "baseline" in file_results and file_results["baseline"]["status"] == "success":
                baseline_time = file_results["baseline"]["mean_time"]
                
                for stage_name, stage_result in file_results.items():
                    if stage_result["status"] != "success":
                        report.append(f"❌ {stage_name}: FAILED")
                        continue
                    
                    stage_time = stage_result["mean_time"]
                    speedup = baseline_time / stage_time if stage_time > 0 else float('inf')
                    
                    report.append(f"✅ {stage_name}:")
                    report.append(f"   Time: {stage_time:.2f} ms")
                    report.append(f"   Speedup: {speedup:.2f}x")
                    
                    if "output_lines" in stage_result:
                        report.append(f"   Output: {stage_result['output_lines']} lines")
                    
                    if stage_name != "baseline":
                        improvement = ((baseline_time - stage_time) / baseline_time) * 100
                        report.append(f"   Improvement: {improvement:+.1f}%")
            
            report.append("")
        
        # Summary statistics
        report.append("📊 SUMMARY STATISTICS")
        report.append("-" * 40)
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() 
                             if any(s["status"] == "success" for s in r.values()))
        
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Successful Tests: {successful_tests}")
        report.append(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        return "\n".join(report)
    
    def save_results(self, output_file: str = "performance_results.json"):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Results saved to {output_file}")

def main():
    """Main performance measurement execution"""
    print("🚀 Standalone Dialect Pipeline Performance Measurement")
    print("=" * 60)
    
    # Initialize measurer
    measurer = PipelinePerformanceMeasurer()
    
    # Test files to analyze
    test_files = [
        "test_simple_auto_warp.mlir",
        "test_comprehensive_pipeline.mlir", 
        "benchmark_pipeline_performance.mlir"
    ]
    
    # Run performance analysis
    results = measurer.run_performance_analysis(test_files)
    
    # Generate and display report
    report = measurer.generate_performance_report()
    print("\n" + report)
    
    # Save results
    measurer.save_results()
    
    print("\n🎉 Performance analysis complete!")
    print("Next step: Real GPU testing with actual execution times")

if __name__ == "__main__":
    main() 