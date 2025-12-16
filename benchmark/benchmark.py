"""
Benchmark script for Kuwahara filter implementations.

This script benchmarks multiple Kuwahara filter implementations on sample images,
measures inference speed, and generates visualizations.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ============================================================================
# IMPLEMENTATIONS TO BENCHMARK
# ============================================================================
# Add your implementations here by importing them and adding to IMPLEMENTATIONS dict
#
# Example usage:
#   from implementation.numba import kuwahara_numba
#   from implementation.other_impl import kuwahara_other
#
#   IMPLEMENTATIONS = {
#       "numba": kuwahara_numba,
#       "other": kuwahara_other,
#   }
#
# Note: All implementations should have the signature: func(img: np.ndarray, radius: int) -> np.ndarray

# TODO: Import your implementations here

from implementation.numba import kuwahara_numba

IMPLEMENTATIONS: Dict[str, callable] = {
    "numba": kuwahara_numba,
    # Add more implementations here as you import them
}


# ============================================================================
# CONFIGURATION
# ============================================================================
SAMPLES_DIR = Path(__file__).parent.parent / "samples"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
GRAPHS_DIR = Path(__file__).parent.parent / "graphs"
JSON_DIR = OUTPUTS_DIR / "json"

# Benchmark parameters
RADII = [6, 15]  # Radii to benchmark for Kuwahara filter
NUM_RUNS = 2  # Number of runs per image for averaging


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def load_image(image_path: Path) -> np.ndarray:
    """Load an image and convert it to a numpy array."""
    img = Image.open(image_path)
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Convert to numpy array as float64 (keep original range [0, 255])
    img_array = np.array(img, dtype=np.float64)
    return img_array


def benchmark_implementation(
    impl_func: callable, image: np.ndarray, radius: int, num_runs: int = 2
) -> Tuple[float, float, np.ndarray]:
    """
    Benchmark a single implementation on an image.
    
    Returns:
        Tuple of (mean_time, std_time, result_image) in seconds and numpy array
    """
    times = []
    result_image = None
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        result_image = impl_func(image, radius)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return mean_time, std_time, result_image


def get_sample_images() -> List[Path]:
    """Get all image files from the samples directory."""
    image_extensions = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    samples = []
    
    if SAMPLES_DIR.exists():
        for file_path in SAMPLES_DIR.iterdir():
            if file_path.suffix in image_extensions:
                samples.append(file_path)
    
    return sorted(samples)


def save_output_image(result_image: np.ndarray, output_path: Path):
    """Save the filtered image to disk (I/O operation, not timed)."""
    # Clip values to valid range and convert to uint8
    result_uint8 = np.clip(result_image, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image and save
    result_pil = Image.fromarray(result_uint8)
    result_pil.save(output_path)


# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================
def run_benchmarks() -> Dict:
    """
    Run benchmarks on all implementations, all sample images, and all radii.
    
    Returns:
        Dictionary containing benchmark results
    """
    if not IMPLEMENTATIONS:
        raise ValueError(
            "No implementations found! Please import and add implementations to IMPLEMENTATIONS dict."
        )
    
    sample_images = get_sample_images()
    if not sample_images:
        raise ValueError(f"No sample images found in {SAMPLES_DIR}")
    
    results = {
        "implementations": list(IMPLEMENTATIONS.keys()),
        "samples": [img.name for img in sample_images],
        "radii": RADII,
        "num_runs": NUM_RUNS,
        "results": {},
    }
    
    print(f"Benchmarking {len(IMPLEMENTATIONS)} implementation(s) on {len(sample_images)} sample(s)...")
    print(f"Using radii={RADII}, num_runs={NUM_RUNS}\n")
    
    # Benchmark each implementation
    for impl_name, impl_func in IMPLEMENTATIONS.items():
        print(f"Benchmarking {impl_name}...")
        impl_results = {}
        
        # Create output directory for this implementation
        impl_output_dir = OUTPUTS_DIR / impl_name
        impl_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Benchmark for each radius
        for radius in RADII:
            print(f"  Radius {radius}:")
            radius_results = {
                "times": {},
                "average_time": 0.0,
            }
            
            all_times = []
            
            for sample_path in sample_images:
                print(f"    Processing {sample_path.name}...", end=" ")
                image = load_image(sample_path)
                
                # Benchmark (timing only, no I/O)
                mean_time, std_time, result_image = benchmark_implementation(
                    impl_func, image, radius, NUM_RUNS
                )
                
                # Save output image (after timing, I/O not measured)
                # Include radius in filename
                sample_stem = sample_path.stem
                sample_suffix = sample_path.suffix
                output_filename = f"{sample_stem}_r{radius}{sample_suffix}"
                output_image_path = impl_output_dir / output_filename
                save_output_image(result_image, output_image_path)
                
                radius_results["times"][sample_path.name] = {
                    "mean": mean_time,
                    "std": std_time,
                }
                all_times.append(mean_time)
                print(f"{mean_time:.4f}s ± {std_time:.4f}s")
            
            radius_results["average_time"] = np.mean(all_times)
            impl_results[radius] = radius_results
            
            print(f"    Average time for {impl_name} (radius {radius}): {radius_results['average_time']:.4f}s\n")
        
        results["results"][impl_name] = impl_results
        print(f"  Output images saved to {impl_output_dir}\n")
    
    return results


# ============================================================================
# SAVE RESULTS
# ============================================================================
def save_results(results: Dict):
    """Save benchmark results to JSON and per-implementation files."""
    # Create output directories
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save complete results to JSON
    json_path = JSON_DIR / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved complete results to {json_path}")
    
    # Save per-implementation results
    for impl_name, impl_results in results["results"].items():
        impl_output = {
            "implementation": impl_name,
            "radii": results["radii"],
            "num_runs": results["num_runs"],
            "results": impl_results,
        }
        
        impl_file = OUTPUTS_DIR / f"{impl_name}_results.json"
        with open(impl_file, "w") as f:
            json.dump(impl_output, f, indent=2)
        print(f"Saved {impl_name} results to {impl_file}")


# ============================================================================
# VISUALIZATION
# ============================================================================
def create_bar_chart(results: Dict):
    """Create a bar chart comparing all implementations on all samples for each radius."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    
    implementations = results["implementations"]
    samples = results["samples"]
    radii = results["radii"]
    
    # Calculate total number of bars per sample (implementations * radii)
    num_impls = len(implementations)
    num_radii = len(radii)
    total_bars_per_sample = num_impls * num_radii
    
    # Prepare data for plotting
    x = np.arange(len(samples))
    width = 0.8 / total_bars_per_sample  # Bar width
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create bars for each implementation and radius combination
    bar_idx = 0
    for impl_name in implementations:
        for radius in radii:
            times = [
                results["results"][impl_name][radius]["times"][sample]["mean"]
                for sample in samples
            ]
            avg_time = results["results"][impl_name][radius]["average_time"]
            
            offset = (bar_idx - total_bars_per_sample / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                times,
                width,
                label=f"{impl_name} r={radius} (avg: {avg_time:.4f}s)",
                alpha=0.8,
            )
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}s",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            
            bar_idx += 1
    
    # Customize the plot
    ax.set_xlabel("Sample Images", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("Kuwahara Filter Implementation Benchmark", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45, ha="right")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    # Save the chart
    chart_path = GRAPHS_DIR / "benchmark_results.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    print(f"Saved bar chart to {chart_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main function to run benchmarks."""
    print("=" * 60)
    print("Kuwahara Filter Benchmark")
    print("=" * 60)
    print()
    
    try:
        # Run benchmarks
        results = run_benchmarks()
        
        # Save results
        print("\n" + "=" * 60)
        print("Saving Results")
        print("=" * 60)
        save_results(results)
        
        # Create visualization
        print("\n" + "=" * 60)
        print("Generating Visualization")
        print("=" * 60)
        create_bar_chart(results)
        
        print("\n" + "=" * 60)
        print("Benchmark Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during benchmarking: {e}")
        raise


if __name__ == "__main__":
    main()

