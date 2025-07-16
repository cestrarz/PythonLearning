# 07 - Multi-core Parallelization for Data Analysis
# Learn to leverage multiple CPU cores for faster data processing

"""
MULTI-CORE PARALLELIZATION
==========================

Python offers several approaches for parallelization:
1. multiprocessing - True parallelism using multiple processes
2. concurrent.futures - High-level interface for parallel execution
3. joblib - Optimized for NumPy/scientific computing
4. threading - For I/O-bound tasks (limited by GIL for CPU-bound)
5. asyncio - For asynchronous programming

Key Concepts:
- CPU-bound vs I/O-bound tasks
- Global Interpreter Lock (GIL) limitations
- Process vs Thread parallelism
- Shared memory and communication
- Load balancing and overhead considerations

When to Use Parallelization:
✅ Large datasets that can be split
✅ Independent computations
✅ CPU-intensive operations
✅ Embarrassingly parallel problems
❌ Small datasets (overhead > benefit)
❌ Sequential dependencies
❌ Memory-intensive tasks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from joblib import Parallel, delayed
import threading
import asyncio
import aiohttp
import requests
from functools import partial
import psutil
import os

print(f"Number of CPU cores: {mp.cpu_count()}")
print(f"Current process ID: {os.getpid()}")

# =============================================================================
# 1. UNDERSTANDING THE GLOBAL INTERPRETER LOCK (GIL)
# =============================================================================

print("\n" + "="*60)
print("1. UNDERSTANDING THE GLOBAL INTERPRETER LOCK (GIL)")
print("="*60)

def cpu_intensive_task(n):
    """CPU-intensive task to demonstrate GIL impact"""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

def io_intensive_task(duration):
    """I/O-intensive task (simulated with sleep)"""
    time.sleep(duration)
    return f"Task completed after {duration} seconds"

# Measure sequential execution
print("Sequential execution:")
start_time = time.time()
results_sequential = []
for i in range(4):
    results_sequential.append(cpu_intensive_task(1000000))
sequential_time = time.time() - start_time
print(f"Sequential CPU tasks: {sequential_time:.2f} seconds")

# Measure threading (limited by GIL for CPU-bound tasks)
print("\nThreading execution (CPU-bound):")
start_time = time.time()
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(cpu_intensive_task, 1000000) for _ in range(4)]
    results_threading = [future.result() for future in futures]
threading_time = time.time() - start_time
print(f"Threading CPU tasks: {threading_time:.2f} seconds")
print(f"Threading speedup: {sequential_time/threading_time:.2f}x")

# Measure multiprocessing (true parallelism)
print("\nMultiprocessing execution (CPU-bound):")
start_time = time.time()
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(cpu_intensive_task, 1000000) for _ in range(4)]
    results_multiprocessing = [future.result() for future in futures]
multiprocessing_time = time.time() - start_time
print(f"Multiprocessing CPU tasks: {multiprocessing_time:.2f} seconds")
print(f"Multiprocessing speedup: {sequential_time/multiprocessing_time:.2f}x")

# Demonstrate threading advantage for I/O-bound tasks
print("\nI/O-bound task comparison:")
start_time = time.time()
for _ in range(4):
    io_intensive_task(0.5)
io_sequential_time = time.time() - start_time
print(f"Sequential I/O tasks: {io_sequential_time:.2f} seconds")

start_time = time.time()
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(io_intensive_task, 0.5) for _ in range(4)]
    results_io_threading = [future.result() for future in futures]
io_threading_time = time.time() - start_time
print(f"Threading I/O tasks: {io_threading_time:.2f} seconds")
print(f"I/O Threading speedup: {io_sequential_time/io_threading_time:.2f}x")

print("✅ GIL impact demonstration completed")

# =============================================================================
# 2. MULTIPROCESSING BASICS
# =============================================================================

print("\n" + "="*60)
print("2. MULTIPROCESSING BASICS")
print("="*60)

def square_number(x):
    """Simple function to square a number"""
    return x ** 2

def process_chunk(chunk):
    """Process a chunk of data"""
    return [x ** 2 for x in chunk]

def worker_info(x):
    """Function that returns worker process information"""
    return {
        'input': x,
        'result': x ** 2,
        'process_id': os.getpid(),
        'process_name': mp.current_process().name
    }

# Basic multiprocessing with Pool
print("Basic multiprocessing with Pool:")
data = list(range(1, 21))
print(f"Input data: {data}")

# Sequential processing
start_time = time.time()
sequential_results = [square_number(x) for x in data]
sequential_time = time.time() - start_time

# Parallel processing
start_time = time.time()
with mp.Pool(processes=4) as pool:
    parallel_results = pool.map(square_number, data)
parallel_time = time.time() - start_time

print(f"Sequential results: {sequential_results}")
print(f"Parallel results: {parallel_results}")
print(f"Sequential time: {sequential_time:.4f} seconds")
print(f"Parallel time: {parallel_time:.4f} seconds")
print(f"Speedup: {sequential_time/parallel_time:.2f}x")

# Demonstrate process information
print(f"\nWorker process information:")
with mp.Pool(processes=4) as pool:
    worker_results = pool.map(worker_info, range(8))

for result in worker_results[:8]:  # Show first 8 results
    print(f"Input: {result['input']}, Result: {result['result']}, "
          f"PID: {result['process_id']}, Name: {result['process_name']}")

print("✅ Multiprocessing basics completed")

# =============================================================================
# 3. CONCURRENT.FUTURES - HIGH-LEVEL INTERFACE
# =============================================================================

print("\n" + "="*60)
print("3. CONCURRENT.FUTURES - HIGH-LEVEL INTERFACE")
print("="*60)

def analyze_data_chunk(chunk_data):
    """Analyze a chunk of data and return statistics"""
    chunk_array = np.array(chunk_data)
    return {
        'mean': np.mean(chunk_array),
        'std': np.std(chunk_array),
        'min': np.min(chunk_array),
        'max': np.max(chunk_array),
        'size': len(chunk_array)
    }

# Generate large dataset
np.random.seed(42)
large_dataset = np.random.normal(100, 15, 100000)

# Split into chunks
chunk_size = 10000
chunks = [large_dataset[i:i+chunk_size] for i in range(0, len(large_dataset), chunk_size)]

print(f"Dataset size: {len(large_dataset)}")
print(f"Number of chunks: {len(chunks)}")
print(f"Chunk size: {chunk_size}")

# Sequential processing
print("\nSequential processing:")
start_time = time.time()
sequential_stats = [analyze_data_chunk(chunk) for chunk in chunks]
sequential_time = time.time() - start_time
print(f"Sequential time: {sequential_time:.3f} seconds")

# ProcessPoolExecutor
print("\nProcessPoolExecutor:")
start_time = time.time()
with ProcessPoolExecutor(max_workers=4) as executor:
    parallel_stats = list(executor.map(analyze_data_chunk, chunks))
parallel_time = time.time() - start_time
print(f"Parallel time: {parallel_time:.3f} seconds")
print(f"Speedup: {sequential_time/parallel_time:.2f}x")

# Using submit() for more control
print("\nUsing submit() for individual task control:")
start_time = time.time()
with ProcessPoolExecutor(max_workers=4) as executor:
    # Submit all tasks
    futures = [executor.submit(analyze_data_chunk, chunk) for chunk in chunks]
    
    # Collect results as they complete
    submit_stats = []
    for future in futures:
        submit_stats.append(future.result())

submit_time = time.time() - start_time
print(f"Submit method time: {submit_time:.3f} seconds")

# Combine results
combined_stats = {
    'total_mean': np.mean([stat['mean'] for stat in parallel_stats]),
    'total_std': np.mean([stat['std'] for stat in parallel_stats]),
    'global_min': min([stat['min'] for stat in parallel_stats]),
    'global_max': max([stat['max'] for stat in parallel_stats]),
    'total_size': sum([stat['size'] for stat in parallel_stats])
}

print(f"\nCombined statistics:")
for key, value in combined_stats.items():
    print(f"  {key}: {value:.2f}")

print("✅ Concurrent.futures demonstration completed")

# =============================================================================
# 4. JOBLIB - OPTIMIZED FOR SCIENTIFIC COMPUTING
# =============================================================================

print("\n" + "="*60)
print("4. JOBLIB - OPTIMIZED FOR SCIENTIFIC COMPUTING")
print("="*60)

def compute_statistics(data_chunk):
    """Compute comprehensive statistics for a data chunk"""
    return {
        'mean': np.mean(data_chunk),
        'median': np.median(data_chunk),
        'std': np.std(data_chunk),
        'percentile_25': np.percentile(data_chunk, 25),
        'percentile_75': np.percentile(data_chunk, 75),
        'skewness': float(pd.Series(data_chunk).skew()),
        'kurtosis': float(pd.Series(data_chunk).kurtosis())
    }

def monte_carlo_pi(n_samples):
    """Estimate π using Monte Carlo method"""
    np.random.seed()  # Ensure different seeds in different processes
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    inside_circle = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.sum(inside_circle) / n_samples
    return pi_estimate

# Joblib with different backends
print("Joblib with different backends:")

# Generate data
data_chunks = [np.random.normal(100, 15, 50000) for _ in range(8)]

# Sequential
start_time = time.time()
sequential_results = [compute_statistics(chunk) for chunk in data_chunks]
sequential_time = time.time() - start_time
print(f"Sequential: {sequential_time:.3f} seconds")

# Joblib with multiprocessing backend
start_time = time.time()
joblib_results = Parallel(n_jobs=4, backend='multiprocessing')(
    delayed(compute_statistics)(chunk) for chunk in data_chunks
)
joblib_time = time.time() - start_time
print(f"Joblib (multiprocessing): {joblib_time:.3f} seconds")
print(f"Speedup: {sequential_time/joblib_time:.2f}x")

# Joblib with threading backend (for comparison)
start_time = time.time()
joblib_threading_results = Parallel(n_jobs=4, backend='threading')(
    delayed(compute_statistics)(chunk) for chunk in data_chunks
)
joblib_threading_time = time.time() - start_time
print(f"Joblib (threading): {joblib_threading_time:.3f} seconds")

# Monte Carlo π estimation
print(f"\nMonte Carlo π estimation:")
n_experiments = 8
samples_per_experiment = 1000000

# Sequential
start_time = time.time()
sequential_pi = [monte_carlo_pi(samples_per_experiment) for _ in range(n_experiments)]
sequential_pi_time = time.time() - start_time
sequential_pi_mean = np.mean(sequential_pi)

# Parallel
start_time = time.time()
parallel_pi = Parallel(n_jobs=4)(
    delayed(monte_carlo_pi)(samples_per_experiment) for _ in range(n_experiments)
)
parallel_pi_time = time.time() - start_time
parallel_pi_mean = np.mean(parallel_pi)

print(f"Sequential π estimate: {sequential_pi_mean:.6f} (time: {sequential_pi_time:.3f}s)")
print(f"Parallel π estimate: {parallel_pi_mean:.6f} (time: {parallel_pi_time:.3f}s)")
print(f"Actual π: {np.pi:.6f}")
print(f"Speedup: {sequential_pi_time/parallel_pi_time:.2f}x")

print("✅ Joblib demonstration completed")

# =============================================================================
# 5. PANDAS PARALLEL PROCESSING
# =============================================================================

print("\n" + "="*60)
print("5. PANDAS PARALLEL PROCESSING")
print("="*60)

def process_dataframe_chunk(df_chunk):
    """Process a chunk of DataFrame"""
    # Simulate some complex processing
    df_chunk = df_chunk.copy()
    df_chunk['value_squared'] = df_chunk['value'] ** 2
    df_chunk['value_log'] = np.log(df_chunk['value'] + 1)
    df_chunk['rolling_mean'] = df_chunk['value'].rolling(window=min(10, len(df_chunk))).mean()
    df_chunk['category_encoded'] = pd.Categorical(df_chunk['category']).codes
    return df_chunk

def parallel_apply(df, func, n_cores=4):
    """Apply function to DataFrame in parallel"""
    # Split DataFrame into chunks
    chunk_size = len(df) // n_cores
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Process chunks in parallel
    processed_chunks = Parallel(n_jobs=n_cores)(
        delayed(func)(chunk) for chunk in chunks
    )
    
    # Combine results
    return pd.concat(processed_chunks, ignore_index=True)

# Create large DataFrame
np.random.seed(42)
n_rows = 100000
large_df = pd.DataFrame({
    'id': range(n_rows),
    'value': np.random.normal(100, 15, n_rows),
    'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
    'date': pd.date_range('2023-01-01', periods=n_rows, freq='1min')
})

print(f"DataFrame shape: {large_df.shape}")
print("DataFrame sample:")
print(large_df.head())

# Sequential processing
print(f"\nSequential processing:")
start_time = time.time()
sequential_df = process_dataframe_chunk(large_df)
sequential_df_time = time.time() - start_time
print(f"Sequential time: {sequential_df_time:.3f} seconds")

# Parallel processing
print(f"\nParallel processing:")
start_time = time.time()
parallel_df = parallel_apply(large_df, process_dataframe_chunk, n_cores=4)
parallel_df_time = time.time() - start_time
print(f"Parallel time: {parallel_df_time:.3f} seconds")
print(f"Speedup: {sequential_df_time/parallel_df_time:.2f}x")

# Verify results are similar
print(f"\nResult verification:")
print(f"Sequential result shape: {sequential_df.shape}")
print(f"Parallel result shape: {parallel_df.shape}")
print(f"Results are equal: {np.allclose(sequential_df['value_squared'].values, 
                                        parallel_df['value_squared'].values, rtol=1e-10)}")

print("✅ Pandas parallel processing completed")

# =============================================================================
# 6. MEMORY-EFFICIENT PARALLEL PROCESSING
# =============================================================================

print("\n" + "="*60)
print("6. MEMORY-EFFICIENT PARALLEL PROCESSING")
print("="*60)

def memory_efficient_processor(file_chunk_info):
    """Process data chunk efficiently without loading everything into memory"""
    start_idx, end_idx, data_source = file_chunk_info
    
    # Simulate reading chunk from file/database
    chunk_data = data_source[start_idx:end_idx]
    
    # Process chunk
    processed_data = {
        'chunk_start': start_idx,
        'chunk_end': end_idx,
        'chunk_size': end_idx - start_idx,
        'mean': np.mean(chunk_data),
        'std': np.std(chunk_data),
        'sum': np.sum(chunk_data)
    }
    
    return processed_data

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Create large dataset (simulating file/database)
print("Creating large dataset...")
initial_memory = get_memory_usage()
large_data_source = np.random.normal(0, 1, 1000000)  # 1M data points
after_creation_memory = get_memory_usage()
print(f"Memory usage - Initial: {initial_memory:.1f} MB, After creation: {after_creation_memory:.1f} MB")

# Define chunks without loading all data
chunk_size = 100000
n_chunks = len(large_data_source) // chunk_size
chunk_info = [(i * chunk_size, min((i + 1) * chunk_size, len(large_data_source)), large_data_source) 
              for i in range(n_chunks)]

print(f"Processing {len(large_data_source)} data points in {n_chunks} chunks")

# Process chunks in parallel
start_time = time.time()
before_processing_memory = get_memory_usage()

chunk_results = Parallel(n_jobs=4)(
    delayed(memory_efficient_processor)(chunk_info) for chunk_info in chunk_info
)

after_processing_memory = get_memory_usage()
processing_time = time.time() - start_time

print(f"Processing time: {processing_time:.3f} seconds")
print(f"Memory usage - Before processing: {before_processing_memory:.1f} MB, "
      f"After processing: {after_processing_memory:.1f} MB")

# Combine results
total_sum = sum(result['sum'] for result in chunk_results)
total_count = sum(result['chunk_size'] for result in chunk_results)
overall_mean = total_sum / total_count

print(f"\nProcessing results:")
print(f"Total data points processed: {total_count}")
print(f"Overall mean: {overall_mean:.6f}")
print(f"Expected mean (should be ~0): {np.mean(large_data_source):.6f}")

print("✅ Memory-efficient parallel processing completed")

# =============================================================================
# 7. ASYNCHRONOUS PROGRAMMING FOR I/O-BOUND TASKS
# =============================================================================

print("\n" + "="*60)
print("7. ASYNCHRONOUS PROGRAMMING FOR I/O-BOUND TASKS")
print("="*60)

async def fetch_data_async(session, url, delay=1):
    """Simulate fetching data from an API"""
    await asyncio.sleep(delay)  # Simulate network delay
    return {
        'url': url,
        'status': 'success',
        'data_size': np.random.randint(100, 1000),
        'timestamp': time.time()
    }

def fetch_data_sync(url, delay=1):
    """Synchronous version for comparison"""
    time.sleep(delay)  # Simulate network delay
    return {
        'url': url,
        'status': 'success',
        'data_size': np.random.randint(100, 1000),
        'timestamp': time.time()
    }

async def async_data_fetching():
    """Demonstrate asynchronous data fetching"""
    urls = [f'https://api.example.com/data/{i}' for i in range(10)]
    
    # Asynchronous fetching
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_data_async(session, url, delay=0.5) for url in urls]
        async_results = await asyncio.gather(*tasks)
    async_time = time.time() - start_time
    
    return async_results, async_time

# Synchronous fetching for comparison
def sync_data_fetching():
    """Synchronous data fetching for comparison"""
    urls = [f'https://api.example.com/data/{i}' for i in range(10)]
    
    start_time = time.time()
    sync_results = [fetch_data_sync(url, delay=0.5) for url in urls]
    sync_time = time.time() - start_time
    
    return sync_results, sync_time

print("Comparing synchronous vs asynchronous I/O:")

# Run synchronous version
sync_results, sync_time = sync_data_fetching()
print(f"Synchronous fetching: {sync_time:.2f} seconds")

# Run asynchronous version
try:
    # Note: In Jupyter notebooks, you might need to use nest_asyncio
    async_results, async_time = asyncio.run(async_data_fetching())
    print(f"Asynchronous fetching: {async_time:.2f} seconds")
    print(f"Async speedup: {sync_time/async_time:.2f}x")
except Exception as e:
    print(f"Async execution failed (common in some environments): {e}")
    print("Async would typically be much faster for I/O-bound tasks")

print("✅ Asynchronous programming demonstration completed")

# =============================================================================
# 8. PRACTICAL EXAMPLE: PARALLEL DATA ANALYSIS PIPELINE
# =============================================================================

print("\n" + "="*60)
print("8. PRACTICAL EXAMPLE: PARALLEL DATA ANALYSIS PIPELINE")
print("="*60)

def generate_sample_data(n_samples, seed):
    """Generate sample data for analysis"""
    np.random.seed(seed)
    return pd.DataFrame({
        'customer_id': range(n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'spending': np.random.normal(2000, 500, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    })

def analyze_customer_segment(data_chunk):
    """Analyze a segment of customer data"""
    analysis = {
        'segment_size': len(data_chunk),
        'avg_age': data_chunk['age'].mean(),
        'avg_income': data_chunk['income'].mean(),
        'avg_spending': data_chunk['spending'].mean(),
        'spending_income_ratio': data_chunk['spending'].sum() / data_chunk['income'].sum(),
        'category_distribution': data_chunk['category'].value_counts().to_dict(),
        'region_distribution': data_chunk['region'].value_counts().to_dict(),
        'high_spenders': len(data_chunk[data_chunk['spending'] > data_chunk['spending'].quantile(0.8)]),
        'correlation_age_spending': data_chunk['age'].corr(data_chunk['spending'])
    }
    return analysis

def parallel_customer_analysis(n_customers=100000, n_segments=8):
    """Complete parallel customer analysis pipeline"""
    
    print(f"Generating {n_customers} customer records...")
    start_time = time.time()
    
    # Generate data in parallel
    segment_size = n_customers // n_segments
    data_generation_tasks = [
        (segment_size, i) for i in range(n_segments)
    ]
    
    # Generate data segments in parallel
    data_segments = Parallel(n_jobs=4)(
        delayed(generate_sample_data)(size, seed) for size, seed in data_generation_tasks
    )
    
    # Combine all data
    full_dataset = pd.concat(data_segments, ignore_index=True)
    generation_time = time.time() - start_time
    
    print(f"Data generation completed in {generation_time:.2f} seconds")
    print(f"Dataset shape: {full_dataset.shape}")
    
    # Analyze data in parallel
    print(f"\nAnalyzing data in {n_segments} parallel segments...")
    start_time = time.time()
    
    # Split data for analysis
    analysis_chunks = np.array_split(full_dataset, n_segments)
    
    # Parallel analysis
    segment_analyses = Parallel(n_jobs=4)(
        delayed(analyze_customer_segment)(chunk) for chunk in analysis_chunks
    )
    
    analysis_time = time.time() - start_time
    print(f"Analysis completed in {analysis_time:.2f} seconds")
    
    # Combine results
    combined_analysis = {
        'total_customers': sum(analysis['segment_size'] for analysis in segment_analyses),
        'overall_avg_age': np.mean([analysis['avg_age'] for analysis in segment_analyses]),
        'overall_avg_income': np.mean([analysis['avg_income'] for analysis in segment_analyses]),
        'overall_avg_spending': np.mean([analysis['avg_spending'] for analysis in segment_analyses]),
        'total_high_spenders': sum(analysis['high_spenders'] for analysis in segment_analyses),
        'avg_correlation_age_spending': np.mean([analysis['correlation_age_spending'] 
                                               for analysis in segment_analyses if not np.isnan(analysis['correlation_age_spending'])])
    }
    
    return full_dataset, segment_analyses, combined_analysis

# Run the parallel analysis pipeline
dataset, segment_results, overall_results = parallel_customer_analysis()

print(f"\nOVERALL ANALYSIS RESULTS:")
print("="*40)
for key, value in overall_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Parallel Customer Analysis Results', fontsize=16, fontweight='bold')

# Age distribution
axes[0, 0].hist(dataset['age'], bins=30, alpha=0.7, color='skyblue')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# Income vs Spending
axes[0, 1].scatter(dataset['income'], dataset['spending'], alpha=0.5, s=1)
axes[0, 1].set_title('Income vs Spending')
axes[0, 1].set_xlabel('Income')
axes[0, 1].set_ylabel('Spending')

# Category distribution
category_counts = dataset['category'].value_counts()
axes[1, 0].bar(category_counts.index, category_counts.values)
axes[1, 0].set_title('Category Distribution')
axes[1, 0].set_xlabel('Category')
axes[1, 0].set_ylabel('Count')

# Region distribution
region_counts = dataset['region'].value_counts()
axes[1, 1].pie(region_counts.values, labels=region_counts.index, autopct='%1.1f%%')
axes[1, 1].set_title('Region Distribution')

plt.tight_layout()
plt.show()

# Performance comparison
print(f"\nPERFORMANCE COMPARISON:")
print("="*40)

# Sequential analysis for comparison
print("Running sequential analysis for comparison...")
start_time = time.time()
sequential_analysis = analyze_customer_segment(dataset)
sequential_time = time.time() - start_time

print(f"Sequential analysis time: {sequential_time:.3f} seconds")
print(f"Parallel analysis time: {analysis_time:.3f} seconds")
print(f"Speedup: {sequential_time/analysis_time:.2f}x")

print("✅ Parallel data analysis pipeline completed")

# =============================================================================
# 9. BEST PRACTICES AND OPTIMIZATION TIPS
# =============================================================================

print("\n" + "="*60)
print("9. BEST PRACTICES AND OPTIMIZATION TIPS")
print("="*60)

def benchmark_parallel_methods(data_size=100000, n_workers_list=[1, 2, 4, 8]):
    """Benchmark different parallelization methods"""
    
    def compute_heavy_task(chunk):
        """Computationally heavy task for benchmarking"""
        result = 0
        for x in chunk:
            result += np.sin(x) * np.cos(x) * np.exp(-x/1000)
        return result
    
    # Generate test data
    test_data = np.random.random(data_size)
    
    results = {}
    
    for n_workers in n_workers_list:
        if n_workers == 1:
            # Sequential
            start_time = time.time()
            sequential_result = compute_heavy_task(test_data)
            sequential_time = time.time() - start_time
            results[f'Sequential'] = sequential_time
        else:
            # Split data into chunks
            chunk_size = len(test_data) // n_workers
            chunks = [test_data[i:i+chunk_size] for i in range(0, len(test_data), chunk_size)]
            
            # Multiprocessing
            start_time = time.time()
            with mp.Pool(n_workers) as pool:
                mp_results = pool.map(compute_heavy_task, chunks)
            mp_time = time.time() - start_time
            results[f'Multiprocessing_{n_workers}'] = mp_time
            
            # Joblib
            start_time = time.time()
            joblib_results = Parallel(n_jobs=n_workers)(
                delayed(compute_heavy_task)(chunk) for chunk in chunks
            )
            joblib_time = time.time() - start_time
            results[f'Joblib_{n_workers}'] = joblib_time
    
    return results

# Run benchmark
print("Benchmarking different parallelization methods...")
benchmark_results = benchmark_parallel_methods(data_size=50000, n_workers_list=[1, 2, 4])

print(f"\nBenchmark Results (seconds):")
print("-" * 30)
for method, time_taken in benchmark_results.items():
    print(f"{method:<20}: {time_taken:.3f}")

# Calculate speedups
sequential_time = benchmark_results['Sequential']
print(f"\nSpeedups compared to sequential:")
print("-" * 35)
for method, time_taken in benchmark_results.items():
    if method != 'Sequential':
        speedup = sequential_time / time_taken
        print(f"{method:<20}: {speedup:.2f}x")

# Best practices summary
print(f"\nBEST PRACTICES SUMMARY:")
print("="*50)
best_practices = [
    "✅ Use multiprocessing for CPU-bound tasks",
    "✅ Use threading for I/O-bound tasks",
    "✅ Consider overhead - don't parallelize small tasks",
    "✅ Choose optimal number of workers (usually = CPU cores)",
    "✅ Use joblib for NumPy/scientific computing",
    "✅ Profile your code to identify bottlenecks",
    "✅ Consider memory usage and data transfer costs",
    "✅ Use chunking for large datasets",
    "✅ Handle exceptions properly in parallel code",
    "✅ Use async/await for I/O-heavy operations"
]

for practice in best_practices:
    print(practice)

print(f"\nCOMMON PITFALLS TO AVOID:")
print("="*50)
pitfalls = [
    "❌ Parallelizing tasks that are too small",
    "❌ Using too many workers (more than CPU cores for CPU-bound)",
    "❌ Ignoring the GIL for CPU-bound threading",
    "❌ Not handling shared state properly",
    "❌ Forgetting to close pools and clean up resources",
    "❌ Not considering memory overhead of multiple processes",
    "❌ Parallelizing already optimized NumPy operations",
    "❌ Not profiling to verify actual speedup"
]

for pitfall in pitfalls:
    print(pitfall)

print("✅ Best practices and optimization tips completed")

print("\n" + "="*70)
print("MULTI-CORE PARALLELIZATION SUMMARY")
print("="*70)
print("✅ Understanding GIL and when to use multiprocessing vs threading")
print("✅ Basic multiprocessing with Pool and Process")
print("✅ High-level concurrent.futures interface")
print("✅ Joblib for scientific computing optimization")
print("✅ Pandas parallel processing techniques")
print("✅ Memory-efficient parallel processing")
print("✅ Asynchronous programming for I/O-bound tasks")
print("✅ Complete parallel data analysis pipeline")
print("✅ Performance benchmarking and best practices")
print("\nCongratulations! You've completed the data analysis learning path!")
print("="*70)