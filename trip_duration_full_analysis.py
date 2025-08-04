import pandas as pd
import time
import threading
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIG ===
DATA_FILE = "train.csv"
RESULT_CSV = "detailed_performance_comparison.csv"
PLOT_FILE = "performance_comparison_plot.png"
FILTER_THRESHOLD = 1000  # seconds

# === FUNCTION DEFINITIONS ===

def load_data():
    """Load dataset and return DataFrame."""
    df = pd.read_csv(DATA_FILE)
    return df[["trip_duration"]]

def process_data(df, process_type):
    """Return result of filtering or sorting."""
    if process_type == "Filter":
        return df[df["trip_duration"] > FILTER_THRESHOLD]
    elif process_type == "Sort":
        return df.sort_values("trip_duration")
    else:
        raise ValueError("Invalid process type")

def benchmark(df, process_type, approach):
    """Run processing with selected approach and return duration."""
    start_time = time.time()

    if approach == "Sequential":
        process_data(df, process_type)

    elif approach == "Thread":
        thread = threading.Thread(target=process_data, args=(df, process_type))
        thread.start()
        thread.join()

    elif approach == "Multiprocessing":
        with multiprocessing.Pool(1) as pool:
            pool.apply(process_data, args=(df, process_type,))

    else:
        raise ValueError("Invalid approach")

    return round(time.time() - start_time, 4)

def run_all_benchmarks():
    """Run benchmarks and save detailed results."""
    df_full = load_data()
    sizes = [0.25, 0.5, 0.75, 1.0]
    approaches = ["Sequential", "Thread", "Multiprocessing"]
    processes = ["Filter", "Sort"]

    results = []

    for size in sizes:
        df_sample = df_full.sample(frac=size, random_state=42)
        for process in processes:
            for approach in approaches:
                time_taken = benchmark(df_sample, process, approach)
                results.append({
                    "Data Size (%)": int(size * 100),
                    "Process": process,
                    "Approach": approach,
                    "Time (s)": time_taken
                })
                print(f"{process}-{approach} ({int(size*100)}%): {time_taken}s")

    # Save results
    pd.DataFrame(results).to_csv(RESULT_CSV, index=False)

def create_performance_plot():
    """Create comparison plot from benchmark CSV."""
    df = pd.read_csv(RESULT_CSV)
    sns.set(style="whitegrid")

    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=df,
        x="Data Size (%)",
        y="Time (s)",
        hue="Approach",
        style="Process",
        markers=True,
        dashes=False
    )

    plt.title("Performance Comparison: Filtering vs Sorting")
    plt.xlabel("Data Size (%)")
    plt.ylabel("Execution Time (s)")
    plt.legend(title="Approach / Process")
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.show()

def show_clean_table():
    """Print pivot table of performance comparison."""
    df = pd.read_csv(RESULT_CSV)
    table = df.pivot_table(
        index=["Data Size (%)", "Process"],
        columns="Approach",
        values="Time (s)"
    ).reset_index()
    print("\n=== Clean Performance Comparison Table ===\n")
    print(table.to_string(index=False))

# === MAIN ===
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows multiprocessing
    if not os.path.exists(RESULT_CSV):
        run_all_benchmarks()
    show_clean_table()
    create_performance_plot()
