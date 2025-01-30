#!/usr/bin/env python3
"""
visualize_activations.py

A script to:
  - Parse and visualize PyTorch Profiler logs (.txt)
  - Read a JSON file (e.g., unified_activation_results.json) of final metrics
  - Plot training, validation, and inference metrics
  - (Optionally) Parse TensorBoard logs for line charts of train_loss, val_loss, etc.

Adjust paths, filenames, and column references to match your environment.
"""

import os
import re
import json
import logging
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorBoard log parsing
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ------------------------------------------------------------------------------------
# 1) Find and Aggregate Profiler Reports
# ------------------------------------------------------------------------------------
def find_profiler_files(log_dir, file_extension='*.txt'):
    """
    Recursively find all profiler files (default: .txt) in log_dir.
    """
    pattern = os.path.join(log_dir, '**', file_extension)
    return glob(pattern, recursive=True)

def parse_profiler_report(profiler_report):
    """
    Parse a single profiler report (text) to extract rows with relevant actions.
    
    This is *heavily format-dependent*. Adjust if your profiler text differs.
    """
    relevant_actions_keywords = [
        'training_step',
        'validation_step',
        'optimizer_step',
        'backward'
    ]
    parsed_data = []
    lines = profiler_report.strip().split('\n')
    
    # A sample regex capturing table rows like:
    #  |  training_step    |     0.00123  |  20   | 0.0246 |  2.3%  |
    row_pattern = re.compile(r'^\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|$')
    
    for line in lines:
        match = row_pattern.match(line)
        if match:
            action, mean_duration, num_calls, total_time, percentage = match.groups()
            action = action.strip().lower()
            if action == 'total':
                continue
            # Check if it's an action we're interested in
            if any(keyword in action for keyword in relevant_actions_keywords):
                # Convert numeric fields carefully
                def parse_float(value):
                    try:
                        return float(value.replace(',', ''))
                    except ValueError:
                        return None
                def parse_int(value):
                    try:
                        return int(value.replace(',', ''))
                    except ValueError:
                        return None
                mean_duration = parse_float(mean_duration.strip())  # s
                num_calls = parse_int(num_calls.strip())
                total_time = parse_float(total_time.strip())        # s
                percentage = parse_float(percentage.strip().replace('%', ''))  # %
                parsed_data.append({
                    'Action': action,
                    'Mean Duration (s)': mean_duration,
                    'Num Calls': num_calls,
                    'Total Time (s)': total_time,
                    'Percentage (%)': percentage
                })
    return parsed_data

def map_action_names(df):
    """
    Convert raw action names to a more friendly label.
    """
    action_mapping = {
        'training_step': 'Training Step',
        'validation_step': 'Validation Step',
        'optimizer_step': 'Optimizer Step',
        'backward': 'Backward Pass',
    }

    def simplify_action(a):
        for k, v in action_mapping.items():
            if k in a:  # e.g. 'training_step'
                return v
        return 'Other Action'
    
    df['Simple Action'] = df['Action'].apply(simplify_action)
    return df

def aggregate_profiler_data(log_dir):
    """
    Find all profiler .txt files in log_dir, parse them, and aggregate into a DataFrame.
    """
    profiler_files = find_profiler_files(log_dir, '*.txt')
    logging.info(f"Found {len(profiler_files)} profiler file(s) in {log_dir}.")

    all_parsed = []
    for file_path in profiler_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            data = parse_profiler_report(content)
            # Tag each record with the source filename
            base_name = os.path.basename(file_path)
            for d in data:
                d['Source File'] = base_name
            all_parsed.extend(data)
        except Exception as e:
            logging.error(f"Error parsing {file_path}: {e}")

    if not all_parsed:
        return pd.DataFrame()  # no data

    df = pd.DataFrame(all_parsed)
    df = map_action_names(df)

    # Example groupby: summing or averaging times across repeated lines
    df = df.groupby(['Simple Action', 'Source File'], as_index=False).agg({
        'Mean Duration (s)': 'mean',
        'Total Time (s)': 'sum',
        'Num Calls': 'sum',
        'Percentage (%)': 'mean'
    })

    return df

def create_comparison_barplot(data, metric, output_dir):
    """
    Plot the given metric by 'Simple Action' and hue='Source File'.
    """
    plt.figure(figsize=(16, 9))
    sns.barplot(
        data=data,
        x='Simple Action', y=metric,
        hue='Source File',
        palette='viridis',
        edgecolor='white'
    )
    plt.title(f'{metric} by Action - Across Profiler Reports', fontsize=16)
    plt.xlabel('Action', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Source File', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    safe_metric = metric.lower().replace(' ', '_').replace('(', '').replace(')', '')
    out_path = os.path.join(output_dir, f'profiler_{safe_metric}_barplot.png')
    plt.savefig(out_path)
    plt.close()
    logging.info(f"Saved profiler barplot to {out_path}")

# ------------------------------------------------------------------------------------
# 2) Parse & Plot the JSON Results (from unified_activation_results.json or similar)
# ------------------------------------------------------------------------------------
def parse_profiling_results(json_file):
    """
    Load a results JSON that has columns like:
       - Activation
       - Optimizer Setup
       - Precision
       - Validation Loss
       - Validation Accuracy
       - Training Time (s)
       - Average Inference Time (ms)
    Adjust if your data structure differs.
    """
    if not os.path.exists(json_file):
        logging.warning(f"No results file at: {json_file}")
        return pd.DataFrame()
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        logging.error(f"Error loading {json_file}: {e}")
        return pd.DataFrame()

def plot_inference_speed(df, output_dir):
    """
    Plot the Average Inference Time (ms) vs. Activation (or any relevant dimension).
    Adjust x/y/hue based on how your DataFrame is structured.
    """
    if 'Average Inference Time (ms)' not in df.columns:
        logging.warning("No 'Average Inference Time (ms)' column found in DataFrame.")
        return

    plt.figure(figsize=(10, 6))
    # Example: x='Activation', y='Average Inference Time (ms)', hue='Optimizer Setup'
    # Adjust if you want a different grouping
    sns.barplot(
        data=df,
        x='Activation',
        y='Average Inference Time (ms)',
        hue='Optimizer Setup' if 'Optimizer Setup' in df.columns else None,
        palette='magma',
        edgecolor='white'
    )
    plt.title('Average Inference Time per Batch', fontsize=16)
    plt.xlabel('Activation', fontsize=14)
    plt.ylabel('Avg Inference Time (ms)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'avg_inference_time.png')
    plt.savefig(out_path)
    plt.close()
    logging.info(f"Saved inference speed plot to {out_path}")

def plot_validation_metrics(df, output_dir):
    """
    Plot validation metrics, e.g. Validation Loss & Validation Accuracy, 
    grouped by Activation or Optimizer Setup.
    """
    if 'Validation Loss' not in df.columns or 'Validation Accuracy' not in df.columns:
        logging.warning("No validation metrics to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Validation Loss
    sns.barplot(
        data=df,
        x='Activation',
        y='Validation Loss',
        hue='Optimizer Setup' if 'Optimizer Setup' in df.columns else None,
        ax=axes[0],
        palette='coolwarm',
        edgecolor='white'
    )
    axes[0].set_title('Validation Loss', fontsize=16)
    axes[0].set_xlabel('Activation', fontsize=14)
    axes[0].set_ylabel('Val Loss', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)

    # Plot Validation Accuracy
    sns.barplot(
        data=df,
        x='Activation',
        y='Validation Accuracy',
        hue='Optimizer Setup' if 'Optimizer Setup' in df.columns else None,
        ax=axes[1],
        palette='viridis',
        edgecolor='white'
    )
    axes[1].set_title('Validation Accuracy', fontsize=16)
    axes[1].set_xlabel('Activation', fontsize=14)
    axes[1].set_ylabel('Val Accuracy', fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'validation_metrics.png')
    plt.savefig(out_path)
    plt.close(fig)
    logging.info(f"Saved validation metrics plot to {out_path}")

def plot_training_time(df, output_dir):
    """
    Plot training time by Activation or Optimizer Setup.
    """
    if 'Training Time (s)' not in df.columns:
        logging.warning("No 'Training Time (s)' in DataFrame.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x='Activation',
        y='Training Time (s)',
        hue='Optimizer Setup' if 'Optimizer Setup' in df.columns else None,
        palette='Set2',
        edgecolor='white'
    )
    plt.title('Training Time by Activation', fontsize=16)
    plt.xlabel('Activation', fontsize=14)
    plt.ylabel('Training Time (s)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'training_time.png')
    plt.savefig(out_path)
    plt.close()
    logging.info(f"Saved training time plot to {out_path}")


# ------------------------------------------------------------------------------------
# 3) Parse TensorBoard logs for line charts
# ------------------------------------------------------------------------------------
def plot_line_charts_from_tensorboard(log_dir, output_dir):
    """
    Parse TensorBoard event files from log_dir and plot train_loss, val_loss, train_acc, val_acc over steps.
    Distinguish each run by its directory name.
    """
    event_files = glob(os.path.join(log_dir, '**', 'events.out.tfevents.*'), recursive=True)
    logging.info(f"Found {len(event_files)} TensorBoard event file(s) in {log_dir}.")

    # Store data in { run_name: { metric: (steps, values) } }
    run_data = {}

    for ef in event_files:
        # e.g. path: tb_logs/comparison_RationalCUDA_single_20250130-134500/version_0/events.out.tfevents...
        parts = ef.split(os.sep)
        # Attempt to get the run name from the last 2 or 3 segments
        # e.g. 'comparison_RationalCUDA_single_20250130-134500' from parts[-3]
        run_name = parts[-3]

        # Initialize EventAccumulator
        ea = EventAccumulator(ef)
        try:
            ea.Reload()
        except Exception as e:
            logging.error(f"Failed to load {ef}: {e}")
            continue

        scalar_tags = ea.Tags().get('scalars', [])
        metrics_of_interest = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
        for metric in metrics_of_interest:
            if metric in scalar_tags:
                events = ea.Scalars(metric)
                steps = [ev.step for ev in events]
                values = [ev.value for ev in events]

                if run_name not in run_data:
                    run_data[run_name] = {}
                run_data[run_name][metric] = (steps, values)

    # For each metric, plot a line chart across runs
    for metric in ['train_loss', 'val_loss', 'train_acc', 'val_acc']:
        plt.figure(figsize=(12, 7))
        found_something = False

        for run_name, metrics_dict in run_data.items():
            if metric in metrics_dict:
                steps, values = metrics_dict[metric]
                plt.plot(steps, values, label=run_name)
                found_something = True

        if not found_something:
            logging.warning(f"No data for metric '{metric}' across runs.")
            plt.close()
            continue

        plt.title(f'{metric} over Steps', fontsize=16)
        plt.xlabel('Step', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        out_fn = os.path.join(output_dir, f'{metric}_line_chart.png')
        plt.savefig(out_fn)
        plt.close()
        logging.info(f"Saved TensorBoard line chart for {metric} to {out_fn}")


# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------
def main():
    # Directories
    profiler_log_dir = './log/profiler/'
    tb_log_dir = './tb_logs/'
    output_dir = './visualizations/'
    os.makedirs(output_dir, exist_ok=True)

    # 1) Parse & Plot Profiler Data
    df_profiler = aggregate_profiler_data(profiler_log_dir)
    if not df_profiler.empty:
        logging.info("Profiler data aggregated. Sample:\n")
        logging.info(df_profiler.head())
        
        # Save CSV
        df_profiler.to_csv(os.path.join(output_dir, 'aggregated_profiler_report.csv'), index=False)

        # Create barplots for some metrics
        for metric in ['Mean Duration (s)', 'Total Time (s)']:
            if metric in df_profiler.columns:
                create_comparison_barplot(df_profiler, metric, output_dir)
    else:
        logging.warning("No profiler data found or parsed.")

    # 2) Parse & Plot Results from JSON file
    #    By default, let's look for the JSON file used in the combined script.
    #    Adjust filename if needed (e.g. 'unified_activation_results.json').
    results_json = 'unified_activation_results.json'
    df_results = parse_profiling_results(results_json)
    if not df_results.empty:
        # Save it to CSV for convenience
        df_results.to_csv(os.path.join(output_dir, 'unified_activation_results.csv'), index=False)
        logging.info("Profiling results loaded. Sample:\n")
        logging.info(df_results.head())

        plot_inference_speed(df_results, output_dir)
        plot_validation_metrics(df_results, output_dir)
        plot_training_time(df_results, output_dir)
    else:
        logging.warning(f"No data in {results_json} or file not found.")

    # 3) Parse TensorBoard logs for line charts
    plot_line_charts_from_tensorboard(tb_log_dir, output_dir)


if __name__ == "__main__":
    main()
