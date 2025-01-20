import pandas as pd
import re
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_profiler_files(log_dir, file_extension='*.txt'):
    """
    Traverse the log_dir and find all files matching the file_extension pattern.
    
    Args:
        log_dir (str): Path to the log directory.
        file_extension (str): Pattern to match profiler report files.
    
    Returns:
        list: List of file paths matching the pattern.
    """
    pattern = os.path.join(log_dir, '**', file_extension)
    return glob(pattern, recursive=True)

def parse_profiler_report(profiler_report):
    """
    Parse a single profiler report and extract relevant data.
    
    Args:
        profiler_report (str): Multiline string of the profiler report.
    
    Returns:
        list of dict: Parsed data containing relevant actions.
    """
    # Define the actions of interest
    relevant_actions_keywords = [
        'training_step',
        'validation_step',
        'optimizer_step',
        'backward'
    ]
    
    # Initialize a list to hold parsed data
    parsed_data = []
    
    # Split the report into lines
    lines = profiler_report.strip().split('\n')
    
    # Regular expression to match the table rows
    # This regex assumes that the columns are separated by '|'
    row_pattern = re.compile(r'^\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|$')
    
    for line in lines:
        # Use regex to parse the line
        match = row_pattern.match(line)
        if match:
            action, mean_duration, num_calls, total_time, percentage = match.groups()
            
            # Skip the 'Total' row or any other non-relevant rows if necessary
            if action.strip().lower() == 'total':
                continue
            
            # Check if the action contains any of the relevant keywords
            if any(keyword in action for keyword in relevant_actions_keywords):
                # Clean and convert the data
                action = action.strip()
                
                # Handle mean_duration, which might be '-' or in scientific notation
                mean_duration = mean_duration.strip()
                if mean_duration == '-':
                    mean_duration = None
                else:
                    try:
                        mean_duration = float(mean_duration.replace(',', ''))
                    except ValueError:
                        mean_duration = None
                
                # Convert num_calls to integer
                num_calls = num_calls.strip()
                try:
                    num_calls = int(num_calls.replace(',', ''))
                except ValueError:
                    num_calls = None
                
                # Convert total_time to float
                total_time = total_time.strip()
                try:
                    total_time = float(total_time.replace(',', ''))
                except ValueError:
                    total_time = None
                
                # Convert percentage to float, removing the '%' sign
                percentage = percentage.strip().replace('%', '').strip()
                try:
                    percentage = float(percentage)
                except ValueError:
                    percentage = None
                
                # Append to the list
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
    Map long action names to simpler identifiers.
    
    Args:
        df (pd.DataFrame): DataFrame containing the 'Action' column.
    
    Returns:
        pd.DataFrame: DataFrame with mapped 'Action' names.
    """
    # Define a mapping dictionary
    action_mapping = {
        'training_step': 'Training Step',
        'validation_step': 'Validation Step',
        'optimizer_step': 'Optimizer Step',
        'backward': 'Backward Pass',
        'transfer_batch_to_device': 'Transfer Batch',
        'batch_to_device': 'Batch Transfer',
        # Add more mappings as needed
    }
    
    # Function to map each action
    def simplify_action(action):
        for key in action_mapping:
            if key in action:
                return action_mapping[key]
        return 'Other Action'  # Default for unmapped actions
    
    # Apply the mapping
    df['Simple Action'] = df['Action'].apply(simplify_action)
    return df

def aggregate_profiler_data(log_dir):
    """
    Aggregate profiler data from all report files in the log directory.
    
    Args:
        log_dir (str): Path to the log directory.
    
    Returns:
        pd.DataFrame: Aggregated DataFrame containing data from all profiler reports.
    """
    profiler_files = find_profiler_files(log_dir)
    logging.info(f"Found {len(profiler_files)} profiler report files.")
    
    all_parsed_data = []
    for file_path in profiler_files:
        logging.info(f"Processing file: {file_path}")
        try:
            with open(file_path, 'r') as file:
                profiler_report = file.read()
                parsed_data = parse_profiler_report(profiler_report)
                # Optionally, add a column for the file name to distinguish data sources
                for entry in parsed_data:
                    entry['Source File'] = os.path.basename(file_path)
                all_parsed_data.extend(parsed_data)
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
    
    if not all_parsed_data:
        logging.warning("No data parsed from profiler reports.")
        return pd.DataFrame()
    
    # Create a DataFrame from the aggregated data
    df = pd.DataFrame(all_parsed_data)
    
    # Map action names to simpler identifiers
    df = map_action_names(df)
    
    # Optional: Remove duplicates by aggregating
    df = df.groupby(['Simple Action', 'Source File'], as_index=False).agg({
        'Mean Duration (s)': 'mean',
        'Total Time (s)': 'sum',
        'Num Calls': 'sum',
        'Percentage (%)': 'mean'
    })
    
    return df

def create_comparison_barplot(data, metric, output_dir):
    """
    Create and save a bar plot comparing the specified metric across actions and source files.
    
    Args:
        data (pd.DataFrame): DataFrame containing the aggregated data.
        metric (str): The metric to visualize (e.g., 'Mean Duration (s)', 'Total Time (s)').
        output_dir (str): Directory where the plot image will be saved.
    """
    plt.figure(figsize=(16, 10))
    sns.barplot(
        data=data,
        x='Simple Action',
        y=metric,
        hue='Source File',
        palette='viridis',
        edgecolor='white',
        linewidth=1
    )
    plt.title(f'{metric} by Action Across Different Profiler Reports', fontsize=18)
    plt.xlabel('Action', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Source File', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a safe filename
    safe_metric = metric.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    output_path = os.path.join(output_dir, f'{safe_metric}_comparison.png')
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved plot to {output_path}")

def parse_profiling_results(json_file):
    """
    Parse the profiling_results.json file to extract inference speed and validation metrics.
    
    Args:
        json_file (str): Path to the profiling_results.json file.
    
    Returns:
        pd.DataFrame: DataFrame containing the profiling results.
    """
    if not os.path.exists(json_file):
        logging.error(f"Profiling results file '{json_file}' does not exist.")
        return pd.DataFrame()
    
    try:
        with open(json_file, 'r') as f:
            results = json.load(f)
        df = pd.DataFrame(results)
        return df
    except Exception as e:
        logging.error(f"Failed to parse '{json_file}': {e}")
        return pd.DataFrame()

def plot_inference_speed(df, output_dir):
    """
    Plot the Average Inference Time (ms) per configuration.
    
    Args:
        df (pd.DataFrame): DataFrame containing profiling results.
        output_dir (str): Directory where the plot image will be saved.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x='Precision',
        y='Average Inference Time (ms)',
        hue='Optimizer Setup',
        palette='magma',
        edgecolor='white',
        linewidth=1
    )
    plt.title('Average Inference Time per Batch by Configuration', fontsize=16)
    plt.xlabel('Precision', fontsize=14)
    plt.ylabel('Average Inference Time (ms)', fontsize=14)
    plt.legend(title='Optimizer Setup', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Create a safe filename
    output_path = os.path.join(output_dir, 'average_inference_time_comparison.png')
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved inference speed plot to {output_path}")

def plot_validation_metrics(df, output_dir):
    """
    Plot Validation Loss and Validation Accuracy per configuration.
    
    Args:
        df (pd.DataFrame): DataFrame containing profiling results.
        output_dir (str): Directory where the plot images will be saved.
    """
    # Create a combined plot with two subplots: Validation Loss and Validation Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot Validation Loss
    sns.barplot(
        data=df,
        x='Precision',
        y='Validation Loss',
        hue='Optimizer Setup',
        palette='coolwarm',
        edgecolor='white',
        linewidth=1,
        ax=axes[0]
    )
    axes[0].set_title('Validation Loss by Configuration', fontsize=16)
    axes[0].set_xlabel('Precision', fontsize=14)
    axes[0].set_ylabel('Validation Loss', fontsize=14)
    axes[0].legend(title='Optimizer Setup', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot Validation Accuracy
    sns.barplot(
        data=df,
        x='Precision',
        y='Validation Accuracy',
        hue='Optimizer Setup',
        palette='viridis',
        edgecolor='white',
        linewidth=1,
        ax=axes[1]
    )
    axes[1].set_title('Validation Accuracy by Configuration', fontsize=16)
    axes[1].set_xlabel('Precision', fontsize=14)
    axes[1].set_ylabel('Validation Accuracy', fontsize=14)
    axes[1].legend(title='Optimizer Setup', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Create safe filenames
    fig.savefig(os.path.join(output_dir, 'validation_metrics_comparison.png'))
    plt.close(fig)
    logging.info(f"Saved validation metrics plot to {os.path.join(output_dir, 'validation_metrics_comparison.png')}")

def plot_training_time(df, output_dir):
    """
    Plot Training Time (s) per configuration.
    
    Args:
        df (pd.DataFrame): DataFrame containing profiling results.
        output_dir (str): Directory where the plot image will be saved.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x='Precision',
        y='Training Time (s)',
        hue='Optimizer Setup',
        palette='Set2',
        edgecolor='white',
        linewidth=1
    )
    plt.title('Training Time by Configuration', fontsize=16)
    plt.xlabel('Precision', fontsize=14)
    plt.ylabel('Training Time (s)', fontsize=14)
    plt.legend(title='Optimizer Setup', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Create a safe filename
    output_path = os.path.join(output_dir, 'training_time_comparison.png')
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved training time plot to {output_path}")

def plot_line_charts_from_tensorboard(log_dir, output_dir):
    """
    Parse TensorBoard logs and plot line charts for train_loss, val_loss, train_acc, and val_acc.
    
    Args:
        log_dir (str): Path to the TensorBoard log directory.
        output_dir (str): Directory where the plot images will be saved.
    """
    # Find all TensorBoard event files
    event_files = glob(os.path.join(log_dir, '**', 'events.out.tfevents.*'), recursive=True)
    logging.info(f"Found {len(event_files)} TensorBoard event files.")
    
    # Initialize a dictionary to hold data per configuration
    config_data = {}
    
    for event_file in event_files:
        # Extract configuration from the path
        # Example path: activations/cuda_torch/tb_logs/vit_profiling_Separate_Optimizers_16-mixed_20241205-132706/version_0/events.out.tfevents.1733405226.8ec8fd63d7a9.397749.1.
        path_parts = event_file.split(os.sep)
        try:
            # Assuming the log directory structure is consistent
            # Extract the configuration name from the parent directory
            # e.g., 'vit_profiling_Separate_Optimizers_16-mixed_20241205-132706'
            config_dir = path_parts[-3]
            config_name = config_dir  # Or parse further if needed
            
            if config_name not in config_data:
                config_data[config_name] = {}
            
            # Initialize EventAccumulator
            ea = EventAccumulator(event_file)
            ea.Reload()
            
            # Extract scalar tags
            scalar_tags = ea.Tags().get('scalars', [])
            
            # Define the metrics to extract
            metrics = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
            for metric in metrics:
                if metric in scalar_tags:
                    events = ea.Scalars(metric)
                    steps = [event.step for event in events]
                    values = [event.value for event in events]
                    if metric not in config_data[config_name]:
                        config_data[config_name][metric] = {'steps': steps, 'values': values}
        except Exception as e:
            logging.error(f"Failed to process event file {event_file}: {e}")
            continue
    
    # Now, plot each metric as a line chart per configuration
    for metric in ['train_loss', 'val_loss', 'train_acc', 'val_acc']:
        plt.figure(figsize=(12, 8))
        for config_name, metrics_dict in config_data.items():
            if metric in metrics_dict:
                plt.plot(metrics_dict[metric]['steps'], metrics_dict[metric]['values'], label=config_name)
            else:
                logging.warning(f"Metric '{metric}' not found for configuration '{config_name}'.")
        
        plt.title(f'{metric.capitalize()} Over Steps per Configuration', fontsize=16)
        plt.xlabel('Step', fontsize=14)
        plt.ylabel(metric.replace('_', ' ').capitalize(), fontsize=14)
        plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the plot
        safe_metric = metric.lower().replace(' ', '_')
        output_path = os.path.join(output_dir, f'{safe_metric}_line_chart.png')
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved line chart for {metric} to {output_path}")

def main():
    # Define the paths
    profiler_log_dir = './log/profiler/'  # Directory containing profiler .txt files
    tb_log_dir = './tb_logs/'  # Directory containing TensorBoard logs
    output_dir = './visualizations/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate profiler data and plot
    df_profiler = aggregate_profiler_data(profiler_log_dir)
    
    if not df_profiler.empty:
        # Display the aggregated DataFrame
        logging.info("\nAggregated Profiler DataFrame:")
        print(df_profiler)
        
        # Save the aggregated DataFrame to a CSV file for further analysis
        aggregated_csv_path = os.path.join(output_dir, 'aggregated_profiler_report.csv')
        df_profiler.to_csv(aggregated_csv_path, index=False)
        logging.info(f"Saved aggregated profiler data to {aggregated_csv_path}")
        
        # Define the metrics to visualize from profiler reports
        profiler_metrics = ['Mean Duration (s)', 'Total Time (s)']
        
        # Set the style for seaborn
        sns.set(style="whitegrid")
        
        # Create and save bar plots for each profiler metric
        for metric in profiler_metrics:
            if metric in df_profiler.columns:
                create_comparison_barplot(df_profiler, metric, output_dir)
            else:
                logging.warning(f"Profiler metric '{metric}' not found in the data.")
    else:
        logging.warning("No profiler data to plot.")
    
    # Parse and plot profiling_results.json for inference speed and validation metrics
    profiling_results_file = 'profiling_results.json'
    df_profiling_results = parse_profiling_results(profiling_results_file)
    
    if not df_profiling_results.empty:
        logging.info("\nParsed Profiling Results DataFrame:")
        print(df_profiling_results)
        
        # Save the profiling results to CSV
        profiling_results_csv = os.path.join(output_dir, 'profiling_results.csv')
        df_profiling_results.to_csv(profiling_results_csv, index=False)
        logging.info(f"Saved profiling results to {profiling_results_csv}")
        
        # Plot Average Inference Time
        if 'Average Inference Time (ms)' in df_profiling_results.columns:
            plot_inference_speed(df_profiling_results, output_dir)
        else:
            logging.warning("Average Inference Time (ms) not found in profiling results.")
        
        # Plot Validation Loss and Validation Accuracy
        validation_metrics = ['Validation Loss', 'Validation Accuracy']
        existing_metrics = [metric for metric in validation_metrics if metric in df_profiling_results.columns]
        if existing_metrics:
            plot_validation_metrics(df_profiling_results, output_dir)
        else:
            logging.warning("Validation metrics not found in profiling results.")
        
        # Plot Training Time
        if 'Training Time (s)' in df_profiling_results.columns:
            plot_training_time(df_profiling_results, output_dir)
        else:
            logging.warning("Training Time (s) not found in profiling results.")
    else:
        logging.warning("No profiling results data to plot.")
    
    # Parse and plot TensorBoard logs for loss and accuracy as line charts
    plot_line_charts_from_tensorboard(tb_log_dir, output_dir)

if __name__ == "__main__":
    main()
