# compare_or_single_profiler.py

import os
import gc
import json
import time
import shutil
import logging
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.profilers import SimpleProfiler
from torch.profiler import profile, ProfilerActivity, schedule



#####################################################################
# TO DO: import the activation modules from this repo instead of nn for comparsion 
#####################################################################



# 1) Data Loader
from data_loader import get_imagenette_dataloaders

# 2) Model Builder
from model_loader import create_simple_vit

# 3) CUDA-based rational activation init (if used)
from rationals_cuda_inline import init_rationals_cuda_activation

# 4) Other possible activations
import torch.nn as nn

###############################################################################
#                          Configuration Section
###############################################################################
# Choose whether to compare multiple activations OR just use a single one
# and run profiling. You can change this boolean or read from sys.argv/env.
COMPARE_ACTIVATIONS = True  # If False, only a single CUDA Rational run.

# If COMPARE_ACTIVATIONS is False, we define only the CUDA-based rational:
# If True, we define a list of activations to compare.

# Base training hyperparams
BATCH_SIZE = 16
IMG_SIZE = 160
MAX_EPOCHS = 6

###############################################################################
#                    Utility Functions (Profiler & Logging)
###############################################################################
def clean_profiler_logs(profiler_log_dir: str):
    """
    Delete all files in the profiler_log_dir to avoid clutter.
    """
    if os.path.exists(profiler_log_dir):
        for filename in os.listdir(profiler_log_dir):
            file_path = os.path.join(profiler_log_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    logging.info(f"Deleted file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    logging.info(f"Deleted directory: {file_path}")
            except Exception as e:
                logging.error(f"Failed to delete {file_path}. Reason: {e}")
    else:
        logging.info(f"Profiler log directory {profiler_log_dir} does not exist. No cleanup needed.")


def benchmark_inference(model, dataloader, device, num_batches=100):
    """
    Measures average inference time (milliseconds) per batch over `num_batches`.
    """
    model.eval()
    model.to(device)
    total_time = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            if count >= num_batches:
                break
            inputs, _ = batch
            inputs = inputs.to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()
            _ = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()

            total_time += (end_time - start_time) * 1000  # ms
            count += 1

    avg_time = total_time / count if count > 0 else 0.0
    return avg_time


###############################################################################
#                      Combined Profiling Logic
###############################################################################
def run_profiling_comprehensive(dataset_path: str, batch_size: int, img_size: int, max_epochs: int):
    """
    Runs the training & profiling either for a single activation or compares 
    multiple, depending on COMPARE_ACTIVATIONS.

    1. Load data
    2. Define which activations to test
    3. Run for each activation
       - If activation has parameters => run single & separate optim optimizer
       - If no parameters => run single optimizer
    4. Log results (training time, val metrics, inference time).
    """

    logging.info("Loading Datasets and DataLoaders...")
    train_loader, val_loader, test_loader = get_imagenette_dataloaders(
        dataset_path=None,
        img_size=img_size,
        batch_size=batch_size
    )

    # If just a single CUDA-based rational activation is desired:
    if not COMPARE_ACTIVATIONS:
        # We create one rational activation:
        single_act = init_rationals_cuda_activation(numerator_size=5, denominator_size=4, init="normal", init_std=0.1)
        # We'll test it with single vs. separate optimizers:
        all_activations = [("CUDA_Rational", single_act)]
    else:
        # Compare multiple activations. For example:
        # 1) CUDA-based rational
        # 2) Plain ReLU (no trainable params)
        # 3) Possibly a built-in GELU or something else
        # You can add as many as you want, e.g. a CPU-based rational or custom.
        rational_cuda = init_rationals_cuda_activation(numerator_size=5, denominator_size=4, init="normal", init_std=0.1)
        relu = nn.ReLU()
        gelu = nn.GELU()
        
        # Build a list of (name, module)
        all_activations = [
            ("CUDA_Rational", rational_cuda),
            ("ReLU", relu),
            ("GELU", gelu),
        ]

    # Create a set of results to store everything
    results = []

    # Prepare the profiler log directory
    profiler_log_dir = "./log/profiler"
    os.makedirs(profiler_log_dir, exist_ok=True)
    clean_profiler_logs(profiler_log_dir)

    # Directory for any additional logs
    visualizations_dir = "./visualizations"
    os.makedirs(visualizations_dir, exist_ok=True)

    # We'll test at 32-bit only. So no 16-bit references.
    # Also define possible "optimizer setups"
    #   - If the activation has parameters, we do [Single, Separate].
    #   - If not, we do only [Single].
    single_opt_cfg = {"optimizer_setup": "Single", "additional_activation_optimizer": False}
    separate_opt_cfg = {"optimizer_setup": "Separate", "additional_activation_optimizer": True}

    # Now iterate over each activation
    for (activation_name, activation_module) in all_activations:
        # Check if the activation has trainable parameters
        has_params = any(p.requires_grad for p in activation_module.parameters())

        # Build a list of config(s) for this activation
        if has_params:
            # param-based => test single & separate
            run_cfgs = [single_opt_cfg, separate_opt_cfg]
        else:
            # no params => only single
            run_cfgs = [single_opt_cfg]

        # For each config, run the training
        for cfg in run_cfgs:
            # Construct a name for logging
            config_name = cfg["optimizer_setup"]
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            unique_name = f"{activation_name}_{config_name}_{timestamp}"
            profiler_log_file = os.path.join(profiler_log_dir, f"{unique_name}_profiler.txt")

            # Build the model from create_simple_vit
            # We'll give it the 'activation_module' and the relevant "additional_activation_optimizer" flag
            model = create_simple_vit(
                image_size=img_size,
                patch_size=16,
                num_classes=10,
                dim=128,
                depth=4,
                heads=8,
                mlp_dim=256,
                channels=3,
                dim_head=32,
                lr=5e-5,
                activation_lr=5e-5,
                additional_activation_optimizer=cfg["additional_activation_optimizer"],
                accumulation_steps=1,
                max_norm_model=5.0,
                max_norm_activation=2.0,
            )
            # Overwrite the default rational activation with our chosen one
            model.activation = activation_module

            # Setup Logger
            logger = pl.loggers.TensorBoardLogger(
                save_dir="tb_logs",
                name=f"compare_{unique_name}"
            )

            # Setup profiler
            simple_profiler = SimpleProfiler(dirpath=profiler_log_dir, filename=f"{unique_name}_profiler_logs.txt")

            # Setup Trainer (32-bit only, no "16-mixed")
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                precision=32,  # Force 32-bit
                log_every_n_steps=1,
                logger=logger,
                enable_progress_bar=False,
                profiler=simple_profiler
            )

            # Profiler schedule - example: 1 wait, 2 warmup, 3 active
            prof_schedule = schedule(wait=1, warmup=2, active=3, repeat=1)

            try:
                # Train under profiler
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=prof_schedule,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                ) as prof:
                    start_time = time.time()

                    trainer.fit(model, train_loader, val_loader)

                    end_time = time.time()
                    prof.step()
                    train_time = end_time - start_time

                    final_val_loss = trainer.callback_metrics.get("val_loss")
                    final_val_acc = trainer.callback_metrics.get("val_acc")

                    # Store result
                    results.append({
                        "Activation": activation_name,
                        "Optimizer Setup": config_name,
                        "Val Loss": (final_val_loss.item() if final_val_loss else None),
                        "Val Acc": (final_val_acc.item() if final_val_acc else None),
                        "Train Time (s)": train_time,
                    })

                # Write profiler logs
                logging.info(f"Writing profiler logs to {profiler_log_file}")
                with open(profiler_log_file, 'w') as f:
                    f.write(f"Activation: {activation_name}, Optimizer={config_name}\n")
                    f.write("Profiler Key Averages:\n")
                    f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
                    f.write("\n\nDetailed Info:\n")
                    f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=50))
                    f.write("\n\n")

                if final_val_loss and final_val_acc:
                    print(f"Done: Act={activation_name}, Opt={config_name}, "
                          f"ValLoss={final_val_loss:.4f}, ValAcc={final_val_acc:.4f}, "
                          f"Time={train_time:.2f}s")

                # Benchmark inference
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                inf_ms = benchmark_inference(model, test_loader, device)
                logging.info(f"Inference per batch: {inf_ms:.2f} ms")

                # Store inference time
                results[-1]["Inference (ms/batch)"] = inf_ms

            except Exception as e:
                logging.error(f"Error for {unique_name}: {e}")

            # Cleanup GPU memory
            del model, trainer
            torch.cuda.empty_cache()
            gc.collect()

    # Print final summary
    print("\n==================== FINAL RESULTS ====================")
    for r in results:
        act = r["Activation"]
        opt = r["Optimizer Setup"]
        v_loss = r["Val Loss"]
        v_acc = r["Val Acc"]
        t_time = r["Train Time (s)"]
        i_time = r.get("Inference (ms/batch)", 0.0)
        print(f"Activation={act}, Optim={opt}, "
              f"ValLoss={v_loss:.4f if v_loss else 'N/A'}, "
              f"ValAcc={v_acc:.4f if v_acc else 'N/A'}, "
              f"TrainTime={t_time:.2f}s, InfTime={i_time:.2f}ms")

    # Optionally save to JSON
    out_file = "combined_profiling_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Saved results to {out_file}")


###############################################################################
#                               MAIN
###############################################################################
def main():
    # Basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Run the combined profiling with a single or multiple activations
    run_profiling_comprehensive(
        dataset_path=None,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        max_epochs=MAX_EPOCHS
    )

if __name__ == "__main__":
    main()
