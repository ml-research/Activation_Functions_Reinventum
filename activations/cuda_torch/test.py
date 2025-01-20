import os
import math
import time
import json
import torch
import torch.nn as nn
import pytorch_lightning as pl
import shutil  # For removing directories
import logging  # For logging information
import gc  # Garbage collection for cleanup
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchmetrics import Accuracy
from torch.autograd import Function
from einops import rearrange
from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pytorch_lightning.profilers import SimpleProfiler
from torch.profiler import profile, ProfilerActivity, schedule

import rationals_cuda_inline


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================
# Constants
# =============================
BATCH_SIZE = 16
IMG_SIZE = 160
NUM_CLASSES = 10
LR = 5e-5
ACTIVATION_LR = 5e-5
WEIGHT_DECAY = 1e-7  # Weight decay for main optimizer
GRAD_ACCUMULATION_STEPS = 1  # Number of steps to accumulate gradients
MAX_NORM_MODEL = 5.0  # Max norm for model parameters
MAX_NORM_ACTIVATION = 2.0  # Max norm for activation parameters

# =============================
# Data Preparation
# =============================
# Transforms
manual_transforms_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4600, 0.4549, 0.4274],
                         std=[0.2287, 0.2226, 0.2309])
])

manual_transforms_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4600, 0.4549, 0.4274],
                         std=[0.2287, 0.2226, 0.2309])
])

# Dataset Paths
image_path = "/workspaces/Act_Func_Reinventum/activations/cuda_torch/datasets/imagenette-160"
train_data_path = os.path.join(image_path, "train")
val_data_path = os.path.join(image_path, "val")

# Load datasets
train_dataset = datasets.ImageFolder(train_data_path, transform=manual_transforms_train)
test_dataset = datasets.ImageFolder(val_data_path, transform=manual_transforms_val)

# Split train into train/val
train_len = int(0.95 * len(train_dataset))
val_len = len(train_dataset) - train_len
train_data, val_data = random_split(train_dataset, [train_len, val_len])

# DataLoaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# =============================
# Custom CUDA Activation Function
# =============================
class RationalsCUDAFunction(Function):
    @staticmethod
    def forward(ctx, x, coeff_numerator, coeff_denominator):
        if not x.is_cuda:
            x = x.cuda()
        if not coeff_numerator.is_cuda:
            coeff_numerator = coeff_numerator.cuda()
        if not coeff_denominator.is_cuda:
            coeff_denominator = coeff_denominator.cuda()

        outputs = rationals_cuda_inline.rationals_cuda_forward(x, coeff_numerator, coeff_denominator)
        ctx.save_for_backward(x, coeff_numerator, coeff_denominator)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        x, coeff_numerator, coeff_denominator = ctx.saved_tensors
        grad_output = grad_output.to(dtype=torch.float32, device='cuda')
        grad_x, grad_coeff_numerator, grad_coeff_denominator = rationals_cuda_inline.rationals_cuda_backward(
            grad_output.contiguous(),
            x.contiguous(),
            coeff_numerator.contiguous(),
            coeff_denominator.contiguous()
        )
        return grad_x, grad_coeff_numerator, grad_coeff_denominator

# Activation Module
class RationalsCUDAActivation(nn.Module):
    def __init__(self):
        super(RationalsCUDAActivation, self).__init__()
        self.coeff_numerator = nn.Parameter(torch.randn((5,), dtype=torch.float32))
        self.coeff_denominator = nn.Parameter(torch.randn((4,), dtype=torch.float32))

    def forward(self, x):
        device = x.device
        coeff_numerator = self.coeff_numerator.to(device)
        coeff_denominator = self.coeff_denominator.to(device)

        orig_dtype = x.dtype
        if orig_dtype == torch.float16:
            x = x.to(torch.float32)
            output = RationalsCUDAFunction.apply(x, coeff_numerator, coeff_denominator)
            output = output.to(orig_dtype)
        else:
            output = RationalsCUDAFunction.apply(x, coeff_numerator, coeff_denominator)
        return output

# =============================
# Helper Functions
# =============================
def posemb_sincos_2d(h, w, dim, device, temperature=10000):
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]  # Shape: (h*w, dim//4)
    x = x.flatten()[:, None] * omega[None, :]  # Shape: (h*w, dim//4)
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)  # Now pe has shape (h*w, dim)
    return pe  # Shape: (h*w, dim)

# Patch Embedding Module
class PatchEmbedding(nn.Module):
    def __init__(self, channels, dim, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.conv(x)          # Shape: (B, D, H, W)
        x = x.flatten(2)          # Shape: (B, D, H*W)
        x = x.transpose(1, 2)     # Shape: (B, H*W, D)
        x = self.norm(x)
        return x

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, activation):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            activation,
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        x_res = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = x_res + attn_output  # Residual connection

        x_res = x
        x = self.norm2(x)
        x = x_res + self.ff(x)   # Residual connection
        return x

# =============================
# Vision Transformer Model with Configurable Optimizers and Manual Optimization
# =============================
class SimpleViT(pl.LightningModule):
    def __init__(self, 
                 image_size, 
                 patch_size, 
                 num_classes, 
                 dim, 
                 depth, 
                 heads, 
                 mlp_dim, 
                 channels, 
                 dim_head,  # Note: `dim_head` is defined but not used in this implementation
                 lr, 
                 activation,
                 activation_lr,
                 additional_activation_optimizer=True,  # New flag to toggle optimizer usage
                 accumulation_steps=GRAD_ACCUMULATION_STEPS,  # Gradient accumulation steps
                 max_norm_model=MAX_NORM_MODEL,  # Max norm for model parameters
                 max_norm_activation=MAX_NORM_ACTIVATION  # Max norm for activation parameters
                ):
        super().__init__()
        self.save_hyperparameters(ignore=['activation'])  # Saves all constructor arguments except 'activation'

        self.lr = lr
        self.activation_lr = activation_lr
        self.additional_activation_optimizer = additional_activation_optimizer

        self.accumulation_steps = accumulation_steps
        self.current_accumulation_step = 0

        self.max_norm_model = max_norm_model
        self.max_norm_activation = max_norm_activation

        self.to_patch_embedding = PatchEmbedding(channels, dim, patch_size)

        self.transformer = nn.ModuleList([
            TransformerBlock(dim=dim, heads=heads, mlp_dim=mlp_dim, activation=activation)
            for _ in range(depth)
        ])

        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

        self.activation = activation
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # Set automatic optimization based on the optimizer configuration
        if self.additional_activation_optimizer:
            self.automatic_optimization = False
        else:
            self.automatic_optimization = True

    def forward(self, img):
        x = self.to_patch_embedding(img)  # Shape: (B, N_patches, D)
        b, n_patches, d = x.shape
        h = w = int(math.sqrt(n_patches))  # Assuming square patches
        pe = posemb_sincos_2d(h, w, d, device=x.device)  # Shape: (h*w, dim)
        x = x + pe.unsqueeze(0)  # Shape: (B, h*w, dim)

        for block in self.transformer:
            x = block(x)

        x = x.mean(dim=1)  # Global average pooling
        x = self.to_latent(x)
        return self.linear_head(x)

    def configure_optimizers(self):
        if self.additional_activation_optimizer:
            # Get the IDs of activation parameters to exclude them from the main optimizer
            activation_param_ids = {id(p) for p in self.activation.parameters()}

            # Parameters for the main optimizer: all parameters not in activation
            main_params = [p for p in self.parameters() if id(p) not in activation_param_ids and p.requires_grad]

            # Parameters for the activation optimizer
            activation_params = list(self.activation.parameters())

            # Define the main optimizer
            main_optimizer = torch.optim.Adam(
                main_params,
                lr=self.lr,
                weight_decay=WEIGHT_DECAY
            )

            # Define the activation optimizer
            activation_optimizer = torch.optim.Adam(
                activation_params,
                lr=self.activation_lr
            )

            return [main_optimizer, activation_optimizer]
        else:
            # Single optimizer for all parameters
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=WEIGHT_DECAY
            )
            return optimizer

    def training_step(self, batch, batch_idx):
        # Reset accuracy at the start of an epoch
        if batch_idx == 0:
            self.train_acc.reset()

        # Get data from batch
        x, y = batch 

        # Forward pass
        y_pred = self(x)

        # Calculate loss
        loss = self.loss_fn(y_pred, y)

        if self.additional_activation_optimizer:
            # Manual backward pass
            self.manual_backward(loss)

            # Increment the accumulation step counter
            self.current_accumulation_step += 1

            # Perform optimizer step if accumulation_steps is reached
            if self.current_accumulation_step >= self.accumulation_steps:
                optimizers = self.optimizers()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.parameters() if p.requires_grad],
                    max_norm=self.max_norm_model
                )
                torch.nn.utils.clip_grad_norm_(
                    self.activation.parameters(),
                    max_norm=self.max_norm_activation
                )

                # Optimizer step and zero_grad
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()

                # Reset accumulation counter
                self.current_accumulation_step = 0  

            # Compute and update accuracy
            preds = torch.argmax(y_pred, dim=1)
            self.train_acc.update(preds, y)

            # Logging: Log once per epoch
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("train_acc", self.train_acc.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        else:
            # Automatic optimization handles the backward and optimizer steps
            self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("train_acc", self.train_acc.compute(), on_step=True, prog_bar=True, logger=True, sync_dist=True)
            return loss  # Necessary for automatic optimization

    def validation_step(self, batch, batch_idx):
        # Get data from batch
        x, y = batch

        # Forward pass
        y_pred = self(x)

        # Calculate loss
        loss = self.loss_fn(y_pred, y)

        # Compute accuracy
        preds = torch.argmax(y_pred, dim=1)
        self.val_acc.update(preds, y)

        # Logging
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_acc.compute(), on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        # Optional: Implement test step if needed
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        preds = torch.argmax(y_pred, dim=1)
        self.val_acc.update(preds, y)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', self.val_acc.compute(), on_epoch=True, prog_bar=True, logger=True)

# =============================
# Cleanup Function
# =============================
def clean_profiler_logs(profiler_log_dir):
    """
    Delete all files in the profiler_log_dir to prevent accumulation of old reports.

    Args:
        profiler_log_dir (str): Path to the profiler log directory.
    """
    if os.path.exists(profiler_log_dir):
        for filename in os.listdir(profiler_log_dir):
            file_path = os.path.join(profiler_log_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file or link
                    logging.info(f"Deleted file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory and its contents
                    logging.info(f"Deleted directory: {file_path}")
            except Exception as e:
                logging.error(f'Failed to delete {file_path}. Reason: {e}')
    else:
        logging.info(f"Profiler log directory {profiler_log_dir} does not exist. No cleanup needed.")


def benchmark_inference(model, dataloader, device, num_batches=100):
    """
    Benchmark the inference speed of a model.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for inference data.
        device (torch.device): Device to run inference on.
        num_batches (int): Number of batches to benchmark.

    Returns:
        float: Average inference time per batch in milliseconds.
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

            # Ensure CUDA operations are complete before timing
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()
            outputs = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()

            batch_time = (end_time - start_time) * 1000  # Convert to milliseconds
            total_time += batch_time
            count += 1

    avg_time = total_time / count if count > 0 else 0.0
    return avg_time


# =============================
# Profiling Function
# =============================
def run_profiling():
    # Define the configurations
    configurations = [
        {"optimizer_setup": "Separate Optimizers", "precision": 32, "additional_activation_optimizer": True},
        {"optimizer_setup": "Separate Optimizers", "precision": "16-mixed", "additional_activation_optimizer": True},
        {"optimizer_setup": "Single Optimizer", "precision": 32, "additional_activation_optimizer": False},
        {"optimizer_setup": "Single Optimizer", "precision": "16-mixed", "additional_activation_optimizer": False},
    ]

    # Initialize a list to store results
    results = []

    # Create the directory for profiler logs if it doesn't exist
    profiler_log_dir = "./log/profiler"
    os.makedirs(profiler_log_dir, exist_ok=True)

    # **Cleanup Step: Delete existing profiler log files**
    logging.info("Starting cleanup of existing profiler log files.")
    clean_profiler_logs(profiler_log_dir)
    logging.info("Cleanup completed.")

    # Directory to save visualization plots
    visualizations_dir = "./visualizations"
    os.makedirs(visualizations_dir, exist_ok=True)


    for config in configurations:
        print(f"\nRunning Configuration: {config['optimizer_setup']} with Precision {config['precision']}-bit")

        # Generate a unique log filename based on the configuration and current timestamp
        config_name = f"{config['optimizer_setup'].replace(' ', '_')}_{config['precision']}"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_name = f"{config_name}_{timestamp}"
        profiler_log_file = os.path.join(profiler_log_dir, f"{unique_name}_profiler.txt")

        try:

            activation = RationalsCUDAActivation()
            

            # Initialize the model with the desired configuration
            model = SimpleViT(
                image_size=IMG_SIZE,
                patch_size=16,
                num_classes=NUM_CLASSES,
                dim=128,
                depth=4,
                heads=8,
                mlp_dim=256,
                channels=3,
                dim_head=32,  # Note: `dim_head` is defined but not used in this implementation
                lr=LR,
                activation=activation,
                activation_lr=ACTIVATION_LR,
                additional_activation_optimizer=config['additional_activation_optimizer'],
                accumulation_steps=GRAD_ACCUMULATION_STEPS,
                max_norm_model=MAX_NORM_MODEL,
                max_norm_activation=MAX_NORM_ACTIVATION
            )

            # Initialize the Logger with a unique name per configuration
            logger = pl.loggers.TensorBoardLogger(
                save_dir="tb_logs",
                name=f"vit_profiling_{unique_name}",
                # version is omitted to allow auto-incrementing
            )

            # Log the logger's directory for verification
            logging.info(f"Logger directory: {logger.log_dir}")

            # Initialize the Trainer
            simple_profiler = SimpleProfiler(dirpath=profiler_log_dir, filename=f"{unique_name}_profiler_logs.txt")
            trainer = pl.Trainer(
                max_epochs=6,  # Increased epochs as per user request
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1 if torch.cuda.is_available() else None,
                precision=config['precision'],  # 16 or 32
                log_every_n_steps=1,
                logger=logger,
                enable_progress_bar=False,  # Disable progress bar for cleaner output
                profiler=simple_profiler
            )

            # Profiling schedule
            prof_schedule = schedule(wait=1, warmup=2, active=3, repeat=2)

            # Start profiling
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=prof_schedule,
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                # Record the start time
                start_time = time.time()

                # Fit the model
                trainer.fit(model, train_loader, val_loader)

                # Record the end time
                end_time = time.time()

                # Step profiler (if necessary)
                prof.step()

                # Calculate training time
                training_time = end_time - start_time

                # Retrieve the final validation loss and accuracy
                final_val_loss = trainer.callback_metrics.get("val_loss")
                final_val_acc = trainer.callback_metrics.get("val_acc")

                # Append the results
                results.append({
                    "Optimizer Setup": config['optimizer_setup'],
                    "Precision": f"{config['precision']}-bit",
                    "Validation Loss": final_val_loss.item() if final_val_loss is not None else None,
                    "Validation Accuracy": final_val_acc.item() if final_val_acc is not None else None,
                    "Training Time (s)": training_time
                })

            # Write profiler results to a unique file after the context
            logging.info(f"Writing profiler logs to {profiler_log_file}")
            with open(profiler_log_file, 'w') as f:
                f.write(f"Configuration: {config['optimizer_setup']} with Precision {config['precision']}-bit\n")
                f.write("Profiler Key Averages:\n")
                f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=100))
                f.write("\n\nDetailed Profiler Information:\n")
                f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=100))
                f.write("\n\n")
            logging.info("Profiler logs written.")

            print(f"Completed: Validation Loss={final_val_loss:.4f}, Validation Acc={final_val_acc:.4f}, Training Time={training_time:.2f}s")

            # Benchmark Inference Speed
            logging.info("Starting inference speed benchmarking...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            avg_inference_time = benchmark_inference(model, test_loader, device, num_batches=100)
            logging.info(f"Average Inference Time per Batch: {avg_inference_time:.2f} ms")

            # Append inference time to results
            results[-1]["Average Inference Time (ms)"] = avg_inference_time

            # Cleanup to free GPU memory
            logging.info("Cleaning up GPU memory...")
            del model, trainer, activation
            torch.cuda.empty_cache()
            gc.collect()
            logging.info("Cleanup completed.")

        except Exception as e:
            logging.error(f"An error occurred during configuration '{config_name}': {e}")
            continue  # Proceed to the next configuration

    # Print a summary of results
    print("\n===== Profiling Results =====")
    for res in results:
        print(f"Optimizer: {res['Optimizer Setup']}, Precision: {res['Precision']}, "
              f"Val Loss: {res['Validation Loss']:.4f}, Val Acc: {res['Validation Accuracy']:.4f}, "
              f"Training Time: {res['Training Time (s)']:.2f}s, "
              f"Avg Inference Time: {res['Average Inference Time (ms)']:.2f} ms")

    # Optionally, save the results to a file
    with open('profiling_results.json', 'w') as f:
        json.dump(results, f, indent=4)

# =============================
# Test with PyTorch Lightning Trainer
# =============================
def test_with_imagenette():
    run_profiling()

if __name__ == "__main__":
    test_with_imagenette()
