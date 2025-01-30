# model_loader.py

import math
import logging
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy

from rationals_cuda_inline import init_rationals_cuda_activation


###############################################################################
#                            Put your models here:
###############################################################################
#                            Model 1: SimpleViT
###############################################################################
class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patch embeddings.
    """
    def __init__(self, channels: int, dim: int, patch_size: int):
        """
        Args:
            channels (int): Number of input channels (3 for RGB).
            dim (int): Dimensionality for the patch embeddings.
            patch_size (int): Size of each image patch (square).
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Shape (B, C, H, W)

        Returns:
            Tensor of shape (B, N_patches, D)
        """
        x = self.conv(x)          # (B, D, H//patch_size, W//patch_size)
        x = x.flatten(2)          # (B, D, H'*W')
        x = x.transpose(1, 2)     # (B, H'*W', D)
        x = self.norm(x)
        return x

def posemb_sincos_2d(h: int, w: int, dim: int, device, temperature: float = 10000.0):
    """
    2D sinusoidal positional embedding, often used in Vision Transformers.
    Returns a tensor of shape (h*w, dim).
    """
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe  # shape: (h*w, dim)

class TransformerBlock(nn.Module):
    """
    A standard Transformer block with:
      - Multihead Self-Attention
      - MLP (2-layer feedforward + activation)
      - Residual connections
      - Layer Normalization
    """
    def __init__(self, dim: int, heads: int, mlp_dim: int, activation: nn.Module):
        """
        Args:
            dim (int): Embedding dimension.
            heads (int): Number of attention heads.
            mlp_dim (int): Hidden dimension in the MLP.
            activation (nn.Module): Activation function for the MLP (e.g. rational CUDA activation).
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            activation,  # Custom rational activation goes here
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Self-Attention sublayer ---
        x_res = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)  # (B, N, D)
        x = x_res + attn_output  # residual

        # --- Feedforward (MLP) sublayer ---
        x_res = x
        x = self.norm2(x)
        x = x_res + self.ff(x)   # residual

        return x

class SimpleViT(pl.LightningModule):
    """
    A Vision Transformer model that can optionally use a separate optimizer
    for the rational activation parameters vs. the rest of the model.
    """
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int,
        dim_head: int,  # Not specifically used here, but left for clarity
        lr: float,
        activation: nn.Module,  # A pre-initialized activation module
        activation_lr: float,
        additional_activation_optimizer: bool = True,
        accumulation_steps: int = 1,
        max_norm_model: float = 5.0,
        max_norm_activation: float = 2.0,
    ):
        """
        Args:
            image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dim_head:
                Standard Vision Transformer hyperparameters.
            lr (float): Learning rate for main model parameters.
            activation (nn.Module): Custom activation module to use (e.g. rationals_cuda_inline).
            activation_lr (float): Learning rate for the activation coefficients if using a separate optimizer.
            additional_activation_optimizer (bool): Whether to use a separate optimizer for the activation's params.
            accumulation_steps (int): Gradient accumulation steps if using manual optimization.
            max_norm_model (float): Gradient clipping norm for main model parameters.
            max_norm_activation (float): Gradient clipping norm for activation parameters.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["activation"])  # Store hyperparameters except the actual module

        self.lr = lr
        self.activation_lr = activation_lr
        self.additional_activation_optimizer = additional_activation_optimizer

        self.accumulation_steps = accumulation_steps
        self.current_accumulation_step = 0

        self.max_norm_model = max_norm_model
        self.max_norm_activation = max_norm_activation

        # Model components
        self.to_patch_embedding = PatchEmbedding(channels, dim, patch_size)

        self.transformer = nn.ModuleList([
            TransformerBlock(dim=dim, heads=heads, mlp_dim=mlp_dim, activation=activation)
            for _ in range(depth)
        ])

        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

        self.activation = activation
        self.loss_fn = nn.CrossEntropyLoss()

        # TorchMetrics: track training/validation accuracy
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # If we want a separate activation optimizer, we turn off Lightning's automatic optimization
        self.automatic_optimization = not self.additional_activation_optimizer

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(img)
        b, n_patches, d = x.shape

        # Create 2D sinusoidal positional embeddings
        h = w = int(math.sqrt(n_patches))
        pe = posemb_sincos_2d(h, w, d, device=x.device)
        x = x + pe.unsqueeze(0)  # broadcast over batch

        # Pass through Transformer blocks
        for block in self.transformer:
            x = block(x)

        # Global average pooling over patch dimension
        x = x.mean(dim=1)  # shape (B, D)

        # Optionally transform to latent space (Identity here), then classification head
        x = self.to_latent(x)
        return self.linear_head(x)

    def configure_optimizers(self):
        """
        Create either:
         1) Two separate optimizers (one for main model, one for activation) if
            `self.additional_activation_optimizer` is True.
         2) A single optimizer covering all parameters if False.
        """
        if self.additional_activation_optimizer:
            # Extract activation parameters separately
            activation_param_ids = {id(p) for p in self.activation.parameters()}
            main_params = [p for p in self.parameters()
                           if id(p) not in activation_param_ids and p.requires_grad]
            activation_params = list(self.activation.parameters())

            main_optimizer = torch.optim.Adam(main_params, lr=self.lr)
            activation_optimizer = torch.optim.Adam(activation_params, lr=self.activation_lr)
            return [main_optimizer, activation_optimizer]
        else:
            # Single optimizer for everything
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

    def training_step(self, batch, batch_idx):
        """
        If using separate optimizers, we handle the backward pass and 
        optimizer steps manually. Otherwise, we return a loss for Lightning's
        automatic handling.
        """
        if batch_idx == 0:
            self.train_acc.reset()

        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)

        if self.additional_activation_optimizer:
            # Manual optimization
            self.manual_backward(loss)
            self.current_accumulation_step += 1

            # Once we hit accumulation_steps, step the optimizers
            if self.current_accumulation_step >= self.accumulation_steps:
                optimizers = self.optimizers()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.parameters() if p.requires_grad],
                    max_norm=self.max_norm_model
                )
                torch.nn.utils.clip_grad_norm_(
                    self.activation.parameters(),
                    max_norm=self.max_norm_activation
                )

                # Step each optimizer
                for opt in optimizers:
                    opt.step()
                    opt.zero_grad()

                self.current_accumulation_step = 0

            # Update accuracy
            preds = torch.argmax(y_pred, dim=1)
            self.train_acc.update(preds, y)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_acc", self.train_acc.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            # Automatic optimization
            self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
            self.log("train_acc", self.train_acc.compute(), on_step=True, prog_bar=True, logger=True)
            return loss

    def validation_step(self, batch, batch_idx):
        """
        Standard validation step. Lightning handles the forward pass and logging.
        """
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)

        preds = torch.argmax(y_pred, dim=1)
        self.val_acc.update(preds, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_acc.compute(), on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        """
        Optional test step. Here we re-use val_acc for demonstration.
        """
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)

        preds = torch.argmax(y_pred, dim=1)
        self.val_acc.update(preds, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", self.val_acc.compute(), on_epoch=True, prog_bar=True, logger=True)

###############################################################################
#                      Model 2: "Your Model here"
###############################################################################



###############################################################################




###############################################################################
#                      Init Functions
###############################################################################
#                      Init SimleVit:
###############################################################################
def create_simple_vit(
    image_size: int = 160,
    patch_size: int = 16,
    num_classes: int = 10,
    dim: int = 128,
    depth: int = 4,
    heads: int = 8,
    mlp_dim: int = 256,
    channels: int = 3,
    dim_head: int = 32, 
    lr: float = 5e-5,
    activation_lr: float = 5e-5,
    additional_activation_optimizer: bool = True,
    accumulation_steps: int = 1,
    max_norm_model: float = 5.0,
    max_norm_activation: float = 2.0,
    activation_kwargs: dict = None,
):
    """
    Factory function to build a SimpleViT model with a custom CUDA-based rational activation.
    
    Args:
        image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dim_head:
            Vision Transformer parameters.
        lr (float): Learning rate for main model params.
        activation_lr (float): Learning rate for the rational activation params (if separate optimizer).
        additional_activation_optimizer (bool): Whether to use separate optimizers for model vs. activation.
        accumulation_steps (int): Steps of grad accumulation before performing an optimizer step.
        max_norm_model (float): Gradient clipping norm for model parameters.
        max_norm_activation (float): Gradient clipping norm for activation parameters.
        activation_kwargs (dict): Dictionary of arguments for the init_rationals_cuda_activation function 
                                  (e.g. {"numerator_size":6, "denominator_size":5, "init":"uniform"}).
    """
    logging.info("Creating SimpleViT with custom rational CUDA activation...")

    if activation_kwargs is None:
        activation_kwargs = {}

    # Obtain the rational activation module from the external function
    activation = init_rationals_cuda_activation(**activation_kwargs)

    # Instantiate the Vision Transformer
    model = SimpleViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        channels=channels,
        dim_head=dim_head,
        lr=lr,
        activation=activation,  # Pass in the rational activation module
        activation_lr=activation_lr,
        additional_activation_optimizer=additional_activation_optimizer,
        accumulation_steps=accumulation_steps,
        max_norm_model=max_norm_model,
        max_norm_activation=max_norm_activation
    )
    return model

###############################################################################
#                      Your init code here:
###############################################################################



###############################################################################