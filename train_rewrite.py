"""
Clean version of NanoGPT training script focusing on single GPU training
with empirical learning rate scheduling.
"""
import os
import time
import math
import pickle
from contextlib import nullcontext
from tqdm import tqdm

import numpy as np
import torch
import wandb

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
class TrainingConfig:
    def __init__(self):
        # Model config
        self.n_layer = 6
        self.n_head = 6
        self.n_embd = 384
        self.block_size = 256
        self.dropout = 0.0
        self.bias = False

        # Training config
        self.batch_size = 4
        self.max_iters = 6000
        self.eval_interval = 100
        self.eval_iters = 100
        self.log_interval = 10
        self.device = 'cuda'
        self.compile = True

        # Optimizer config
        self.learning_rate = 6e-4
        self.min_lr = 6e-5
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        
        # Data config
        self.dataset = 'shakespeare'
        
        # Wandb config
        self.wandb_log = True
        self.wandb_project = 'nanogpt_clean'
        self.wandb_run_name = f'run_{time.strftime("%Y%m%d_%H%M%S")}'

        # System/device config
        self.dtype = ('bfloat16' if torch.cuda.is_available() 
                     and torch.cuda.is_bf16_supported() else 'float16')

# -----------------------------------------------------------------------------
# Training infrastructure
# -----------------------------------------------------------------------------
class Trainer:
    def __init__(self, config):
        self.config = config
        self.setup_device()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        if config.wandb_log:
            self.setup_wandb()

    def setup_device(self):
        """Configure device and precision settings"""
        self.device_type = 'cuda' if 'cuda' in self.config.device else 'cpu'
        self.ptdtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16
        }[self.config.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else \
                  torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)

    def setup_data(self):
        """Setup data loading"""
        data_dir = os.path.join('data', self.config.dataset)
        self.train_data = np.memmap(os.path.join(data_dir, 'train.bin'), 
                                   dtype=np.uint16, mode='r')
        self.val_data = np.memmap(os.path.join(data_dir, 'val.bin'), 
                                 dtype=np.uint16, mode='r')
        
        # Load meta data (vocab size)
        meta_path = os.path.join(data_dir, 'meta.pkl')
        self.meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                self.meta_vocab_size = meta['vocab_size']
        
    def setup_model(self):
        """Initialize the model"""
        model_args = dict(
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_embd=self.config.n_embd,
            block_size=self.config.block_size,
            bias=self.config.bias,
            vocab_size=self.meta_vocab_size or 50304,
            dropout=self.config.dropout
        )
        gptconf = GPTConfig(**model_args)
        self.model = GPT(gptconf)
        self.model.to(self.config.device)
        
        if self.config.compile:
            print("Compiling model (this may take a minute)...")
            self.model = torch.compile(self.model)

    def setup_optimizer(self):
        """Setup optimizer and gradient scaler"""
        self.optimizer = self.model.configure_optimizers(
            self.config.weight_decay,
            self.config.learning_rate,
            (self.config.beta1, self.config.beta2),
            self.device_type
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config.dtype == 'float16'))

    def setup_wandb(self):
        """Initialize wandb logging"""
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=self.config.__dict__
        )

    def get_batch(self, split):
        """Get a batch of data"""
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
        x = torch.stack([torch.from_numpy(
            (data[i:i+self.config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(
            (data[i+1:i+1+self.config.block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        """Estimate loss on train and validation sets"""
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config.__dict__,
        }
        torch.save(checkpoint, os.path.join('out', 'ckpt.pt'))

# -----------------------------------------------------------------------------
# Actually run the training
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Initialize config and trainer
    config = TrainingConfig()
    trainer = Trainer(config)
    
    # Here we'll add the empirical LR scheduler and warmup comparison code
    # [Previous LR scheduler code would go here]
    
    print("Starting training...")