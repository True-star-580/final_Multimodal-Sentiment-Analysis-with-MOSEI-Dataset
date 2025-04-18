import os
import sys
import logging
import time
from pathlib import Path
import json

def setup_logging(log_file=None, level=logging.INFO):
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Configure formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is provided
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(exist_ok=True, parents=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to {log_file}")

def log_to_file(message, log_file, step=None):
    log_file = Path(log_file)
    log_file.parent.mkdir(exist_ok=True, parents=True)
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        # Write step information if provided
        if step is not None:
            f.write(f"[Step {step}] {timestamp} - {message}\n")
        else:
            f.write(f"{timestamp} - {message}\n")

# Wrapper for TensorBoard logging
class TensorboardLogger:
    def __init__(self, log_dir, experiment_name=None):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logging.warning("TensorBoard not installed. TensorBoard logging disabled.")
            self.writer = None
            return
        
        log_dir = Path(log_dir)
        if experiment_name is not None:
            log_dir = log_dir / experiment_name
            
        log_dir.mkdir(exist_ok=True, parents=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        logging.info(f"TensorBoard logging to {log_dir}")
    
    def log_scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag, tag_value_dict, step):
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_value_dict, step)
    
    def log_histogram(self, tag, values, step):
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)
    
    def log_model_graph(self, model, input_to_model):
        if self.writer is not None:
            self.writer.add_graph(model, input_to_model)
    
    def close(self):
        if self.writer is not None:
            self.writer.close()

# Helper class to track training and validation metrics
class MetricsTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.metrics = {}
    
    def update_train_loss(self, loss):
        self.train_losses.append(loss)
    
    def update_val_loss(self, loss):
        self.val_losses.append(loss)
    
    def update_metrics(self, metrics_dict, step):
        for metric_name, value in metrics_dict.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            
            # Ensure metrics list is long enough for current step
            while len(self.metrics[metric_name]) <= step:
                self.metrics[metric_name].append(None)
            
            self.metrics[metric_name][step] = value
    
    def get_latest_metrics(self):
        return {name: values[-1] for name, values in self.metrics.items()}
    
    def get_best_metrics(self):
        result = {}
        for name, values in self.metrics.items():
            # For loss-like metrics (lower is better)
            if "loss" in name or "mae" in name:
                # Remove None values
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    result[name] = min(valid_values)
            # For accuracy-like metrics (higher is better)
            else:
                # Remove None values
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    result[name] = max(valid_values)
        
        return result
    
    def save_metrics(self, path):
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.metrics, f)
    
    def reset(self):
        self.train_losses.clear()
        self.val_losses.clear()
        self.metrics.clear()