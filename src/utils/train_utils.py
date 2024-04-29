# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import socket
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging
from datetime import datetime
import evaluate

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from tqdm import tqdm
from transformers import LlamaTokenizer
import json
import numpy as np

from src.utils.models_checkpointing.models_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from src.utils.models_checkpointing.mixed_precision import fpSixteen,bfSixteen
from src.utils.models_checkpointing.wrapping import get_bert_wrapper
from src.utils.memory_utils import MemoryTrace

import wandb
from memory_profiler import profile

from src.config.training_config import train_config as TRAIN_CONFIG
train_config = TRAIN_CONFIG()
def profile_to_file():
    def decorator(func):
        def wrapper(*args, **kwargs):
            with open(f"{train_config.output_dir}/{train_config.expr_name}.txt", 'a') as f:
                prof = profile(func, stream=f)
                return prof(*args, **kwargs)
        return wrapper
    return decorator


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)

# @profile_to_file()
def train(model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler, gradient_accumulation_steps, train_config):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    autocast =  nullcontext
    train_prep = []
    train_loss = []
    train_accu = []
    val_prep = []
    val_loss =[]
    val_accu = []
    test_prep = []
    test_loss = []
    test_accu = []

    # wandb
    wandb.init(
        config={
            'learning_rate': train_config.lr,
            'batch_size': train_config.batch_size_training,
            'num_epochs': train_config.num_epochs,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'dataset': train_config.dataset_name,
            'model': train_config.model_name,
            'scheduler': "StepLR",
            'mixed_precision': train_config.mixed_precision,
        },
        project='bert-sparsity-assi',
        # entity='bert-sparsify',
        notes=socket.gethostname(),
        name=train_config.expr_name,
        job_type="training",
        reinit=True
        )

    if train_config.save_metrics:
        metrics_filename = f"{train_config.output_dir}/{train_config.expr_name}/metrics_data-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []
        
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    best_val_accu = 0.0
    best_test_loss = float("inf")
    # add metrics method
    metrics = evaluate.load("accuracy")
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(train_config.num_epochs):
        # here I only update the default branch, with sparse adam
        epoch_start_time = time.perf_counter()
        total_gradient_memory_usage = 0
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            for step, batch in enumerate(train_dataloader):
                for key in batch.keys():
                    batch[key] = batch[key].to('cuda:0') #  101, 102 is start and end token

                # ===== run test ===== 
                if step == int(len(train_dataloader)/2) or step == len(train_dataloader):
                    test_ppl, test_epoch_loss, temp_test_loss, temp_step_perplexity, epoch_test_accu = evaluation(model, train_config, test_dataloader, epoch = epoch*2+2 if step ==total_length-1 else epoch*2+1, type="test")
                    if test_epoch_loss < best_test_loss:
                        best_test_loss = test_epoch_loss
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                    test_loss.append(float(best_test_loss))
                    test_prep.append(float(test_ppl))
                    test_accu.append(epoch_test_accu)
                    
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                predicted_labels = torch.argmax(logits, dim=1)
                metrics.add_batch(predictions=predicted_labels, references=batch["labels"])
                loss = loss / gradient_accumulation_steps
                if train_config.save_metrics:
                    train_step_loss.append(loss.detach().float().item())
                    train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                total_loss += loss.detach().float()
                # wandb.watch(model, log="gradients")
                # torch.cuda.synchronize()  # 确保所有之前的CUDA操作已完成, 这里可以清理一下内存。cuda.empty_cache()
                memory_usage_before = torch.cuda.memory_allocated() / (1024 ** 3) # 转换为GB
                loss.backward()  # retain_graph=True keep住前项的内容。peek mem以及差值memory
                # torch.cuda.synchronize()
                # memory_usage_after = torch.cuda.memory_allocated() / (1024 ** 3) # 转换为GB
                memory_usage_after = torch.cuda.memory_allocated() / (1024 ** 3) # 转换为GB
                gradient_memory_usage = memory_usage_before - memory_usage_after
                total_gradient_memory_usage += gradient_memory_usage
                avg_gradient_memory_usage = total_gradient_memory_usage / (step + 1)
                wandb.log({"Before Memory Usage (GB)": memory_usage_before, 
                           "After Memory Usage (GB)": memory_usage_after, 
                           "Gradient Memory Usage (GB)": gradient_memory_usage, 
                           "Avg Gradient Memory Usage": avg_gradient_memory_usage},
                           step=step+epoch*total_length)

                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                    optimizer.step()
                    optimizer.zero_grad() # gradient -> 0 float to int 0, set to 0, still here.
                    pbar.update(1)
                pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")
                if train_config.save_metrics:
                    save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                # log_memory_usage(step)
            pbar.close()
            
        wandb.log({"loss": loss.item()})
        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_perplexity = torch.exp(train_epoch_loss)
        
        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))
        accu = metrics.compute()['accuracy']
        train_accu.append(accu)
        # wandb.log({"train_perplexity": train_perplexity, "train_loss": train_epoch_loss, "train_accuracy": accu}, step=epoch+1)
        with open(f"{train_config.log_path}/{train_config.expr_name}.txt", "a") as f:
            f.write(f"===================Epoch {epoch+1}: train_accu={accu}=================\n")
            f.write(f"Time taken for epoch {epoch+1} is {epoch_end_time}\n")
            f.write(f"Max CUDA memory allocated was {memtrace.peak} GB\n")
            f.write(f"Max CUDA memory reserved was {memtrace.max_reserved} GB\n")
            f.write(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB\n")
            f.write(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}\n")
            f.write(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB\n")
            f.write(f'==============================================epoch end=============================================\n')

        # Update the learning rate as needed
        lr_scheduler.step()

        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity, epoch_val_accu = evaluation(model, train_config, eval_dataloader, epoch = epoch, type="eval")
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)

            checkpoint_start_time = time.perf_counter()
            # 只有loss更小的时候才会存。
            # if train_config.save_model and eval_epoch_loss < best_val_loss:
            if train_config.save_model and epoch_val_accu > best_val_accu:
                # best_val_accu = epoch_val_accu
                save_model_checkpoint(
                    model, optimizer, train_config, epoch=epoch
                )
                if train_config.save_optimizer:
                    save_optimizer_checkpoint(
                        model, optimizer, train_config, epoch=epoch
                    )
                    print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                    print("=====================================================")
                
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))
            val_accu.append(epoch_val_accu)

        print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        
        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, train_accu, val_step_loss, val_loss, val_step_perplexity, val_prep, val_accu)

    wandb.finish()
    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    avg_train_accu = sum(train_accu)/len(train_accu)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)
        avg_eval_accu = sum(val_accu)/len(val_accu)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
        results['avg_eval_accu'] = avg_eval_accu
        results['eval_accu'] = val_accu
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    results['avg_train_accu'] = avg_train_accu
    results['train_accu'] = train_accu
    
    
    with open(f"{train_config.log_path}/{train_config.expr_name}.txt", "a") as f:
        f.write(f"==========================================\n")
        f.write(f"results are : \n{results}\n")
        f.write(f"=====================training stage=====================\n")
    return results

def evaluation(model,train_config, eval_dataloader, epoch, type="eval"):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    # # wandb
    # wandb.init(
    #     config={
    #         'learning_rate': train_config.lr,
    #         'batch_size': train_config.batch_size_training,
    #         'num_epochs': train_config.num_epochs,
    #         'dataset': train_config.dataset_name,
    #         'model': train_config.model_name,
    #         'scheduler': "StepLR",
    #         'mixed_precision': train_config.mixed_precision,
    #     },
    #     project='bert-sparsity-test-during-train',
    #     entity='backward-sparsify',
    #     notes=socket.gethostname(),
    #     name=train_config.expr_name,
    #     job_type="test",
    #     reinit=True
    #     )
    model.eval()
    eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    metrics = evaluate.load("accuracy")
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            for key in batch.keys():
                batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                predicted_labels = torch.argmax(logits, dim=1)
                metrics.add_batch(predictions=predicted_labels, references=batch["labels"])
                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))  

                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            # eval_preds.extend(
            #     tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            # )
    val_accu = metrics.compute()['accuracy']
    if type == "test":
        wandb.log({"test_accuracy": val_accu}, step=epoch)
    # If there's more than one CUDA device, reduce evaluation loss across all devices
    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    # Print evaluation metrics
    print(f" {eval_ppl=} {eval_epoch_loss=}")
    # wandb.finish()
    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity, val_accu


def save_to_json(output_filename, train_step_loss, train_epoch_loss, train_step_ppl, train_epoch_ppl, train_epoch_accu, val_step_loss, val_epoch_loss, val_step_ppl, val_epoch_ppl, val_epoch_accu):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "train_accuracy": train_epoch_accu,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl,
        "val_accuracy": val_epoch_accu,
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)