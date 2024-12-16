"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import argparse
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tlam.models import gpt2_hermite
from tlam.models.gpt2 import GPTConfig, GPT
from tlam.models.gpt2_hermite import GPTHermite
from tlam.models.gpt2_fourier import GPTFourier


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script for GPT models.")

    # I/O
    parser.add_argument(
        "--out_dir", type=str, default="out", help="Output directory for checkpoints."
    )
    parser.add_argument(
        "--eval_interval", type=int, default=2000, help="Interval for evaluation."
    )
    parser.add_argument(
        "--log_interval", type=int, default=1, help="Interval for logging."
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=200,
        help="Number of iterations for evaluation.",
    )
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation.")
    parser.add_argument(
        "--always_save_checkpoint",
        action="store_true",
        help="Always save a checkpoint after each eval.",
    )
    parser.add_argument(
        "--init_from",
        type=str,
        default="scratch",
        choices=["scratch", "resume", "gpt2*"],
        help="Initialization type.",
    )

    # wandb logging
    parser.add_argument(
        "--wandb_log", action="store_true", help="Enable Weights & Biases logging."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="owt",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=f"gpt2-{int(time.time())}",
        help="Weights & Biases run name.",
    )

    # Data
    parser.add_argument(
        "--dataset", type=str, default="openwebtext", help="Dataset name."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--batch_size", type=int, default=12, help="Micro-batch size.")
    parser.add_argument(
        "--block_size", type=int, default=1024, help="Block size for model."
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["GPT", "GPTHermite", "GPTFourier"],
        help="Specify the model to train",
    )
    parser.add_argument(
        "--n_layer", type=int, default=12, help="Number of layers in the model."
    )
    parser.add_argument(
        "--n_head", type=int, default=12, help="Number of attention heads."
    )
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding size.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument(
        "--bias", action="store_true", help="Use bias in LayerNorm and Linear layers."
    )

    # AdamW optimizer
    parser.add_argument(
        "--learning_rate", type=float, default=6e-4, help="Learning rate."
    )
    parser.add_argument(
        "--max_iters", type=int, default=600000, help="Maximum number of iterations."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-1,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="Beta1 for AdamW optimizer."
    )
    parser.add_argument(
        "--beta2", type=float, default=0.95, help="Beta2 for AdamW optimizer."
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping value."
    )

    # Learning rate decay
    parser.add_argument(
        "--decay_lr", action="store_true", help="Enable learning rate decay."
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=2000, help="Number of warmup iterations."
    )
    parser.add_argument(
        "--lr_decay_iters",
        type=int,
        default=600000,
        help="Number of iterations to decay learning rate.",
    )
    parser.add_argument(
        "--min_lr", type=float, default=6e-5, help="Minimum learning rate."
    )

    # DDP settings
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Backend for DDP.",
    )

    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "bfloat16", "float16"],
        help="Data type for training.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use PyTorch 2.0 compile for faster training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    return parser.parse_args()


# Parse arguments
args = parse_arguments()

# Access parsed arguments as needed, e.g., args.out_dir, args.eval_interval, etc.
config = vars(args)


# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=args.backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    # assert args.gradient_accumulation_steps % ddp_world_size == 0
    # args.gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = (
    args.gradient_accumulation_steps
    * ddp_world_size
    * args.batch_size
    * args.block_size
)
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(args.out_dir, exist_ok=True)
torch.manual_seed(args.seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = (
    "cuda" if "cuda" in args.device else "cpu"
)  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[args.dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# poor man's data loader
data_dir = os.path.join("data", args.dataset)


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + args.block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + args.block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = (
            x.pin_memory().to(args.device, non_blocking=True),
            y.pin_memory().to(args.device, non_blocking=True),
        )
    else:
        x, y = x.to(args.device), y.to(args.device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_embd=args.n_embd,
    block_size=args.block_size,
    bias=args.bias,
    vocab_size=None,
    dropout=args.dropout,
)  # start with model_args from command line
if args.init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    # Select and train the specified model
    if args.model == "GPT":
        model = GPT(gptconf)
    elif args.model == "GPTHermite":
        model = GPTHermite(gptconf)
    elif args.model == "GPTFourier":
        model = GPTFourier(gptconf)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
elif args.init_from == "resume":
    print(f"Resuming training from {args.out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    # Select and train the specified model
    if args.model == "GPT":
        model = GPT(gptconf)
    elif args.model == "GPTHermite":
        model = GPTHermite(gptconf)
    elif args.model == "GPTFourier":
        model = GPTFourier(gptconf)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
elif args.init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=args.dropout)
    model = GPT.from_pretrained(args.init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if args.block_size < model.config.block_size:
    model.crop_block_size(args.block_size)
    model_args["block_size"] = (
        args.block_size  # so that the checkpoint will have the right value
    )
model.to(args.device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type
)
if args.init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if args.compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.lr_decay_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


# logging
if args.wandb_log and master_process:
    import wandb

    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)

# training loop
X, Y = get_batch("train")  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if args.decay_lr else args.learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % args.eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if args.wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }
            )
        if losses["val"] < best_val_loss or args.always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {args.out_dir}")
                torch.save(checkpoint, os.path.join(args.out_dir, "ckpt.pt"))
    if iter_num == 0 and args.eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(args.gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == args.gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss = model(X, Y)
            loss = (
                loss / args.gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if args.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % args.log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * args.gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(
                args.batch_size * args.gradient_accumulation_steps, dt
            )
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > args.max_iters:
        break

if ddp:
    destroy_process_group()
