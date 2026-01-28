# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop."""

import copy
import json
import os
import pickle
import time
from typing import Tuple

import numpy as np
import PIL.Image
import psutil
import torch
from torch.utils.data.distributed import DistributedSampler

import dnnlib
import legacy
from manifold_metrics import (
    HYPERPARAMS,
    CNN2D_Residual,
    DatasetMock,
    calculate_fid,
    compute_precision_recall,
    get_synthetic_features,
    get_train_features,
)
from torch_utils import misc, training_stats
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
from visualization_utils import compute_avg_accuracy

# ----------------------------------------------------------------------------


def safe_remove_files(file_path, extensions, other_best_path=None):
    """
    Safely remove files with given extensions if the path is different from another best path.

    Args:
        file_path: Base path of files to remove (without extension)
        extensions: List of file extensions to remove (e.g., ['.png', '.raw'] or ['.pkl'])
        other_best_path: Path that should not be removed (e.g., best path for another metric)
    """
    if file_path is not None and file_path != other_best_path:
        for ext in extensions:
            try:
                os.remove(file_path + ext)
            except OSError:
                pass


def update_best_paths(
    is_best,
    save_all,
    current_path,
    old_best_path,
    other_metric_best_path,
    files_extensions,
    metric_name="",
    verbose=False,
):
    """
    Update best model paths and remove old files if needed.

    Args:
        is_best: Whether this is the best result so far
        save_all: Whether to save all checkpoints
        current_path: Current file path to potentially set as best
        old_best_path: Previous best path for this metric
        other_metric_best_path: Best path for another metric (to avoid deletion)
        files_extensions: List of file extensions to remove
        metric_name: Name of the metric for logging
        verbose: Whether to print removal messages

    Returns:
        Updated best path or None if not applicable
    """
    if is_best and not save_all:
        if verbose and old_best_path is not None and old_best_path != other_metric_best_path:
            print(f"Removing previous best {metric_name}:", old_best_path)

        safe_remove_files(old_best_path, files_extensions, other_metric_best_path)
        new_best_path = current_path

        if verbose:
            print(f"Updated best {metric_name} path to:", new_best_path)

        return new_best_path
    return old_best_path


# ----------------------------------------------------------------------------


def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


# ----------------------------------------------------------------------------


def save_image_grid(img, fname, drange, grid_size, save_rgb=True):
    """Save a grid of images into a single image file."""
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    # Save a copy keeping the original range
    original_img = img.copy()

    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    assert C in [1, 3, 5], f"Invalid value for C: {C}. Must be one of [1, 3, 5]."
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], "L").save(fname)
    if C == 3:
        PIL.Image.fromarray(img, "RGB").save(fname)
    if C == 5:
        if save_rgb:
            PIL.Image.fromarray(img[:, :, [2, 1, 0]], "RGB").save(fname)  # Select rgb channels
        # Save as raw image with all channels
        save_raw_image(fname.replace(".png", ".raw"), original_img, drange=drange)


# ----------------------------------------------------------------------------


def save_raw_image(filename: str, image: np.ndarray, drange: Tuple[float, float] = (-1, 1)):
    """
    Save a multi-channel image to a .raw file in the format:
    [num_channels, height, width] (uint32) + image data (uint32).

    The image is scaled to the range [0, 65535] (uint16) based on its current range,
    which can be [-1, 1], [0, 1], or [0, 255].

    Args:
        filename (str): path to the output .raw file
        image (np.ndarray): array with shape (height, width, num_channels)
        drange (Tuple[float, float]): range of the input image values
    """
    if image.ndim != 3:
        raise ValueError("Image must have 3 dimensions: (height, width, num_channels)")

    height, width, num_channels = image.shape
    lo, hi = drange

    # Normalize image and convert to uint32
    image = image.astype(np.float32)
    image = (image - lo) * (65535 / (hi - lo))
    image = np.rint(image).clip(0, 65535).astype(np.uint32)

    header = np.array([num_channels, height, width], dtype=np.uint32)

    with open(filename, "wb") as f:
        header.tofile(f)
        image.tofile(f)


# ----------------------------------------------------------------------------


def training_loop(
    run_dir=".",  # Output directory.
    training_set_kwargs={},  # Options for training set.
    validation_set_kwargs={},  # Options for validation set.
    data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
    G_kwargs={},  # Options for generator network.
    D_kwargs={},  # Options for discriminator network.
    G_opt_kwargs={},  # Options for generator optimizer.
    D_opt_kwargs={},  # Options for discriminator optimizer.
    augment_kwargs=None,  # Options for augmentation pipeline. None = disable.
    loss_kwargs={},  # Options for loss function.
    metrics=[],  # Metrics to evaluate during training.
    random_seed=0,  # Global random seed.
    num_gpus=1,  # Number of GPUs participating in the training.
    rank=0,  # Rank of the current process in [0, num_gpus[.
    batch_size=4,  # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu=4,  # Number of samples processed at a time by one GPU.
    ema_kimg=10,  # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup=0.05,  # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval=None,  # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval=16,  # How often to perform regularization for D? None = disable lazy regularization.
    augment_p=0,  # Initial value of augmentation probability.
    ada_target=None,  # ADA target value. None = fixed p.
    ada_interval=4,  # How often to perform ADA adjustment?
    ada_kimg=500,  # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg=25000,  # Total length of the training, measured in thousands of real images.
    kimg_per_tick=4,  # Progress snapshot interval.
    image_snapshot_ticks=50,  # How often to save image snapshots? None = disable.
    network_snapshot_ticks=50,  # How often to save network snapshots? None = disable.
    resume_pkl=None,  # Network pickle to resume training from.
    resume_kimg=0,  # First kimg to report when resuming training.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    abort_fn=None,  # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn=None,  # Callback function for updating training progress. Called for all ranks.
    uniform_class_labels=False,  # Use uniform class labels for generated images or use the training set distribution?
    disc_on_gen=False,  # Whether to run the discriminator on generated images.
    save_all_snaps=True,  # Whether to save all snapshots or only the best one based on validation average accuracy.
    save_all_fakes=True,  # Whether to save all fake image grids or only the best one based on validation average accuracy.
    autoencoder_kimg=0,  # Number of kimg to train the autoencoder for at the beginning of training.
    autoencoder_patience=0,  # Number of ticks to wait for improvement before stopping the autoencoder training.
    autoencoder_min_delta=0.0,  # Minimum change in validation loss to be considered an improvement for early stopping of the autoencoder.
    judge_model_path=None,  # Path to the judge model for FID/Manifold metrics.
    manifold_interval=1,  # Interval (in ticks) to calculate FID/Manifold metrics.
    manifold_num_images_per_class=200,  # Number of samples per class for FID/Manifold metrics.
    manifold_experiments=1,  # Number of times to repeat the FID/Manifold experiment.
):
    # Initialize.
    start_time = time.time()
    device = torch.device("cuda", rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.

    # Initialize the best validation average accuracy
    best_val_avg_acc = 0.0
    best_val_avg_acc_tick = 0
    best_aa_snapshot_pkl_path = None
    best_aa_fake_grid_path = None

    # Initialize best model based on FID
    best_fid = float("inf")
    best_fid_tick = 0
    best_fid_snapshot_pkl_path = None
    best_fid_fake_grid_path = None

    # Initialize autoencoder early stopping variables
    best_ae_val_loss = float("inf")
    patience_counter = 0

    # We have two ways of specifying a parameter initialization through an AE: setting a
    # fixed number of kimg to train or using early stopping based on validation AE loss
    autoencoder_training_flag = (autoencoder_kimg > 0) or (autoencoder_patience > 0)

    # Load training set.
    if rank == 0:
        print("Loading training set...")
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(
        dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed, shuffle=True, window_size=0.5
    )
    training_set_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=training_set, sampler=training_set_sampler, batch_size=batch_size // num_gpus, **data_loader_kwargs
        )
    )
    # Load validation set
    validation_set = dnnlib.util.construct_class_by_name(
        **validation_set_kwargs
    )  # subclass of training.dataset.Dataset

    if rank == 0:
        print()
        print("Num images: ", len(training_set))
        print("Image shape:", training_set.image_shape)
        print("Label shape:", training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print("Constructing networks...")
    common_kwargs = dict(
        c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels
    )
    # Add output_dim to epilogue_kwargs to perform classification
    D_kwargs.epilogue_kwargs.output_dim = (
        training_set.label_shape[0] if training_set.has_labels and loss_kwargs.classification_weight > 0 else 0
    )
    # Pass label_map to loss class
    loss_kwargs.label_map = (
        training_set.get_label_map() if training_set.has_labels and training_set_kwargs.use_label_map else None
    )

    G = (
        dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    )  # subclass of torch.nn.Module
    D = (
        dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    )  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [("G", G), ("D", D), ("G_ema", G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print("Setting up augmentation...")
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = (
            dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device)
        )  # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex="Loss/signs/real")

    # Distribute across GPUs.
    if rank == 0:
        print(f"Distributing across {num_gpus} GPUs...")
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print("Setting up training phases...")
    loss = dnnlib.util.construct_class_by_name(
        device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs
    )  # subclass of training.loss.Loss
    phases = []
    if autoencoder_training_flag:
        G_opt = dnnlib.util.construct_class_by_name(params=G.parameters(), **G_opt_kwargs)
        D_opt = dnnlib.util.construct_class_by_name(params=D.parameters(), **D_opt_kwargs)
        phases += [dnnlib.EasyDict(name="AE", modules=[G, D], opts=[G_opt, D_opt], interval=1)]
    for name, module, opt_kwargs, reg_interval in [
        ("G", G, G_opt_kwargs, G_reg_interval),
        ("D", D, D_opt_kwargs, D_reg_interval),
    ]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(
                params=module.parameters(), **opt_kwargs
            )  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + "both", modules=[module], opts=[opt], interval=1)]
        else:  # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta**mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(
                module.parameters(), **opt_kwargs
            )  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + "main", modules=[module], opts=[opt], interval=1)]
            phases += [dnnlib.EasyDict(name=name + "reg", modules=[module], opts=[opt], interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    latent_tensor = None
    label_tensor = None
    if rank == 0:
        print("Exporting sample images...")
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, "reals.png"), drange=[0, 255], grid_size=grid_size)
        latent_tensor = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        label_tensor = torch.from_numpy(labels).to(device).split(batch_gpu)
        images = torch.cat(
            [G_ema(z=z, c=c, noise_mode="const").cpu() for z, c in zip(latent_tensor, label_tensor)]
        ).numpy()
        save_image_grid(images, os.path.join(run_dir, "fakes_init.png"), drange=[-1, 1], grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print("Initializing logs...")
    stats_collector = training_stats.Collector(regex=".*")
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None

    # Setup FID/Manifold metrics
    judge_model = None
    if judge_model_path is not None:
        if rank == 0:
            print(f"Loading judge model from {judge_model_path}...")

        # Load judge model
        judge_model = CNN2D_Residual(DatasetMock(), device, HYPERPARAMS)
        judge_model.load_state_dict(torch.load(judge_model_path, map_location=device))
        judge_model.eval()

    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "wt")
        process = None
        use_tensorboard = False
        if use_tensorboard:
            try:
                import subprocess

                import torch.utils.tensorboard as tensorboard

                stats_tfevents = tensorboard.SummaryWriter(run_dir)
                print(f"Launching TensorBoard in {run_dir}...")
                command = ["tensorboard", "--logdir=" + run_dir, "--port=8888"]
                try:
                    # Popen runs the command in the background
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(f"TensorBoard started in the background with PID {process.pid}.")
                except Exception as e:
                    print(f"Could not start TensorBoard: {e}")
            except ImportError as err:
                print("Skipping tfevents export:", err)

    # Train.
    if rank == 0:
        print(f"Training for {total_kimg} kimg...")
        print()
    cur_nimg = resume_kimg * 1000
    autoencoder_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:
        # This consumes time during autoencoder training that can be avoided (we do not need them)
        # Fetch training data.
        with torch.autograd.profiler.record_function("data_fetch"):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]

            # Generate random class vectors
            all_gen_c = []
            if uniform_class_labels:
                # Generate random class vectors following a uniform distribution
                class_ids = np.random.randint(0, training_set.label_dim, size=len(phases) * batch_size)
                for cid in class_ids:
                    vec = np.zeros(training_set.label_dim, dtype=np.float32)
                    vec[cid] = 1.0
                    all_gen_c.append(vec)
            else:
                # Generate random class vectors following the training set distribution
                all_gen_c = [
                    training_set.get_label(np.random.randint(len(training_set)))
                    for _ in range(len(phases) * batch_size)
                ]

            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if phase.name != "AE" and autoencoder_training_flag:
                continue
            if phase.name == "AE" and not autoencoder_training_flag:
                continue
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            for module, opt in zip(phase.modules, phase.opts):
                opt.zero_grad(set_to_none=True)
                module.requires_grad_(True)
            for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(
                    phase=phase.name,
                    real_img=real_img,
                    real_c=real_c,
                    gen_z=gen_z,
                    gen_c=gen_c,
                    gain=phase.interval,
                    cur_nimg=cur_nimg,
                    autoencoder_nimg=autoencoder_nimg,
                    disc_on_gen=disc_on_gen,
                )
            for module in phase.modules:
                module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + "_opt"):
                for module, opt in zip(phase.modules, phase.opts):
                    params = [param for param in module.parameters() if param.grad is not None]
                    if len(params) > 0:
                        flat = torch.cat([param.grad.flatten() for param in params])
                        if num_gpus > 1:
                            torch.distributed.all_reduce(flat)
                            flat /= num_gpus
                        misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                        grads = flat.split([param.numel() for param in params])
                        for param, grad in zip(params, grads):
                            param.grad = grad.reshape(param.shape)
                    opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function("Gema"):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        if autoencoder_training_flag:
            autoencoder_nimg += batch_size
            # Check if we need to disable autoencoder training flag because we have reached the specified number of kimg
            if (autoencoder_kimg > 0) and (autoencoder_nimg >= autoencoder_kimg * 1000):
                autoencoder_training_flag = False
                if rank == 0:
                    print(
                        f"Finished autoencoder training at {autoencoder_nimg / 1000:.1f} kimg out of {autoencoder_kimg} kimg."
                    )
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = (
                np.sign(ada_stats["Loss/signs/real"] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            )
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = cur_nimg >= total_kimg * 1000
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"ae_kimg {training_stats.report0('Progress/autoencoder_kimg', autoencoder_nimg / 1e3):<8.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        fields += [
            f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"
        ]
        training_stats.report0("Timing/total_hours", (tick_end_time - start_time) / (60 * 60))
        training_stats.report0("Timing/total_days", (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(" ".join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print("Aborting...")

        # Evaluate validation set
        if loss_kwargs.classification_weight > 0:
            if rank == 0:
                print("Evaluating validation set...", end="")

            start_time_val = time.time()

            # Create a fresh DataLoader for validation to avoid worker issues
            validation_set_sampler = DistributedSampler(
                validation_set,
                num_replicas=num_gpus,
                rank=rank,
                shuffle=False,
            )
            validation_set_loader = torch.utils.data.DataLoader(
                dataset=validation_set,
                sampler=validation_set_sampler,
                batch_size=batch_size // num_gpus,
                **data_loader_kwargs,
            )

            with torch.no_grad():
                for val_real_img, val_real_c in validation_set_loader:
                    val_real_img = (val_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
                    val_real_c = val_real_c.to(device).split(batch_gpu)

                    for real_img, real_c in zip(val_real_img, val_real_c):
                        loss.evaluate_discriminator(real_img=real_img, real_c=real_c, batch_size=batch_gpu)
                        loss.evaluate_autoencoder(real_img=real_img, real_c=real_c, batch_size=batch_gpu)

            end_time_val = time.time()
            training_stats.report0("Timing/val_sec", end_time_val - start_time_val)

            if rank == 0:
                print(f" Finished! ({end_time_val - start_time_val:.2f}s)")

        # Evaluate Manifold Metrics (FID, Precision, Recall)
        if (judge_model is not None) and (cur_tick % manifold_interval == 0):
            if rank == 0:
                print(f"Evaluating Manifold Metrics ({manifold_experiments} experiments per gpu)...", end="")

            start_time_manifold = time.time()

            class_labels = list(training_set.get_label_map().values())
            total_images = manifold_num_images_per_class * len(class_labels)

            # Distribute image generation across GPUs
            images_per_gpu = total_images // num_gpus
            remainder = total_images % num_gpus

            # Each rank generates a portion of images (distribute remainder to first ranks)
            if rank < remainder:
                rank_start_idx = rank * (images_per_gpu + 1)
                rank_total_images = images_per_gpu + 1
            else:
                rank_start_idx = rank * images_per_gpu + remainder
                rank_total_images = images_per_gpu

            # Construct the label tensor for this rank's portion
            labels_all = torch.zeros([total_images, G.c_dim], device=device)
            for i in range(len(class_labels)):
                labels_all[
                    i * manifold_num_images_per_class : (i + 1) * manifold_num_images_per_class, class_labels[i]
                ] = 1

            # Get only the labels needed for this rank
            rank_labels = labels_all[rank_start_idx : rank_start_idx + rank_total_images]

            # Generate this rank's portion of synthetic images in batches to avoid OOM
            images_list = []
            num_batches = (rank_total_images + batch_gpu - 1) // batch_gpu  # Ceiling division

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_gpu
                end_idx = min(start_idx + batch_gpu, rank_total_images)
                current_batch_size = end_idx - start_idx

                # Generate latents and labels for this batch
                z_batch = torch.randn([current_batch_size, G.z_dim], device=device)
                c_batch = rank_labels[start_idx:end_idx]

                # Generate images (keep in GPU)
                with torch.no_grad():
                    batch_images = G_ema(z=z_batch, c=c_batch, noise_mode="const")
                    batch_images = batch_images.permute(0, 2, 3, 1)  # NCHW -> NHWC

                images_list.append(batch_images)
                del z_batch, c_batch, batch_images

            # Concatenate all batches for this rank
            rank_images = torch.cat(images_list, dim=0)
            del images_list

            # Gather images from all ranks if using multiple GPUs
            if num_gpus > 1:
                # Prepare list to gather all tensors
                gathered_images = [torch.zeros_like(rank_images) for _ in range(num_gpus)]
                torch.distributed.all_gather(gathered_images, rank_images)

                # Concatenate in correct order, handling different sizes
                all_images_parts = []
                for gpu_rank in range(num_gpus):
                    if gpu_rank < remainder:
                        expected_size = images_per_gpu + 1
                    else:
                        expected_size = images_per_gpu
                    all_images_parts.append(gathered_images[gpu_rank][:expected_size])

                images = torch.cat(all_images_parts, dim=0)
                del gathered_images, all_images_parts
            else:
                images = rank_images

            del rank_images
            torch.cuda.empty_cache()

            # Move images to CPU for metric calculations
            images = images.cpu()
            # Save the samples in a dictionary, one key for each class
            pool = dict()
            for i, class_idx in enumerate(class_labels):
                pool[class_idx] = images[
                    i * manifold_num_images_per_class : (i + 1) * manifold_num_images_per_class, :, :, :
                ]

            end_time_image_gen = time.time()
            # Report time taken for image generation only from rank 0. Both GPUs collaborated in generation.
            training_stats.report0("Metrics/image_gen_sec", end_time_image_gen - start_time_manifold)

            for _ in range(manifold_experiments):
                # Generate and extract features
                start_time_real_features = time.time()
                real_features = get_train_features(
                    dataset=training_set,
                    num_images_per_class=manifold_num_images_per_class,
                    judge_model=judge_model,
                    batch_size=512,  # use a big batch_size and fixed
                    num_workers=data_loader_kwargs.num_workers,  # reutilize the num_workers parameter for dataloader
                    device=device,
                )
                end_time_real_features = time.time()
                fake_features = get_synthetic_features(
                    pool=pool,
                    classes=class_labels,
                    num_images_per_class=manifold_num_images_per_class,
                    judge_model=judge_model,
                    batch_size=512,  # use a big batch_size and fixed
                    device=device,
                )
                end_time_fake_features = time.time()

                start_fid_time = time.time()
                fid_score = calculate_fid(real_features, fake_features)
                end_fid_time = time.time()
                precision, recall = compute_precision_recall(real_features, fake_features, num_gpus=1)
                end_precision_recall_time = time.time()

                # Report from all ranks, at the end, we compute num_gpus * manifold_experiments values
                training_stats.report("Metrics/fid", fid_score)
                training_stats.report("Metrics/precision", precision)
                training_stats.report("Metrics/recall", recall)

                training_stats.report("Timing/real_features_sec", end_time_real_features - start_time_real_features) # fmt: skip
                training_stats.report("Timing/fake_features_sec", end_time_fake_features - end_time_real_features)

                training_stats.report("Timing/fid_sec", end_fid_time - start_fid_time)
                training_stats.report("Timing/precision_recall_sec", end_precision_recall_time - end_fid_time)

            end_time_manifold = time.time()
            if rank == 0:
                print(f"Finished! ({end_time_manifold - start_time_manifold:.2f}s)")

        # Collect statistics.
        for phase in phases:
            value = []
            if phase.start_event is not None and phase.end_event is not None:
                phase.end_event.synchronize()
                try:
                    value = phase.start_event.elapsed_time(phase.end_event)
                except RuntimeError as e:
                    value = float("nan")
            training_stats.report0("Timing/" + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Early stopping for autoencoder
        if autoencoder_patience > 0 and autoencoder_training_flag:
            if "Loss/AE/val/loss" in stats_dict:
                current_ae_val_loss = stats_dict["Loss/AE/val/loss"].mean
                if current_ae_val_loss < best_ae_val_loss - autoencoder_min_delta:
                    best_ae_val_loss = current_ae_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= autoencoder_patience:
                    autoencoder_training_flag = False

        # Update best validation average accuracy
        if loss_kwargs.classification_weight > 0 and (done or cur_tick % image_snapshot_ticks == 0):
            val_stats = {key: [val.mean] for key, val in stats_dict.items() if key.startswith("Accuracy/")}
            class_labels = [k.split("/")[-1] for k in val_stats.keys()]
            _, _, _, _, val_avg_acc, _ = compute_avg_accuracy(val_stats, clean_nan=True, class_labels=class_labels)
            if val_avg_acc.size > 0:
                val_avg_acc = val_avg_acc[0]
                if val_avg_acc > best_val_avg_acc:
                    best_val_avg_acc = val_avg_acc
                    best_val_avg_acc_tick = cur_nimg // 1000

        # Update best FID score
        if judge_model is not None and (done or cur_tick % image_snapshot_ticks == 0):
            fid_scores = stats_dict.get("Metrics/fid", None)
            if fid_scores is not None:
                # Compute average FID from all experiments in this tick
                fid_avg = fid_scores.mean
                if fid_avg < best_fid:
                    best_fid = fid_avg
                    best_fid_tick = cur_nimg // 1000

        snapshot_pkl = None
        snapshot_data = None

        # This requires that tick interval is at least 1 kimg
        is_aa_best_so_far = cur_nimg // 1000 == best_val_avg_acc_tick  # Always true at the beginning
        is_fid_best_so_far = cur_nimg // 1000 == best_fid_tick

        # Save image snapshot.
        if (
            (rank == 0)
            and (image_snapshot_ticks is not None)
            and (done or cur_tick % image_snapshot_ticks == 0)
            and (save_all_fakes or is_aa_best_so_far or is_fid_best_so_far or done)
        ):
            images = torch.cat(
                [G_ema(z=z, c=c, noise_mode="const").cpu() for z, c in zip(latent_tensor, label_tensor)]
            ).numpy()
            save_image_grid(
                images, os.path.join(run_dir, f"fakes{cur_nimg//1000:06d}.png"), drange=[-1, 1], grid_size=grid_size
            )
            current_fake_path = os.path.join(run_dir, f"fakes{cur_nimg//1000:06d}")
            best_aa_fake_grid_path = update_best_paths(
                is_aa_best_so_far,
                save_all_fakes,
                current_fake_path,
                best_aa_fake_grid_path,
                best_fid_fake_grid_path,
                [".png", ".raw"],
                metric_name="AA fake grid",
                verbose=False,
            )
            best_fid_fake_grid_path = update_best_paths(
                is_fid_best_so_far,
                save_all_fakes,
                current_fake_path,
                best_fid_fake_grid_path,
                best_aa_fake_grid_path,
                [".png", ".raw"],
                metric_name="FID fake grid",
                verbose=False,
            )

        # Save network snapshot.
        if (
            network_snapshot_ticks is not None
            and (cur_tick % network_snapshot_ticks == 0 or done)
            and (save_all_snaps or is_aa_best_so_far or is_fid_best_so_far or done)
        ):
            snapshot_data = dict(
                G=G, D=D, G_ema=G_ema, augment_pipe=augment_pipe, training_set_kwargs=dict(training_set_kwargs)
            )
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if num_gpus > 1:
                        misc.check_ddp_consistency(value, ignore_regex=r".*\.[^.]+_(avg|ema)")
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value.cpu()
                del value  # conserve memory

            # Remove previous best snapshot if needed
            snapshot_base = os.path.join(run_dir, f"network-snapshot-{cur_nimg//1000:06d}")
            if rank == 0:
                best_aa_snapshot_pkl_path = update_best_paths(
                    is_aa_best_so_far,
                    save_all_snaps,
                    snapshot_base,
                    best_aa_snapshot_pkl_path,
                    best_fid_snapshot_pkl_path,
                    [".pkl"],
                    metric_name="AA snapshot",
                    verbose=False,
                )
                best_fid_snapshot_pkl_path = update_best_paths(
                    is_fid_best_so_far,
                    save_all_snaps,
                    snapshot_base,
                    best_fid_snapshot_pkl_path,
                    best_aa_snapshot_pkl_path,
                    [".pkl"],
                    metric_name="FID snapshot",
                    verbose=False,
                )

                with open(snapshot_base + ".pkl", "wb") as f:
                    pickle.dump(snapshot_data, f)

        del snapshot_data  # conserve memory

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + "\n")
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f"Metrics/{name}", value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print("Terminating the TensorBoard process...")
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)  # Wait for the process to terminate gracefully
                print("TensorBoard terminated gracefully.")
            except subprocess.TimeoutExpired:
                print("TensorBoard did not terminate in time. Forcing termination.")
                process.kill()
                process.wait()  # Ensure the process is cleaned up
        else:
            print("TensorBoard process not found.")
        print()
        print("Exiting...")


# ----------------------------------------------------------------------------
