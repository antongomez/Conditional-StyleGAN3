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
import subprocess
import time

import numpy as np
import psutil
import torch
from torch.utils.data.distributed import DistributedSampler

import dnnlib
import legacy
from torch_utils import misc, training_stats
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
from visualization_utils import compute_avg_accuracy

# ----------------------------------------------------------------------------


def safe_remove_files(file_path, extensions):
    """
    Safely remove files with given extensions.

    Args:
        file_path: Base path of files to remove (without extension)
        extensions: List of file extensions to remove (e.g., ['.png', '.raw'] or ['.pkl'])
    """
    if file_path is not None:
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
        files_extensions: List of file extensions to remove
        metric_name: Name of the metric for logging
        verbose: Whether to print removal messages

    Returns:
        Updated best path or None if not applicable
    """
    if is_best and not save_all:
        if verbose:
            print(f"Removing previous best {metric_name}:", old_best_path)

        safe_remove_files(old_best_path, files_extensions)
        new_best_path = current_path

        if verbose:
            print(f"Updated best {metric_name} path to:", new_best_path)

        return new_best_path
    return old_best_path


# ----------------------------------------------------------------------------


def training_loop_classifier(
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

    # Construct classifier network.
    if rank == 0:
        print("Constructing classifier network...")
    common_kwargs = dict(
        c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels
    )
    # Add output_dim to epilogue_kwargs to perform classification
    assert training_set.has_labels, "Training set must have labels for classification."
    D_kwargs.epilogue_kwargs.output_dim = training_set.label_shape[0]
    # Rebuild loss_kwargs to avoid make changes in train.py
    loss_kwargs = dnnlib.EasyDict(class_name="training.loss.ClassifierLoss")
    loss_kwargs.label_map = (
        training_set.get_label_map() if training_set.has_labels and training_set_kwargs.use_label_map else None
    )

    classifier = (
        dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    )  # subclass of torch.nn.Module

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        misc.copy_params_and_buffers(resume_data["classifier"], classifier, require_all=False)

    # Print network summary.
    if rank == 0:
        img = torch.empty(
            [batch_gpu, training_set.num_channels, training_set.resolution, training_set.resolution], device=device
        )
        c = torch.empty([batch_gpu, training_set.label_dim], device=device)
        misc.print_module_summary(classifier, [img, c])

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
    for module in [classifier, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup optimizer and loss.
    if rank == 0:
        print("Setting up optimizer and loss...")
    loss = dnnlib.util.construct_class_by_name(
        device=device, classifier=classifier, augment_pipe=augment_pipe, **loss_kwargs
    )  # subclass of training.loss.Loss
    optimizer = dnnlib.util.construct_class_by_name(
        params=classifier.parameters(), **D_opt_kwargs
    )  # subclass of torch.optim.Optimizer
    start_event = None
    end_event = None
    if rank == 0:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    # Initialize logs.
    if rank == 0:
        print("Initializing logs...")
    stats_collector = training_stats.Collector(regex=".*")
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None

    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "wt")
        process = None
        use_tensorboard = False
        if use_tensorboard:
            try:
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
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function("data_fetch"):
            batch_real_img, batch_real_c = next(training_set_iterator)
            batch_real_img = (batch_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            batch_real_c = batch_real_c.to(device).split(batch_gpu)

        if start_event is not None:
            start_event.record(torch.cuda.current_stream(device))

        # Train classifier.
        optimizer.zero_grad(set_to_none=True)
        classifier.requires_grad_(True)

        for real_img, real_c in zip(batch_real_img, batch_real_c):
            loss.accumulate_gradients(real_img=real_img, real_c=real_c)

        classifier.requires_grad_(False)

        # Update weights.
        with torch.autograd.profiler.record_function("classifier_opt"):
            params = [param for param in classifier.parameters() if param.grad is not None]
            if len(params) > 0:
                flat = torch.cat([param.grad.flatten() for param in params])
                if num_gpus > 1:
                    torch.distributed.all_reduce(flat)
                    flat /= num_gpus
                misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                grads = flat.split([param.numel() for param in params])
                for param, grad in zip(params, grads):
                    param.grad = grad.reshape(param.shape)
            optimizer.step()

        # Training step done.
        if end_event is not None:
            end_event.record(torch.cuda.current_stream(device))

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
                    loss.evaluate_classifier(real_img=real_img, real_c=real_c, batch_size=batch_gpu)

        end_time_val = time.time()
        training_stats.report0("Timing/val_sec", end_time_val - start_time_val)

        # Collect statistics.
        value = []
        if start_event is not None and end_event is not None:
            end_event.synchronize()
            try:
                value = start_event.elapsed_time(end_event)
            except RuntimeError as e:
                value = float("nan")
        training_stats.report0("Timing/training_step", value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update best validation average accuracy
        if done or cur_tick % image_snapshot_ticks == 0:
            val_stats = {key: [val.mean] for key, val in stats_dict.items() if key.startswith("Accuracy/")}
            class_labels = [k.split("/")[-1] for k in val_stats.keys()]
            _, _, _, _, val_avg_acc, _ = compute_avg_accuracy(val_stats, clean_nan=True, class_labels=class_labels)
            if val_avg_acc.size > 0:
                val_avg_acc = val_avg_acc[0]
                if val_avg_acc > best_val_avg_acc:
                    best_val_avg_acc = val_avg_acc
                    best_val_avg_acc_tick = cur_nimg // 1000

        snapshot_data = None

        # This requires that tick interval is at least 1 kimg
        is_aa_best_so_far = cur_nimg // 1000 == best_val_avg_acc_tick  # Always true at the beginning

        # Save network snapshot.
        if (
            network_snapshot_ticks is not None
            and (cur_tick % network_snapshot_ticks == 0 or done)
            and (save_all_snaps or is_aa_best_so_far or done)
        ):
            snapshot_data = dict(
                classifier=classifier, augment_pipe=augment_pipe, training_set_kwargs=dict(training_set_kwargs)
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
                    [".pkl"],
                    metric_name="AA snapshot",
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
