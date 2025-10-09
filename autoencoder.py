import copy
import json
import os
import pickle
import re
import time

import numpy as np
import PIL.Image
import psutil
import torch
from torch.utils.data.distributed import DistributedSampler

import dnnlib
import legacy
from metrics import metric_main
from torch_utils import misc, training_stats
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix

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


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    if C == 5:
        img = img[:, :, [2, 1, 0]]  # Select rgb channels
        C = 3

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], "L").save(fname)
    if C == 3:
        PIL.Image.fromarray(img, "RGB").save(fname)


# ----------------------------------------------------------------------------


def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(
            class_name="training.dataset.ImageFolderDataset", path=data, use_labels=True, max_size=None, xflip=False
        )
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution  # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels  # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj)  # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        print(f"--data: {err}")
        exit(1)


def build_datasets_kwargs(opts):
    """Build training set kwargs."""
    # Training set.
    training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    validation_set_kwargs, _ = (
        init_dataset_kwargs(data=opts.data_val)
        if opts.data_val
        else dnnlib.EasyDict(path="", use_labels=True, max_size=None, xflip=False)
    )
    if opts.cond and not training_set_kwargs.use_labels:
        print("--cond=True requires labels specified in dataset.json")
        exit(1)
    training_set_kwargs.use_labels = opts.cond
    training_set_kwargs.xflip = opts.mirror
    training_set_kwargs.use_label_map = opts.use_label_map
    validation_set_kwargs.use_labels = opts.cond
    validation_set_kwargs.xflip = opts.mirror
    validation_set_kwargs.use_label_map = opts.use_label_map
    return training_set_kwargs, validation_set_kwargs, dataset_name


def initialize_parameters(opts, cnn21=True):
    # Initialize training set and validation set kwargs
    training_set_kwargs, validation_set_kwargs, dataset_name = build_datasets_kwargs(opts)
    # Initialize data loader kwargs
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)
    data_loader_kwargs.num_workers = opts.workers
    # Initialize network kwargs
    G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    if cnn21:
        D_kwargs = dnnlib.EasyDict(
            class_name="training.cnn21.CNN21",
        )
    else:
        D_kwargs = dnnlib.EasyDict(
            class_name="training.networks_stylegan2.Discriminator",
            block_kwargs=dnnlib.EasyDict(),
            mapping_kwargs=dnnlib.EasyDict(),
            epilogue_kwargs=dnnlib.EasyDict(),
        )

        D_kwargs.block_kwargs.freeze_layers = opts.freezed
        D_kwargs.epilogue_kwargs.mbstd_num_channels = opts.mbstd_num_channels
        D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group

        D_kwargs.channel_base = opts.cbase
        D_kwargs.channel_max = opts.cmax

    G_opt_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", betas=[0, 0.99], eps=1e-8)
    D_opt_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", betas=[0, 0.99], eps=1e-8)
    # Initialize loss kwargs
    loss_kwargs = dnnlib.EasyDict(class_name="training.loss.StyleGAN2Loss")
    loss_kwargs.r1_gamma = opts.gamma
    loss_kwargs.class_weight = opts.cls_weight
    loss_kwargs.style_mixing_prob = 0 # Disable style mixing for autoencoder

    # Other parameters
    G_kwargs.channel_base = opts.cbase
    G_kwargs.channel_max = opts.cmax
    G_kwargs.mapping_kwargs.num_layers = (
        (8 if opts.cfg == "stylegan2" else 2) if opts.map_depth is None else opts.map_depth
    )
        

    G_opt_kwargs.lr = (0.002 if opts.cfg == "stylegan2" else 0.0025) if opts.glr is None else opts.glr
    D_opt_kwargs.lr = opts.dlr

    # Base configuration.
    ema_kimg = opts.batch * 10 / 32
    if opts.cfg == "stylegan2":
        G_kwargs.class_name = "training.networks_stylegan2.Generator"
        loss_kwargs.style_mixing_prob = 0.9  # Enable style mixing regularization.
        loss_kwargs.pl_weight = 2  # Enable path length regularization.
        G_reg_interval = 4  # Enable lazy regularization for G.
        G_kwargs.fused_modconv_default = (
            "inference_only"  # Speed up training by using regular convolutions instead of grouped convolutions.
        )
        loss_kwargs.pl_no_weight_grad = (
            True  # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.
        )
    else:
        G_kwargs.class_name = "training.networks_stylegan3.Generator"
        G_kwargs.magnitude_ema_beta = 0.5 ** (opts.batch / (20 * 1e3))
        if opts.cfg == "stylegan3-r":
            G_kwargs.conv_kernel = 1  # Use 1x1 convolutions.
            G_kwargs.channel_base *= 2  # Double the number of feature maps.
            G_kwargs.channel_max *= 2
            G_kwargs.use_radial_filters = True  # Use radially symmetric downsampling filters.
            loss_kwargs.blur_init_sigma = 10  # Blur the images seen by the discriminator.
            loss_kwargs.blur_fade_kimg = opts.batch * 200 / 32  # Fade out the blur during the first N kimg.

    # Augmentation.
    if opts.aug != "noaug":
        augment_kwargs = dnnlib.EasyDict(
            class_name="training.augment.AugmentPipe",
            xflip=1,
            rotate90=1,
            xint=1,
            scale=1,
            rotate=1,
            aniso=1,
            xfrac=1,
            brightness=0,
            contrast=0,
            lumaflip=0,
            hue=0,
            saturation=0,
        )
        if opts.aug == "ada":
            ada_target = opts.target
        if opts.aug == "fixed":
            augment_p = opts.p

    return (
        training_set_kwargs,
        validation_set_kwargs,
        data_loader_kwargs,
        G_kwargs,
        D_kwargs,
        G_opt_kwargs,
        D_opt_kwargs,
        loss_kwargs,
        augment_kwargs if opts.aug != "noaug" else None,
    )


def main():
    # Parse command line arguments.
    opts = dnnlib.EasyDict(
        cond=True,
        mirror=False,
        use_label_map=True,
        data="data/oitaven/oitaven_train.zip",
        data_val="data/oitaven/oitaven_val.zip",
        dataset_name="oitaven_train",
        batch=32,
        gpus=1,
        workers=0,
        cls_weight=0,
        gamma=0.125,
        cbase=32768,
        cmax=512,
        cfg="stylegan3-t",
        freezed=0,
        mbstd_num_channels=0,
        mbstd_group=4,
        glr=None,
        dlr=0.002,
        map_depth=None,
        aug="noaug",
        target=0.6,
        p=0.2,
        kimg=150.5,
        kimg_per_tick=10,
        image_snapshot_ticks=1,
        network_snapshot_ticks=1,
    )

    # Initialize. Simulate a distributed environment with one GPU.
    rank = 0
    num_gpus = opts.gpus
    random_seed = 42
    start_time = time.time()
    device = torch.device("cuda", rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = True  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.

    batch_size = opts.batch
    batch_gpu = batch_size // num_gpus
    outdir = f"training-runs-autoencoder/{os.path.splitext(os.path.basename(opts.data))[0].upper()}"
    desc = f"{opts.cfg:s}-{opts.dataset_name:s}-gpus{opts.gpus:d}-batch{opts.batch:d}-gamma{opts.gamma:g}"
    total_kimg = opts.kimg
    resume_kimg = 0
    progress_fn = None
    kimg_per_tick = opts.kimg_per_tick
    image_snapshot_ticks = opts.image_snapshot_ticks
    network_snapshot_ticks = opts.network_snapshot_ticks

    # Other parameters.
    ema_kimg = 10  # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup = 0.05  # EMA ramp-up coefficient. None = no rampup.
    ada_interval = 4  # How often to perform ADA adjustment?
    ada_kimg = 500  # ADA adjustment speed.
    abort_fn = None  # Callback function for determining whether to abort training. Must return consistent results across ranks.

    # Optional arguments
    uniform_class_labels = True
    disc_on_gen = True

    # Which discriminator to use
    cnn21 = False
    if cnn21:
        print("Training autoencoder with CNN21")
    else:
        print("Training autoencoder with StyleGAN3 discriminator")

    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join(outdir, f"{cur_run_id:05d}-{desc}")
    assert not os.path.exists(run_dir)

    print("Creating output directory...")
    os.makedirs(run_dir)

    # Initialize parameters
    (
        training_set_kwargs,
        validation_set_kwargs,
        data_loader_kwargs,
        G_kwargs,
        D_kwargs,
        G_opt_kwargs,
        D_opt_kwargs,
        loss_kwargs,
        augment_kwargs,
    ) = initialize_parameters(opts, cnn21)

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
    validation_set_sampler = DistributedSampler(
        validation_set,
        num_replicas=num_gpus,
        rank=rank,
        shuffle=False,
    )  # no need to shuffle nor an infinite sampler for validation
    validation_set_iterator = torch.utils.data.DataLoader(
        dataset=validation_set,
        sampler=validation_set_sampler,
        batch_size=batch_size // num_gpus,
        **data_loader_kwargs,
    )

    if rank == 0:
        print()
        print("Num images: ", len(training_set))
        print("Image shape:", training_set.image_shape)
        print("Label shape:", training_set.label_shape)
        print()
        print("Num validation images: ", len(validation_set))
        print("Validation image shape:", validation_set.image_shape)
        print("Validation label shape:", validation_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print("Constructing networks...")
    common_kwargs = dict(
        c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels
    )
    # Add output_dim to epilogue_kwargs to perform classification
    if not cnn21:
        D_kwargs.epilogue_kwargs.output_dim = (
            training_set.label_shape[0] if training_set.has_labels and loss_kwargs.class_weight > 0 else 0
        )
    # Pass label_map to loss class
    loss_kwargs.label_map = (
        training_set.get_label_map() if training_set.has_labels and training_set_kwargs.use_label_map else None
    )
    # Set autoencoder_kimg
    loss_kwargs.autoencoder_kimg = opts.kimg

    G = (
        dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    )  # subclass of torch.nn.Module
    if cnn21:
        D = (
            dnnlib.util.construct_class_by_name(**D_kwargs).train().requires_grad_(False).to(device)
        )  # subclass of torch.nn.Module
    else:
        D = (
            dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
        )  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])
        del z, c, img

    # Setup augmentation.
    if rank == 0:
        print("Setting up augmentation...")
    augment_pipe = None
    ada_stats = None
    augment_p = opts.p if opts.aug == "fixed" else 0.0
    ada_target = opts.target if opts.aug == "ada" else None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = (
            dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device)
        )  # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex="Loss/signs/real")

    # Set up training phases.
    if rank == 0:
        print("Setting up training phases...")
    loss = dnnlib.util.construct_class_by_name(
        device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs
    )  # subclass of training.loss.Loss
    phases = [dnnlib.EasyDict(name="AE", module=None, opt=None, interval=1)]  # Autoencoder phase
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print("Exporting sample images...")
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, "reals.png"), drange=[0, 255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        gen_images = torch.cat([G_ema(z=z, c=c, noise_mode="const").cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        save_image_grid(gen_images, os.path.join(run_dir, "fakes_init.png"), drange=[-1, 1], grid_size=grid_size)

        # Prepare real images as a tuple of torch tensors to feed the autoencoder
        real_images_ae = (torch.from_numpy(images).to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
        real_labels_ae = torch.from_numpy(labels).to(device).split(batch_gpu)



    # Initialize logs.
    if rank == 0:
        print("Initializing logs...")
    stats_collector = training_stats.Collector(regex=".*")
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None

    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "wt")

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
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            # all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            # all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]

            # # Generate random class vectors
            # all_gen_c = []
            # if uniform_class_labels:
            #     # Generate random class vectors following a uniform distribution
            #     class_ids = np.random.randint(0, training_set.label_dim, size=len(phases) * batch_size)
            #     for cid in class_ids:
            #         vec = np.zeros(training_set.label_dim, dtype=np.float32)
            #         vec[cid] = 1.0
            #         all_gen_c.append(vec)
            # else:
            #     # Generate random class vectors following the training set distribution
            #     all_gen_c = [
            #         training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)
            #     ]

            # all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            # all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        G_opt = dnnlib.util.construct_class_by_name(params=G.parameters(), **G_opt_kwargs)
        D_opt = dnnlib.util.construct_class_by_name(params=D.parameters(), **D_opt_kwargs)
        for phase in phases:
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            # Accumulate gradients. In the case of the autoencoder phase, we need to update both G and D
            G_opt.zero_grad(set_to_none=True)
            D_opt.zero_grad(set_to_none=True)
            G.requires_grad_(True)
            D.requires_grad_(True)
            for real_img, real_c in zip(phase_real_img, phase_real_c):
                loss.accumulate_gradients(
                    phase=phase.name,
                    real_img=real_img,
                    real_c=real_c,
                    gen_z=None,  # will not be used
                    gen_c=None,  # will not be used
                    gain=phase.interval,  # 1
                    cur_nimg=cur_nimg,  # update to avoid counting iterations when trainign autoencoder
                    disc_on_gen=disc_on_gen,
                )
            G.requires_grad_(False)
            D.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + "_opt"):
                for module, opt in [(G, G_opt), (D, D_opt)]:
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
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            print(
                f"ADA target: {ada_target:.4f}, sign: {ada_stats['Loss/signs/real']:.4f}, batch: {batch_size}, interval: {ada_interval}, kimg: {ada_kimg}"
            )
            adjust = (
                np.sign(ada_stats["Loss/signs/real"] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            )
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))
            print(f"ADA adjust: {adjust:.6f}, p = {augment_pipe.p.item():.6f}")

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

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            latent_images = [
                D(img, label, update_emas=False, return_latents=True)[0]
                for img, label in zip(real_images_ae, real_labels_ae)
            ]
            gen_images_ema = torch.cat(
                [G_ema(z=z, c=c, noise_mode="const").cpu() for z, c in zip(latent_images, real_labels_ae)]
            ).numpy()
            gen_images = torch.cat(
                [G(z=z, c=c, noise_mode="const").cpu() for z, c in zip(latent_images, real_labels_ae)]
            ).numpy()
            save_image_grid(
                gen_images_ema,
                os.path.join(run_dir, f"fakesEMA{cur_nimg//1000:06d}.png"),
                drange=[-1, 1],
                grid_size=grid_size,
            )
            save_image_grid(
                gen_images, os.path.join(run_dir, f"fakes{cur_nimg//1000:06d}.png"), drange=[-1, 1], grid_size=grid_size
            )

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
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
            snapshot_pkl = os.path.join(run_dir, f"network-snapshot-{cur_nimg//1000:06d}.pkl")
            if rank == 0:
                with open(snapshot_pkl, "wb") as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate validation set.
        if rank == 0:
            print("Evaluating validation set...", end=" ")
        with torch.no_grad():
            for val_real_img, val_real_c in validation_set_iterator:
                val_real_img = (val_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
                val_real_c = val_real_c.to(device).split(batch_gpu)

                for real_img, real_c in zip(val_real_img, val_real_c):
                    loss.evaluate_autoencoder(real_img=real_img, real_c=real_c, batch_size=batch_gpu)
        if rank == 0:
            print("Finished!")
        # Evaluate metrics.
        # None
        del snapshot_data  # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0("Timing/" + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

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
        print("Exiting...")


if __name__ == "__main__":
    main()
