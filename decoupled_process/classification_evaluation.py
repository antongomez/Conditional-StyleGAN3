# This code is part of the EMViT-DDPM project and was authored by Victor Barreiro.

##########################################################################################
# ------------------------------------ GLOBAL IMPORTS ----------------------------------#
##########################################################################################


# Basic SO imports
import argparse
import logging
import math
import multiprocessing
import os
import pickle
import random
import sys
import time

# Basic ML imports
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Custom imports
from data_managment_hsiml.data_managment_hsiml import (
    read_raw,
    read_pgm,
    read_seg,
    read_seg_centers,
    select_training_samples_seg,
    HyperDataset,
)

from classifiers.networks import (
    CNN21,
    train,
    test_without_segments,
    print_train_history,
)
from test_function import test

from diffusion.diffusion_generation import generate_diffusion_samples, show_samples
from diffusion.diffusion_datasets import DatasetAugmentedBalancer, DatasetSynthtic


# Time control
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


##########################################################################################
# -------------------------------- TERMINAL PARAMETERS ----------------------------------#
##########################################################################################

# Main parameters
parser = argparse.ArgumentParser(description="Hyperspectral image classification")
parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
parser.add_argument("--patch_size", type=int, default=32, help="Patch size")
parser.add_argument("--normalize", type=int, default=0, help="Normalization applied")
parser.add_argument("--epochs", type=int, default=100, help="Epochs for the classifier")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size for the classifier")
parser.add_argument(
    "--classifier",
    type=str,
    default="CNN21",
    help="Classifier to use. {CNN21, SwinTransformer, ViT}",
)

# Diffusion architecture parameters
parser.add_argument("--model_path", type=str, default="None", help="Model to use")
parser.add_argument("--type", type=str, default="DiT_V_1", help="Diffusion architecture to use")
parser.add_argument(
    "--selection",
    type=int,
    default=0,
    help="The samples would be selected by a CNN21 classifier.",
)

# Diffusion parameters
parser.add_argument(
    "--cfg_scale",
    type=float,
    default=1,
    help="cfg_scale for noise generation in the diffusion model",
)
parser.add_argument(
    "--num_sampling_steps",
    type=int,
    default=250,
    help="num_sampling_steps for noise generation in the diffusion model",
)
parser.add_argument(
    "--synthetic_normalization",
    type=int,
    default=0,
    help="Normalization for the synthetic generation: 1 for True, 0 for False",
)
parser.add_argument("--pool_size", type=int, default=200, help="Pool size for the generated images")
parser.add_argument(
    "--pool_path",
    type=str,
    default="None",
    help="Path to the pool of images. If there is one to use in place to generate it.",
)

# Augmentation parameters
parser.add_argument(
    "--none",
    type=int,
    default=1,
    help="None generation for the balanced generation: 1 for True, 0 for False",
)
parser.add_argument(
    "--balanced",
    type=int,
    default=1,
    help="Balanced generation for the classifier: 1 for True, 0 for False",
)
parser.add_argument(
    "--synthetic",
    type=int,
    default=1,
    help="Synthetic generation for the balanced generation: 1 for True, 0 for False",
)

##########################################################################################
# ------------------------------------ GPU ACCES CHECK ----------------------------------#
##########################################################################################

GPU = True  # Could be disable for resarch proposes but the requiring time to execute without GPU is too high
if GPU:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda = True
else:
    device = torch.device("cpu")

logger.info("Device: %s", device)

##########################################################################################
# ------------------------------ SEEDs FOR REPRODUCIBILITY ------------------------------#
##########################################################################################

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if cuda == False:
    torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(SEED)
else:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##########################################################################################
# -------------------------------------- DATA LOADEING ----------------------------------#
##########################################################################################

logger.info("Loading data")
if parser.parse_args().dataset == "eiras":
    DATASET = "../images/eiras_dam.raw"
    GT = "../images/eiras_dam.pgm"
    SEG = "../images/seg_eiras_wp.raw"
    CENTER = "../images/seg_eiras_wp_centers.raw"
elif parser.parse_args().dataset == "oitaven":
    DATASET = "../images/oitaven_river.raw"
    GT = "../images/oitaven_river.pgm"
    SEG = "../images/seg_oitaven_wp.raw"
    CENTER = "../images/seg_oitaven_wp_centers.raw"
elif parser.parse_args().dataset == "ermidas":
    DATASET = "../images/ermidas_creek.raw"
    GT = "../images/ermidas_creek.pgm"
    SEG = "../images/seg_ermidas_wp.raw"
    CENTER = "../images/seg_ermidas_wp_centers.raw"
elif parser.parse_args().dataset == "ferreiras":
    DATASET = "../images/ferreiras_river.raw"
    GT = "../images/ferreiras_river.pgm"
    SEG = "../images/seg_ferreiras_wp.raw"
    CENTER = "../images/seg_ferreiras_wp_centers.raw"
elif parser.parse_args().dataset == "ulla":
    DATASET = "../images/ulla_river.raw"
    GT = "../images/ulla_river.pgm"
    SEG = "../images/seg_ulla_wp.raw"
    CENTER = "../images/seg_ulla_wp_centers.raw"
elif parser.parse_args().dataset == "mera":
    DATASET = "../images/mera_river.raw"
    GT = "../images/mera_river.pgm"
    SEG = "../images/seg_mera_wp.raw"
    CENTER = "../images/seg_mera_wp_centers.raw"
elif parser.parse_args().dataset == "xesta":
    DATASET = "../images/xesta_basin.raw"
    GT = "../images/xesta_basin.pgm"
    SEG = "../images/seg_xesta_wp.raw"
    CENTER = "../images/seg_xesta_wp_centers.raw"
elif parser.parse_args().dataset == "mestas":
    DATASET = "../images/das_mestas_river.raw"
    GT = "../images/das_mestas_river.pgm"
    SEG = "../images/seg_mestas_wp.raw"
    CENTER = "../images/seg_mestas_wp_centers.raw"
elif parser.parse_args().dataset == "salinas":
    DATASET = "../images/Salinas/RAW/PixelVector_204bands_corrected/Salinas.raw"
    GT = "../images/Salinas/RAW/salinas_gt.pgm"
elif parser.parse_args().dataset == "pavia":
    DATASET = "../images/PaviaUniversity/RAW/PixelVector/PaviaUniversity.raw"
    GT = "../images/PaviaUniversity/RAW/paviau_gt.pgm"
elif parser.parse_args().dataset == "indian":
    DATASET = "../images/IndianPines/RAW/PixelVector/indian_pines_upv.raw"
    GT = "../images/IndianPines/RAW/indian_pines_upv_gt.pgm"
else:
    logger.error("Dataset not found")
    exit()

start = time.time()

dataset = parser.parse_args().dataset

if dataset == "salinas" or dataset == "pavia" or dataset == "indian":
    hyper_dataset = True
else:
    hyper_dataset = False

# Load data
(datos, width, height, bands, unnormalized_data) = read_raw(DATASET, parser.parse_args().normalize)
(truth, width_1, height_1) = read_pgm(GT)
if not hyper_dataset:
    (seg, width_2, height_2) = read_seg(SEG)
    (center, width_3, height_3, nseg) = read_seg_centers(CENTER)
else:
    seg = None
    center = None


patch_size = parser.parse_args().patch_size

# Selection of training, validation and test samples
p_train = 0.15
p_val = 0.05
(train_set, validation_set, test_set, nclases, _) = select_training_samples_seg(
    truth, center, width, height, patch_size, patch_size, [p_train, p_val]
)

model_path = parser.parse_args().model_path
# Loading the data from the experiments
train_set = pickle.load(open(model_path + "/train_set.pickle", "rb"))
validation_set = pickle.load(open(model_path + "/validation_set.pickle", "rb"))
test_set = pickle.load(open(model_path + "/test_set.pickle", "rb"))

patch_size = parser.parse_args().patch_size
dataset_train = HyperDataset(datos, truth, train_set, width, height, bands, patch_size)
dataset_val = HyperDataset(datos, truth, validation_set, width, height, bands, patch_size)
dataset_test = HyperDataset(datos, truth, test_set, width, height, bands, patch_size)

clases_non_baleiras = set()
for patch, label in dataset_train:
    if label not in clases_non_baleiras:
        clases_non_baleiras.add(label)
    if len(clases_non_baleiras) == nclases:
        break
clases_non_baleiras = list(clases_non_baleiras)


##########################################################################################
# ---------------------------------------- DIFFUSION ------------------------------------#
##########################################################################################

image_size = patch_size  # In our RS pipline a patch is an image for the diffusion model
# Should be hilghted that the ViT models uses a internal paramiter called patch_size

logger.info("Creating diffusion model")
logger.info("Bands: %s", bands)
logger.info("Number of classes: %s", nclases)

model_class = getattr(
    __import__("diffusion.models", fromlist=[parser.parse_args().type]),
    parser.parse_args().type,
)
model = model_class(input_size=image_size, num_classes=nclases, in_channels=bands).to(device)

logger.info("Loading weights")
checkpoint_path = model_path + "/checkpoints/"
checkpoint_path = checkpoint_path + "final.pt"
model.load_state_dict(torch.load(checkpoint_path)["ema"])
model.eval()

##########################################################################################
# ------------------------------- EXPERIMENT PRARATION ----------------------------------#
##########################################################################################
arguments = parser.parse_args()
logger.info("Preparing experiment")
EXPERIMENT_FOLDER = (
    f"{model_path}/classification_e{arguments.epochs}_b{arguments.batch_size}"
    f"cfg_scale_{arguments.cfg_scale}_num_sampling_steps_{arguments.num_sampling_steps}"
    f"_BAL_{arguments.balanced}_SYN_{arguments.synthetic}"
    f"_classifier_{arguments.classifier}_NONE_{arguments.none}"
    f"synthetic_normalization_{arguments.synthetic_normalization}/"
)

logger.info("Experiment folder: {EXPERIMENT_FOLDER}")

if not os.path.exists(EXPERIMENT_FOLDER):
    os.makedirs(EXPERIMENT_FOLDER)
log_file = open(EXPERIMENT_FOLDER + "log.txt", "w")
logger.info("Redirecting the output to log file. File: " + EXPERIMENT_FOLDER + "log.txt")
sys.stdout = log_file

logger.info("Parameters:")
logger.info("%s", parser.parse_args())
logger.info(" - training dataset: %s", len(dataset_train))
for i in range(nclases):
    logger.info("   - class %s: %s", i, len([x for x in dataset_train if x[1] == i]))
logger.info(" - validation dataset: %s", len(dataset_val))
for i in range(nclases):
    logger.info("   - class %s: %s", i, len([x for x in dataset_val if x[1] == i]))
logger.info(" - test dataset: %s", len(dataset_test))
for i in range(nclases):
    logger.info("   - class %s: %s", i, len([x for x in dataset_test if x[1] == i]))
logger.info("clases_non_baleiras: %s", clases_non_baleiras)


##########################################################################################
# ------------------------- EXAMPLES OF SYNTHETIC GENERATIONS ---------------------------#
##########################################################################################

logger.info("Generating synthetic examples")
class_labels = []
number_of_samples = 36
logger.info("clases_non_baleiras: %s", clases_non_baleiras)
for i in clases_non_baleiras:
    for j in range(number_of_samples):
        class_labels.append(i)

samples = generate_diffusion_samples(
    model,
    class_labels,
    bands,
    nclases,
    num_sampling_steps=arguments.num_sampling_steps,
    cfg_scale=arguments.cfg_scale,
    device=device,
)
show_samples(
    samples,
    clases_non_baleiras,
    number_of_samples,
    EXPERIMENT_FOLDER,
    normalization=arguments.normalize,
    save=True,
    raw=False,
)
del samples

##########################################################################################
# ---------------------------------- IMAGE POOL GENERATION ------------------------------#
##########################################################################################


number_of_samples = arguments.pool_size

import os
import joblib

pool = dict()

if os.path.exists(EXPERIMENT_FOLDER + "samples.joblib"):
    logger.info("Loading synthetic samples")
    samples = joblib.load(EXPERIMENT_FOLDER + "samples.joblib")
    # Display the number of samples
    logger.info("Samples shape: %s", samples.shape)


else:
    logger.info("Generating synthetic samples pool")
    class_labels = []

    for i in clases_non_baleiras:
        for j in range(number_of_samples):
            class_labels.append(i)

    samples = generate_diffusion_samples(
        model,
        class_labels,
        bands,
        nclases,
        num_sampling_steps=arguments.num_sampling_steps,
        cfg_scale=arguments.cfg_scale,
        device=device,
    )

    # Samples serialization
    joblib.dump(samples, EXPERIMENT_FOLDER + "samples.joblib")

# Save the samples in a dictionary, one key for each class
pool = dict()
for i in range(len(clases_non_baleiras)):
    same_class = []
    for j in range(number_of_samples):
        same_class.append(samples[i * number_of_samples + j].clone())
    pool[clases_non_baleiras[i]] = torch.stack(same_class)

logger.info("Pool completed")
logger.info("Pool keys: %s", pool.keys())
logger.info("Pool generated")


logger.info("Pool stats")
mean = 0
for i in range(len(clases_non_baleiras)):
    for j in range(number_of_samples):
        mean += pool[clases_non_baleiras[i]][j].mean()
mean = mean / (len(clases_non_baleiras) * number_of_samples)
var = 0
for i in range(len(clases_non_baleiras)):
    for j in range(number_of_samples):
        var += (pool[clases_non_baleiras[i]][j].mean() - mean) ** 2
var = var / (len(clases_non_baleiras) * number_of_samples)

logger.info("Mean: %s", mean)
logger.info("Std: %s", np.sqrt(var))
pool_mean = mean
pool_std = np.sqrt(var)

# Mean and std of the train set, to normalize the pool
mean = 0
for i in range(len(dataset_train)):
    mean += dataset_train[i][0].mean()
mean = mean / len(dataset_train)
var = 0
for i in range(len(dataset_train)):
    var += (dataset_train[i][0].mean() - mean) ** 2
var = var / len(dataset_train)
logger.info("Train set stats:")
logger.info("Mean: %s", mean)
logger.info("Std: %s", np.sqrt(var))

real_mean = mean
real_std = np.sqrt(var)

mean = 0
for i in range(len(dataset_test)):
    mean += dataset_test[i][0].mean()
mean = mean / len(dataset_test)
var = 0
for i in range(len(dataset_test)):
    var += (dataset_test[i][0].mean() - mean) ** 2
var = var / len(dataset_test)
logger.info("Test set stats:")
logger.info("Mean: %s", mean)
logger.info("Std: %s", np.sqrt(var))

scale_factor = real_std / pool_std
shift = pool_mean - real_mean
logger.info("Scale factor: %s", scale_factor)
logger.info("Shift: %s", shift)

if parser.parse_args().synthetic_normalization == 1:
    # Pool normalization
    for i in range(len(clases_non_baleiras)):
        for j in range(number_of_samples):
            pool[i][j] = (pool[i][j] - shift) * scale_factor

logger.info("Pool NORMALIZED stats")
mean = 0
for i in clases_non_baleiras:
    for j in range(number_of_samples):
        mean += pool[i][j].mean()
mean = mean / (len(clases_non_baleiras) * number_of_samples)

var = 0
for i in clases_non_baleiras:
    for j in range(number_of_samples):
        var += (pool[i][j].mean() - mean) ** 2
var = var / (len(clases_non_baleiras) * number_of_samples)

logger.info("Mean: %s", mean)
logger.info("Std: %s", np.sqrt(var))
pool_mean = mean
pool_std = np.sqrt(var)

##########################################################################################
# --------------------------------- CHANGES FOR CYTHON TEST -----------------------------#
##########################################################################################

del model
# Free the GPU memory
torch.cuda.empty_cache()
# We convert all the data to numpy arrays for the cython tests
if not hyper_dataset:
    truth = np.array(truth)
    train_set = np.array(train_set)
    validation_set = np.array(validation_set)
    test_set = np.array(test_set)

    truth = truth.astype(np.int64)
    seg = seg.astype(np.int64)
    center = center.astype(np.int64)
    train_set = train_set.astype(np.int64)
    validation_set = validation_set.astype(np.int64)
    test_set = test_set.astype(np.int64)


##########################################################################################
# ------------------------------------------- TRANSFORMERS -------------------------------#
##########################################################################################

batch_size = parser.parse_args().batch_size
epochs = parser.parse_args().epochs
base_learning_rate = 1e-3
weight_decay = 0.05
warmup_epoch = 200
model_path = "./best_model.pt"

##########################################################################################
# ------------------------------------------- UTILES -------------------------------#
##########################################################################################

final_results = dict()
experiments = 5


def gen_classifier(classifier_type):
    if classifier_type == "CNN21":
        logger.info("############              CNN              ############")
        N1 = bands
        D1 = 2
        H1 = image_size
        N2 = 16
        H2 = int(H1 / D1)
        N3 = 32
        D2 = 2
        H3 = int(H2 / D2)
        N4 = H3 * H3 * N3
        N5 = nclases

        classifier = CNN21(N1, N2, N3, N4, N5, D1, D2).to(device)
    elif classifier_type == "ViT":
        logger.info("############              TRANSFORMER              ############")

        depth = 8
        heads = 16
        mlp_ratio = 4
        embed_dim = 64
        transformer_patch_size = 4
        attn_drop_rate = 0.05
        drop_path_rate = 0.05
        drop_rate = 0.05
        classifier = timm.models.vision_transformer.VisionTransformer(
            img_size=image_size,
            patch_size=transformer_patch_size,
            num_classes=nclases,
            in_chans=bands,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
        ).to(device)

        classifier.to(device)
    elif classifier_type == "SwinTransformer":
        depth = [2, 4, 6]
        heads = [4, 8, 16]
        mlp_ratio = 2
        embed_dim = 32
        transformer_patch_size = 2
        window_size = 6
        attn_drop_rate = 0.1
        drop_path_rate = 0.1
        drop_rate = 0.1

        classifier = timm.models.swin_transformer.SwinTransformer(
            img_size=image_size,
            patch_size=transformer_patch_size,
            window_size=window_size,
            num_classes=nclases,
            in_chans=bands,
            embed_dim=embed_dim,
            depths=depth,
            num_heads=heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        ).to(device)
    else:
        logger.error("Error: Classifier not found")
        exit(1)

    return classifier


##########################################################################################
# ------------------------------------ BASELINE -----------------------------------------#
##########################################################################################


# batch_size
train_loader_augmented = DataLoader(dataset_train, batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size, shuffle=False)


if parser.parse_args().none == 1:

    dataset_train = HyperDataset(datos, truth, train_set, width, height, bands, patch_size, augmented=True)

    # batch_size
    train_loader_augmented = DataLoader(dataset_train, batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size, shuffle=False)

    logger.info("None augmentation")
    oas = []
    aas = []
    kappas = []
    f1s = []

    for i in range(experiments):
        logger.info("None augmentation experiment " + str(i) + "/" + str(experiments))
        start = time.time()
        classifier = gen_classifier(arguments.classifier)

        # Redirecting the output to log file
        sys.stdout = log_file

        optim = torch.optim.AdamW(
            classifier.parameters(),
            lr=base_learning_rate * batch_size / 256,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )

        # Cosine learning rate
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-8)

        logger.info("Train model")
        output = train(
            classifier,
            train_loader_augmented,
            val_loader,
            optim,
            device,
            epochs,
            EXPERIMENT_FOLDER + "best_model.pt",
        )
        end = time.time()
        logger.info("Training time: %s", end - start)

        print_train_history(EXPERIMENT_FOLDER + "Baseline_", output[0], output[1], output[2], output[3])
        logger.info("#### Testing Baseline, without validation selection")
        if hyper_dataset:
            oa, aa, aa_c, kappa, f1 = test_without_segments(
                classifier,
                test_loader,
                height,
                width,
                device,
                truth,
                test_set,
                nclases,
                len(clases_non_baleiras),
            )
        else:
            oa, aa, aa_c, kappa, f1 = test(
                classifier,
                test_loader,
                height,
                width,
                device,
                truth,
                seg,
                center,
                train_set,
                validation_set,
                test_set,
                nclases,
                len(clases_non_baleiras),
            )
        logger.info("OA: %s", oa)
        logger.info("AA: %s", aa)
        logger.info("Kappa: %s", kappa)
        logger.info("F1: %s", f1)
        oas.append(oa)
        aas.append(aa)
        kappas.append(kappa)
        f1s.append(f1)

        for i in clases_non_baleiras:
            logger.info("Class %s: %s", i, aa_c[i])

    logger.info("Average OA: %s +/- %s", np.mean(oas), np.std(oas))
    logger.info("Average AA: %s +/- %s", np.mean(aas), np.std(aas))
    logger.info("Average Kappa: %s +/- %s", np.mean(kappas), np.std(kappas))
    logger.info("Average F1: %s +/- %s", np.mean(f1s), np.std(f1s))

    final_results["Baseline"] = (
        np.mean(oas),
        np.std(oas),
        np.mean(aas),
        np.std(aas),
        np.mean(kappas),
        np.std(kappas),
        np.mean(f1s),
        np.std(f1s),
    )

dataset_train = HyperDataset(datos, truth, train_set, width, height, bands, patch_size, augmented=False)


##########################################################################################
# ----------------------------------------- EBDA ----------------------------------------#
##########################################################################################

coherences = [0.1, 0.5, 1]

if parser.parse_args().balanced == 1:

    logger.info("EBDA")
    for c in coherences:
        logger.info("EBDA p = " + str(c))
        start = time.time()
        dataset_train_augmented = DatasetAugmentedBalancer(
            datos,
            truth,
            train_set,
            width,
            height,
            bands,
            patch_size,
            pool,
            coherence=c,
            classes=clases_non_baleiras,
            augmented=True,
        )

        # batch_size
        train_loader_augmented = DataLoader(dataset_train_augmented, batch_size, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size, shuffle=False)
        test_loader = DataLoader(dataset_test, batch_size, shuffle=False)

        logger.info("Train model with EBDA coherence: " + str(c))
        oas = []
        aas = []
        kappas = []
        f1s = []

        for i in range(experiments):
            logger.info("EBDA experiment " + str(i) + "/" + str(experiments))
            start = time.time()
            classifier = gen_classifier(arguments.classifier)

            balanced_epochs = int(epochs + (epochs * 0.5 * c))

            optim = torch.optim.AdamW(
                classifier.parameters(),
                lr=base_learning_rate * batch_size / 256,
                betas=(0.9, 0.95),
                weight_decay=weight_decay,
            )

            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=balanced_epochs, eta_min=1e-8)

            logger.info("Train model")
            output = train(
                classifier,
                train_loader_augmented,
                val_loader,
                optim,
                device,
                balanced_epochs,
                EXPERIMENT_FOLDER + "best_model.pt",
            )

            end = time.time()
            logger.info("Training time: %s", end - start)

            print_train_history(
                EXPERIMENT_FOLDER + "EBDA_" + str(c) + "_",
                output[0],
                output[1],
                output[2],
                output[3],
            )

            if hyper_dataset:
                oa, aa, aa_c, kappa, f1 = test_without_segments(
                    classifier,
                    test_loader,
                    height,
                    width,
                    device,
                    truth,
                    test_set,
                    nclases,
                    len(clases_non_baleiras),
                )
            else:
                oa, aa, aa_c, kappa, f1 = test(
                    classifier,
                    test_loader,
                    height,
                    width,
                    device,
                    truth,
                    seg,
                    center,
                    train_set,
                    validation_set,
                    test_set,
                    nclases,
                    len(clases_non_baleiras),
                )
            logger.info("#### Testing with EBDA, without validation selection")
            logger.info("OA: %s", oa)
            logger.info("AA: %s", aa)
            logger.info("Kappa: %s", kappa)
            logger.info("F1: %s", f1)
            oas.append(oa)
            aas.append(aa)
            kappas.append(kappa)
            f1s.append(f1)

            for i in clases_non_baleiras:
                logger.info("Class %s: %s", i, aa_c[i])

        logger.info("Average OA: %s +/- %s", np.mean(oas), np.std(oas))
        logger.info("Average AA: %s +/- %s", np.mean(aas), np.std(aas))
        logger.info("Average Kappa: %s +/- %s", np.mean(kappas), np.std(kappas))
        logger.info("Average F1: %s +/- %s", np.mean(f1s), np.std(f1s))

        final_results["EBDA" + str(c)] = (
            np.mean(oas),
            np.std(oas),
            np.mean(aas),
            np.std(aas),
            np.mean(kappas),
            np.std(kappas),
            np.mean(f1s),
            np.std(f1s),
        )

##########################################################################################
# ---------------------------------- SYNTHETICS EXPERIMENT ------------------------------#
##########################################################################################

if parser.parse_args().synthetic == 1:
    synthetic_sizes = [100, 1000]

    for s in synthetic_sizes:
        logger.info("Synthetic: %s", s)
        start = time.time()
        dataset_train_augmented = DatasetSynthtic(pool, size=s, classes=clases_non_baleiras)

        # batch_size
        train_loader_augmented = DataLoader(dataset_train_augmented, batch_size, shuffle=True)
        train_loader = DataLoader(dataset_train, batch_size, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size, shuffle=False)
        test_loader = DataLoader(dataset_test, batch_size, shuffle=False)
        logger.info("Train model")
        out_data = dataset_train_augmented.patches
        with open(EXPERIMENT_FOLDER + "synthetic_" + str(s) + "_data.pkl", "wb") as f:
            pickle.dump(out_data, f)

        oas = []
        aas = []
        kappas = []
        f1s = []

        for i in range(experiments):
            start = time.time()
            logger.info("Synthetic generation experiment {i}")
            classifier = gen_classifier(arguments.classifier)

            optim = torch.optim.AdamW(
                classifier.parameters(),
                lr=base_learning_rate * batch_size / 256,
                betas=(0.9, 0.95),
                weight_decay=weight_decay,
            )

            def lr_func(epoch):
                return min(
                    (epoch + 1) / (warmup_epoch + 1e-8),
                    0.5 * (math.cos(epoch / epochs * math.pi) + 1),
                )

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)
            optim.scheduler = lr_scheduler

            logger.info("Train model")

            output = train(
                classifier,
                train_loader_augmented,
                val_loader,
                optim,
                device,
                epochs,
                EXPERIMENT_FOLDER + "best_model.pt",
            )

            end = time.time()
            logger.info("Training time: %s", end - start)

            print_train_history(
                EXPERIMENT_FOLDER + "synthetic_" + str(s) + "_",
                output[0],
                output[1],
                output[2],
                output[3],
            )

            if hyper_dataset:
                oa, aa, aa_c, kappa, f1 = test_without_segments(
                    classifier,
                    test_loader,
                    height,
                    width,
                    device,
                    truth,
                    test_set,
                    nclases,
                    len(clases_non_baleiras),
                )
            else:
                oa, aa, aa_c, kappa, f1 = test(
                    classifier,
                    test_loader,
                    height,
                    width,
                    device,
                    truth,
                    seg,
                    center,
                    train_set,
                    validation_set,
                    test_set,
                    nclases,
                    len(clases_non_baleiras),
                )
            logger.info("#### Fine tuning, without real damages")
            logger.info("OA: %s", oa)
            logger.info("AA: %s", aa)
            logger.info("Kappa: %s", kappa)
            logger.info("F1: %s", f1)
            oas.append(oa)
            aas.append(aa)
            kappas.append(kappa)
            f1s.append(f1)

            for i in clases_non_baleiras:
                logger.info("Class %s: %s", i, aa_c[i])

        logger.info("Average OA: %s +/- %s", np.mean(oas), np.std(oas))
        logger.info("Average AA: %s +/- %s", np.mean(aas), np.std(aas))
        logger.info("Average Kappa: %s +/- %s", np.mean(kappas), np.std(kappas))
        logger.info("Average F1: %s +/- %s", np.mean(f1s), np.std(f1s))
        final_results["Synthetic " + str(c)] = (
            np.mean(oas),
            np.std(oas),
            np.mean(aas),
            np.std(aas),
            np.mean(kappas),
            np.std(kappas),
            np.mean(f1s),
            np.std(f1s),
        )

##########################################################################################
# ---------------------------------- FINAL RESULTS --------------------------------------#
##########################################################################################
logger.info("GLOBAL RESULTS")
for key, value in final_results.items():
    logger.info("%s", key)
    logger.info("OA: %s +/- %s", value[0], value[1])
    logger.info("AA: %s +/- %s", value[2], value[3])
    logger.info("Kappa: %s +/- %s", value[4], value[5])
    logger.info("F1: %s +/- %s", value[6], value[7])

with open(EXPERIMENT_FOLDER + "results.csv", "w") as f:
    # Add the header
    f.write("Experiment,OA,OA_std,AA,AA_std,Kappa,Kappa_std,F1,F1_std\n")
    for key in final_results.keys():
        f.write(
            "%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
            % (
                key,
                str(final_results[key][0]),
                str(final_results[key][1]),
                str(final_results[key][2]),
                str(final_results[key][3]),
                str(final_results[key][4]),
                str(final_results[key][5]),
                str(final_results[key][6]),
                str(final_results[key][7]),
            )
        )

# Closing the log file
log_file.close()
