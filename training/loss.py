# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

# ----------------------------------------------------------------------------


class Loss:
    def accumulate_gradients(
        self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg
    ):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------


def compute_class_prediction_accuracy(predictions, ground_truth):
    """
    Computes a tensor per class with 1 for correctly classified instances
    and 0 for incorrectly classified instances.

    Args:
        predictions (torch.Tensor): Tensor of size (batch_size, dim_label) with predicted values.
        ground_truth (torch.Tensor): Tensor of size (batch_size,) with ground truth labels.

    Returns:
        dict: Dictionary where each key is a class and the value is a tensor
            of size (num_elements_class, 1) with 1 for correct predictions and 0 for incorrect ones.
    """
    predicted_classes = torch.argmax(predictions, dim=1)  # (index of the minimum value per row)
    ground_truth = torch.argmax(ground_truth, dim=1)

    results_per_class = {}
    for cls in range(predictions.shape[1]):
        # Mask to select instances of the current class
        mask = ground_truth == cls

        # Extract predictions and ground truth for the current class
        class_predictions = predicted_classes[mask]
        class_ground_truth = ground_truth[mask]

        # Compare predictions with ground truth (1 if correct, 0 if incorrect)
        correct_tensor = (class_predictions == class_ground_truth).float().unsqueeze(1)

        # Store the tensor in the dictionary
        results_per_class[cls] = correct_tensor.squeeze(1)

    return results_per_class


def compute_confusion_matrix_dict(predictions, ground_truth):
    """
    Computes a confusion matrix in dictionary format.

    Args:
        predictions (torch.Tensor): Tensor of size (batch_size, dim_label) with predicted values.
        ground_truth (torch.Tensor): Tensor of size (batch_size,) with ground truth labels.

    Returns:
        dict: Dictionary where each key is "Classification/{real_class}/{predicted_class}"
              and the value is a tensor with 1s for each instance where the prediction matches
              the real class and 0s otherwise.
    """
    # Get predicted classes and ground truth classes
    predicted_classes = torch.argmax(predictions, dim=1)
    ground_truth_classes = torch.argmax(ground_truth, dim=1)

    # Initialize the confusion matrix dictionary
    confusion_matrix_dict = {}

    # Iterate over all possible real and predicted classes
    num_classes = predictions.shape[1]
    for real_class in range(num_classes):
        for predicted_class in range(num_classes):
            # Mask for instances where the real class matches `real_class`
            real_mask = ground_truth_classes == real_class

            # Mask for instances where the predicted class matches `predicted_class`
            predicted_mask = predicted_classes == predicted_class

            # Combine masks to find instances where real_class and predicted_class match
            combined_mask = real_mask & predicted_mask

            # Create a tensor with 1s for matching instances and 0s otherwise
            confusion_tensor = combined_mask.float().unsqueeze(1)

            # Store the tensor in the dictionary
            key = f"Classification/{real_class}/{predicted_class}"
            confusion_matrix_dict[key] = confusion_tensor.squeeze(1)

    return confusion_matrix_dict


class StyleGAN2Loss(Loss):
    def __init__(
        self,
        device,
        G,
        D,
        augment_pipe=None,
        r1_gamma=10,
        style_mixing_prob=0,
        pl_weight=0,
        pl_batch_shrink=2,
        pl_decay=0.01,
        pl_no_weight_grad=False,
        blur_init_sigma=0,
        blur_fade_kimg=0,
        class_weight=0,
    ):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_no_weight_grad = pl_no_weight_grad
        self.pl_mean = torch.zeros([], device=device)
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.class_weight = class_weight

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function("style_mixing"):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws.device) < self.style_mixing_prob,
                    cutoff,
                    torch.full_like(cutoff, ws.shape[1]),
                )
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function("blur"):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        conditioned_logits, classification_logits = self.D(img, c, update_emas=update_emas)
        return conditioned_logits, classification_logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ["Gmain", "Greg", "Gboth", "Dmain", "Dreg", "Dboth"]
        if self.pl_weight == 0:
            phase = {"Greg": "none", "Gboth": "Gmain"}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {"Dreg": "none", "Dboth": "Dmain"}.get(phase, phase)
        blur_sigma = (
            max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        )

        # Gmain: Maximize logits for generated images.
        if phase in ["Gmain", "Gboth"]:
            with torch.autograd.profiler.record_function("Gmain_forward"):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c)
                gen_conditioned_logits, _gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report("Loss/scores/fake", gen_conditioned_logits)
                training_stats.report("Loss/signs/fake", gen_conditioned_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_conditioned_logits)  # -log(sigmoid(gen_logits))
                training_stats.report("Loss/G/loss", loss_Gmain)
            with torch.autograd.profiler.record_function("Gmain_backward"):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ["Greg", "Gboth"]:
            with torch.autograd.profiler.record_function("Gpl_forward"):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function("pl_grads"), conv2d_gradfix.no_weight_gradients(
                    self.pl_no_weight_grad
                ):
                    pl_grads = torch.autograd.grad(
                        outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True
                    )[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report("Loss/pl_penalty", pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report("Loss/G/reg", loss_Gpl)
            with torch.autograd.profiler.record_function("Gpl_backward"):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ["Dmain", "Dboth"]:
            with torch.autograd.profiler.record_function("Dgen_forward"):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
                gen_conditioned_logits, _gen_logits = self.run_D(
                    gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True
                )
                training_stats.report("Loss/scores/fake", gen_conditioned_logits)
                training_stats.report("Loss/signs/fake", gen_conditioned_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_conditioned_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function("Dgen_backward"):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ["Dmain", "Dreg", "Dboth"]:
            name = "Dreal" if phase == "Dmain" else "Dr1" if phase == "Dreg" else "Dreal_Dr1"
            with torch.autograd.profiler.record_function(name + "_forward"):
                real_img_tmp = real_img.detach().requires_grad_(phase in ["Dreg", "Dboth"])
                real_conditioned_logits, real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report("Loss/scores/real", real_conditioned_logits)
                training_stats.report("Loss/signs/real", real_conditioned_logits.sign())

                # Compute classification results per class (only reporting)
                if phase in ["Dmain", "Dboth"] and real_logits is not None:
                    results_per_class = compute_class_prediction_accuracy(real_logits, real_c)
                    for cls, classification_tensor in results_per_class.items():
                        training_stats.report(f"Accuracy/{cls}", classification_tensor)
                    confusion_matrix_dict = compute_confusion_matrix_dict(real_logits, real_c)
                    for key, confusion_tensor in confusion_matrix_dict.items():
                        training_stats.report(key, confusion_tensor)

                # Compute losses (adversarial and classification)
                loss_Dreal = 0
                loss_cls_real = 0
                if phase in ["Dmain", "Dboth"]:
                    loss_Dreal = torch.nn.functional.softplus(-real_conditioned_logits)  # -log(sigmoid(real_logits))
                    training_stats.report("Loss/D/adversarial", loss_Dgen + loss_Dreal)
                    if real_logits is not None:
                        loss_cls_real = torch.nn.functional.cross_entropy(real_logits, real_c.argmax(dim=1))
                        training_stats.report("Loss/D/classification/real", loss_cls_real)
                    loss_Dreal = loss_Dreal + self.class_weight * loss_cls_real # 0 if no classification loss
                    training_stats.report("Loss/D/loss", loss_Dreal)

                loss_Dr1 = 0
                if phase in ["Dreg", "Dboth"]:
                    with torch.autograd.profiler.record_function("r1_grads"), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(
                            outputs=[real_conditioned_logits.sum()],  # we do not include classification logits in R1
                            inputs=[real_img_tmp],
                            create_graph=True,
                            only_inputs=True,
                        )[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report("Loss/r1_penalty", r1_penalty)
                    training_stats.report("Loss/D/reg", loss_Dr1)

            with torch.autograd.profiler.record_function(name + "_backward"):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

    def evaluate_discriminator(self, real_img, real_c):
        self.D.eval()
        with torch.no_grad():
            _, real_logits = self.D(real_img, real_c)

            # Compute classification results per class and report
            results_per_class = compute_class_prediction_accuracy(real_logits, real_c)
            for cls, classification_tensor in results_per_class.items():
                training_stats.report(f"Accuracy/val/{cls}", classification_tensor)
            confusion_matrix_dict = compute_confusion_matrix_dict(real_logits, real_c)
            for key, confusion_tensor in confusion_matrix_dict.items():
                training_stats.report(key.replace("Classification", "Classification/val"), confusion_tensor)

            # Compute loss and report
            loss_cls_real = torch.nn.functional.cross_entropy(real_logits, real_c.argmax(dim=1))
            training_stats.report("Loss/D/classification/real/val", loss_cls_real)

        self.D.train()


# ----------------------------------------------------------------------------
