import time
import warnings

import numpy as np
import scipy
import torch


def get_features(model, examples, batch_size=128, device="cpu"):
    """Helper to extract features in batches."""
    model.eval()
    features_list = []

    if isinstance(examples, list):
        examples = torch.stack(examples)

    num_samples = len(examples)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = examples[i : i + batch_size].to(device)
            _, features = model(batch)
            features_list.append(features.cpu())

    return torch.cat(features_list, dim=0).numpy()


def calculate_fid(model, examples_1, examples_2, batch_size=128, device="cpu"):
    """Calculates the Fréchet Inception Distance (FID) between two sets of examples.

    The FID is a measure of similarity between two sets of images. It is calculated
    by comparing the statistics of feature representations from an intermediate layer
    of a pre-trained network.

    Args:
        model (torch.nn.Module): The feature extractor network (e.g., InceptionV3).
        examples_1 (torch.Tensor or list): A batch of images from the first set (e.g., real images).
        examples_2 (torch.Tensor or list): A batch of images from the second set (e.g., generated images).
        batch_size (int, optional): The batch size for feature extraction. Defaults to 32.
        device (str, optional): The device to use for computation ('cpu' or 'cuda'). Defaults to "cpu".

    Raises:
        ValueError: If the FID calculation results in a complex number with a
                    significant imaginary component, indicating a numerical instability.

    Returns:
        float: The calculated FID score.
    """

    model = model.to(device)

    act1 = get_features(model, examples_1, batch_size, device)
    act2 = get_features(model, examples_2, batch_size, device)

    torch.cuda.empty_cache()

    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    # Check that it is symmetric
    if not np.allclose(sigma1, sigma1.T, atol=1e-6):
        warnings.warn("Sigma1 is not symmetric")
    else:
        warnings.warn("Sigma1 is symmetric")
    if not np.allclose(sigma2, sigma2.T, atol=1e-6):
        warnings.warn("Sigma2 is not symmetric")
    else:
        warnings.warn("Sigma2 is symmetric")

    diff = mu1 - mu2

    sigma1 = sigma1.astype(np.float64)
    sigma2 = sigma2.astype(np.float64)

    # Product might be almost singular
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        print(f"[!] FID calculation produces a singular product; adding {1e-6} to diagonal of cov estimates")
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"[!] FID calculation produces complex value: {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    del act1, act2, mu1, mu2, sigma1, sigma2, diff, covmean, tr_covmean
    torch.cuda.empty_cache()

    return fid


def batch_pairwise_distances(U, V):
    """
    Compute pairwise distances between two batches of feature vectors.

    Args:
        U (torch.Tensor): A batch of feature vectors of shape (N, F).
        V (torch.Tensor): A batch of feature vectors of shape (M, F).

    Returns:
        torch.Tensor: A distance matrix of shape (N, M).
    """
    norm_u = torch.sum(U**2, dim=1, keepdim=True)
    norm_v = torch.sum(V**2, dim=1, keepdim=True).t()
    D = torch.clamp(norm_u - 2 * torch.mm(U, V.t()) + norm_v, min=0.0)
    return D


class DistanceBlock:
    """
    Provides multi-GPU support to calculate pairwise distances between two batches of feature vectors.

    Args:
        num_features (int): The number of features in the input vectors.
        num_gpus (int): The number of GPUs to use for computation.
    """

    def __init__(self, num_features, num_gpus):
        self.num_features = num_features
        self.num_gpus = num_gpus

    def pairwise_distances(self, U, V):
        """
        Evaluate pairwise distances between two batches of feature vectors.

        Args:
            U (torch.Tensor): A batch of feature vectors.
            V (torch.Tensor): A batch of feature vectors.

        Returns:
            torch.Tensor: A tensor containing the pairwise distances.
        """
        # We split V (columns) across GPUs, and replicate U (rows).
        # This computes the distance matrix in vertical strips and concatenates them horizontally.
        V_split = torch.chunk(V, self.num_gpus)

        distances_split = []
        for i in range(len(V_split)):
            # Determine device for this chunk
            device = torch.device(f"cuda:{i}")

            # Move U to device (replicated)
            U_gpu = U.to(device)
            # Move V chunk to device
            V_gpu = V_split[i].to(device)

            # Compute distances on this device
            dist = batch_pairwise_distances(U_gpu, V_gpu).cpu()
            distances_split.append(dist)

            del U_gpu, V_gpu

        torch.cuda.empty_cache()
        return torch.cat(distances_split, dim=1)


class ManifoldEstimator:
    """
    Estimates the manifold of given feature vectors by calculating k-NN distances.

    This class computes the distances to the k-nearest neighbors for each feature vector
    in a reference set, which defines the manifold. It can then be used to evaluate
    whether new feature vectors fall within this manifold.

    Args:
        distance_block (DistanceBlock): An object to compute pairwise distances.
        features (np.ndarray): The reference feature vectors.
        row_batch_size (int): Batch size for rows in distance calculation.
        col_batch_size (int): Batch size for columns in distance calculation.
        nhood_sizes (list[int]): A list of k values for k-NN.
        clamp_to_percentile (float, optional): If specified, clamps distances to a given percentile.
        eps (float): A small epsilon value to avoid division by zero.
    """

    def __init__(
        self,
        distance_block,
        features,
        row_batch_size=25000,
        col_batch_size=50000,
        nhood_sizes=[3],
        clamp_to_percentile=None,
        eps=1e-5,
    ):
        """Estimate the manifold of given feature vectors."""
        num_images = features.shape[0]
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = torch.tensor(features, dtype=torch.float32)
        self._distance_block = distance_block

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        self.D = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, row_batch_size):
            end1 = min(begin1 + row_batch_size, num_images)
            row_batch = self._ref_features[begin1:end1]

            for begin2 in range(0, num_images, col_batch_size):
                end2 = min(begin2 + col_batch_size, num_images)
                col_batch = self._ref_features[begin2:end2]

                # Compute distances between batches.
                distance_batch[0 : end1 - begin1, begin2:end2] = self._distance_block.pairwise_distances(
                    row_batch, col_batch
                ).numpy()

            # Find the k-nearest neighbor from the current batch.
            self.D[begin1:end1, :] = np.partition(distance_batch[0 : end1 - begin1, :], seq, axis=1)[
                :, self.nhood_sizes
            ]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis=0)
            self.D[self.D > max_distances] = 0

    def evaluate(self, eval_features, return_realism=False, return_neighbors=False):
        """
        Evaluate if new feature vectors are within the estimated manifold.

        Args:
            eval_features (np.ndarray): The feature vectors to evaluate.
            return_realism (bool): Whether to return the realism score.
            return_neighbors (bool): Whether to return the nearest neighbor indices.

        Returns:
            np.ndarray: An array of predictions (1 if in manifold, 0 otherwise).
            (optional) np.ndarray: Realism scores.
            (optional) np.ndarray: Nearest neighbor indices.
        """
        eval_features = torch.tensor(eval_features, dtype=torch.float32)
        num_eval_images = eval_features.shape[0]
        num_ref_images = self.D.shape[0]
        distance_batch = np.zeros([self.row_batch_size, num_ref_images], dtype=np.float32)
        batch_predictions = np.zeros([num_eval_images, self.num_nhoods], dtype=np.int32)
        max_realism_score = np.zeros(
            [
                num_eval_images,
            ],
            dtype=np.float32,
        )
        nearest_indices = np.zeros(
            [
                num_eval_images,
            ],
            dtype=np.int32,
        )

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[0 : end1 - begin1, begin2:end2] = self._distance_block.pairwise_distances(
                    feature_batch, ref_batch
                ).numpy()

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            samples_in_manifold = distance_batch[0 : end1 - begin1, :, None] <= self.D
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis=1).astype(np.int32)

            max_realism_score[begin1:end1] = np.max(
                self.D[:, 0] / (distance_batch[0 : end1 - begin1, :] + self.eps), axis=1
            )
            nearest_indices[begin1:end1] = np.argmin(distance_batch[0 : end1 - begin1, :], axis=1)

        if return_realism and return_neighbors:
            return batch_predictions, max_realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, max_realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices

        return batch_predictions


def knn_precision_recall_features(
    ref_features, eval_features, nhood_sizes=[3], row_batch_size=10000, col_batch_size=50000, num_gpus=1
):
    """
    Calculates k-NN precision and recall for two sets of feature vectors.

    Args:
        ref_features (np.ndarray): Reference feature vectors (e.g., from real data).
        eval_features (np.ndarray): Evaluation feature vectors (e.g., from generated data).
        nhood_sizes (list[int]): List of k values for k-NN.
        row_batch_size (int): Batch size for rows in distance calculation.
        col_batch_size (int): Batch size for columns in distance calculation.
        num_gpus (int): Number of GPUs to use.

    Returns:
        dict: A dictionary containing 'precision' and 'recall' arrays.
    """
    state = dict()
    num_images = ref_features.shape[0]
    num_features = ref_features.shape[1]

    # Initialize DistanceBlock and ManifoldEstimators.
    distance_block = DistanceBlock(num_features, num_gpus)
    ref_manifold = ManifoldEstimator(distance_block, ref_features, row_batch_size, col_batch_size, nhood_sizes)
    eval_manifold = ManifoldEstimator(distance_block, eval_features, row_batch_size, col_batch_size, nhood_sizes)

    # Evaluate precision and recall using k-nearest neighbors.
    print("Evaluating k-NN precision and recall with %i samples..." % num_images)
    start = time.time()

    # Precision: How many points from eval_features are in ref_features manifold.
    precision = ref_manifold.evaluate(eval_features)
    state["precision"] = precision.mean(axis=0)
    print("Precision:", state["precision"])

    # Recall: How many points from ref_features are in eval_features manifold.
    recall = eval_manifold.evaluate(ref_features)
    state["recall"] = recall.mean(axis=0)
    print("Recall:", state["recall"])

    print("Evaluated k-NN precision and recall in: %gs" % (time.time() - start))

    return state


def compute_precision_recall(judge_model, examples_1, examples_2, batch_size=128, device="cpu", num_gpus=1):
    """Computes k-NN precision and recall between two sets of examples using a judge model."""

    judge_model = judge_model.to(device)

    # Extract features using the judge model with batching
    features_1 = get_features(judge_model, examples_1, batch_size, device)
    features_2 = get_features(judge_model, examples_2, batch_size, device)

    state = knn_precision_recall_features(
        features_1, features_2, row_batch_size=10000, col_batch_size=50000, num_gpus=num_gpus
    )
    precision = state["precision"][0]
    recall = state["recall"][0]

    return precision, recall
