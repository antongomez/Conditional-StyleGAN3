import torch
import torch.nn as nn


class Discriminator_ConvBlock(nn.Module):
    """Convolutional block for building the residual blocks in this module.

    A convolutional block contains a convolutional layer, a normalization layer (optional), an activation layer
    (optional), and a dropout layer (optional).

    Attributes
    ----------
    in_channels : int
        The number of channels in the input samples.

    out_channels : int
        The number of channels in the output samples.

    kernel_size : tuple
        The size of the convolutional kernel.

    strides : tuple
        The strides of the convolutional layer.

    norm : bool
        Whether to use normalization or not.

    old_norm : bool
        If True, batch normalization is used. Otherwise, spectral normalization is used.

    lrelu : bool
        Whether to use an activation layer or not.

    dropout : bool
        Whether to use a dropout layer or not.
    """

    def __init__(
        self,
        hyperparams,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        strides=(1, 1),
        norm=True,
        old_norm=False,
        lrelu=True,
        dropout=True,
    ):
        """Defines and initializes the convolutional block.

        Parameters
        ----------
        hyperparams : dict
            Some hyperparameters are used to define the block structure and its behavior.

        in_channels : int
            The number of channels in the input samples.

        out_channels : int
            The number of channels in the output samples.

        kernel_size : tuple, optional
            The size of the convolutional kernel. Defaults to (3, 3).

        strides : tuple, optional
            The strides of the convolutional layer. Defaults to (1, 1).

        norm : bool, optional
            Whether to use normalization or not. Defaults to True.

        old_norm : bool, optional
            If True, batch normalization is used. Otherwise, spectral normalization is used. Defaults to False.

        lrelu : bool, optional
            Whether to use an activation layer or not. The activation function is taken from the hyperparams. Defaults
            to True.

        dropout : bool, optional
            Whether to use a dropout layer or not. The dropout probability is taken from the hyperparams. Defaults to
            True.
        """

        super(Discriminator_ConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.norm = norm
        self.old_norm = old_norm
        self.lrelu = lrelu
        self.dropout = dropout

        layers = []

        if self.norm:
            if self.old_norm:
                layers.append(
                    nn.Conv2d(
                        self.in_channels,
                        self.out_channels,
                        self.kernel_size,
                        stride=self.strides,
                        padding=(1, 1),
                        padding_mode="replicate",
                        bias=True,
                    ),
                )
                layers.append(nn.BatchNorm2d(self.out_channels))
            else:
                layers.append(
                    nn.utils.parametrizations.spectral_norm(
                        nn.Conv2d(
                            self.in_channels,
                            self.out_channels,
                            self.kernel_size,
                            stride=self.strides,
                            padding=(1, 1),
                            padding_mode="replicate",
                            bias=True,
                        ),
                        dim=1,
                    )
                )
        else:
            layers.append(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    stride=self.strides,
                    padding=(1, 1),
                    padding_mode="replicate",
                    bias=True,
                ),
            )

        if self.lrelu:
            if hyperparams["activation"] == "elu":
                layers.append(nn.ELU(alpha=1.0, inplace=True))
            elif hyperparams["activation"] == "prelu":
                layers.append(nn.PReLU(num_parameters=self.out_channels, init=0.25))
            else:
                layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        if self.dropout:
            layers.append(nn.Dropout(p=hyperparams["p_dropout"]))

        self.layers = nn.Sequential(*layers)

    def forward(self, tensor):
        """Processes a batch of samples.

        Parameters
        ----------
        tensor : torch.Tensor
            The input batch.

        Returns
        -------
        torch.Tensor
            The result of applying the block's different transformations over the input batch.
        """

        out = self.layers(tensor)
        return out


class Discriminator_ResBlock(nn.Module):
    """Residual block for building the residual stages in this module.

    A residual block contains two convolutional blocks:

    - The first one changes the number of channels from in_channels to out_channels.
    - The second one changes leaves the number of channels unchanged.

    The first convolutional block is also able to downsample the spatial dimensions of the input tensor using strides.

    Attributes
    ----------
    in_channels : int
        The number of channels in the input samples.

    out_channels : int
        The number of channels in the output samples.

    strides : tuple
        The strides used in the first convolutional block.

    downsample : torch.nn.Module
        If needed, the module used to downsample the input tensor to perform the residual connection at the end.

    norm : bool
        Whether to use normalization or not after each convolutional block.

    old_norm : bool
        If True, batch normalization is used. Otherwise, spectral normalization is used.
    """

    def __init__(
        self, hyperparams, in_channels, out_channels, strides=(1, 1), downsample=None, norm=True, old_norm=False
    ):
        """Defines and initializes the residual block.

        Parameters
        ----------
        hyperparams : dict
            Some hyperparameters are used to define the block structure and its behavior.

        in_channels : int
            The number of channels in the input samples.

        out_channels : int
            The number of channels in the output samples.

        strides : tuple, optional
            The strides used in the first convolutional block. Defaults to (1, 1).

        downsample : nn.Module, optional
            The downsampling layer to be used in the residual block, if any. Defaults to None.

        norm : bool, optional
            Whether to use spectral normalization in the convolutional blocks. Defaults to True.

        old_norm : bool, optional
            If True, batch normalization is used instead of spectral normalization. Defaults to
            False.
        """

        super(Discriminator_ResBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.downsample = downsample
        self.norm = norm
        self.old_norm = old_norm

        # 1. From in_channels to out_channels
        self.conv_1 = Discriminator_ConvBlock(
            hyperparams,
            self.in_channels,
            self.out_channels,
            strides=self.strides,
            norm=self.norm,
            old_norm=self.old_norm,
        )

        # 2. Leaves the same number of channels
        self.conv_2 = Discriminator_ConvBlock(
            hyperparams, self.out_channels, self.out_channels, norm=self.norm, old_norm=self.old_norm
        )

        self.downsample = downsample

    def forward(self, tensor):
        """Processes a batch of samples.

        Parameters
        ----------
        tensor : torch.Tensor
            The input batch.

        Returns
        -------
        torch.Tensor
            The result of applying the block's different transformations over the input batch.
        """

        identity = self.downsample(tensor) if self.downsample is not None else tensor

        out = self.conv_1(tensor)
        out = self.conv_2(out)

        return out + identity


class Discriminator_ResStage(nn.Module):
    """Residual stage for building the residual CNNs in this module.

    A residual stage consists of num_blocks residual blocks that use the same number of channels:

    - The first block changes the number of channels from in_channels to out_channels.
    - The other blocks leave the number of channels unchanged.

    The first residual block is also able to downsample the spatial dimensions of the input tensor using strides.

    Attributes
    ----------
    num_blocks : int
        The number of residual blocks in the stage.

    in_channels : int
        The number of channels in the input samples.

    out_channels : int
        The number of channels in the output samples.

    strides : tuple
        The strides used in the first convolutional block of the stage.

    norm : bool
        Whether to use normalization or not after each convolutional block.

    old_norm : bool
        If True, batch normalization is used. Otherwise, spectral normalization is used.
    """

    def __init__(self, hyperparams, num_blocks, in_channels, out_channels, strides=(1, 1), norm=True, old_norm=False):
        """Defines and initializes the residual stage.

        Parameters
        ----------
        hyperparams : dict
            Some hyperparameters are used to define the stage structure and its behavior.

        num_blocks : int
            The number of residual blocks in the stage.

        in_channels : int
            The number of channels in the input samples.

        out_channels : int
            The number of channels in the output samples.

        strides : tuple, optional
            The strides used in the first convolutional block of the stage. Defaults to (1, 1).

        norm : bool, optional
            Whether to use spectral normalization after the convolutional blocks. Defaults to True.

        old_norm : bool, optional
            If True, batch normalization is used instead of spectral normalization. Defaults to
            False.
        """

        super(Discriminator_ResStage, self).__init__()

        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.norm = norm
        self.old_norm = old_norm

        blocks = []

        # The first convolutional block may:
        #
        #   - Change the number of channels.
        #   - Downsample the spatial dimensions of the input tensor.
        #
        # If any of these is true, we need to provide a downsampling module to the first block so that it is able to
        # apply the residual connection.
        downsample = None

        if (self.in_channels != self.out_channels) or (self.strides != (1, 1)):
            downsample = Discriminator_ConvBlock(
                hyperparams,
                self.in_channels,
                self.out_channels,
                strides=self.strides,
                norm=self.norm,
                lrelu=False,
                old_norm=self.old_norm,
            )

        blocks.append(
            Discriminator_ResBlock(
                hyperparams,
                self.in_channels,
                self.out_channels,
                strides=self.strides,
                downsample=downsample,
                norm=self.norm,
                old_norm=self.old_norm,
            )
        )

        # The remaining convolutional blocks are simpler
        for _ in range(1, self.num_blocks):
            blocks.append(
                Discriminator_ResBlock(
                    hyperparams, self.out_channels, self.out_channels, norm=self.norm, old_norm=self.old_norm
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, tensor):
        """Processes a batch of samples.

        Parameters
        ----------
        tensor : torch.Tensor
            The input batch.

        Returns
        -------
        torch.Tensor
            The result of applying the stage's different transformations over the input batch.
        """

        out = self.blocks(tensor)
        return out


class CNN2D_Residual(nn.Module):
    """Residual 2D Convolutional Neuronal Network.

    This network is composed of 3 residual stages followed by an average pooling layer and a fully-connected block.
    Each residual stage is composed of 3 convolutional blocks. All convolutional blocks in a given stage use the same
    number of channels; this amount is increased with each stage. A feature-fusion that combines the features
    extracted from each stage is performed before the average pooling layer.

    Attributes
    ----------
    bands : int
        The number of bands in the input samples.

    patch_size : int
        The size of the input samples. It is measured in pixels along the patch's side.

    classes_count : int
        The number of classes in the target dataset.

    device : str
        The computational device that will be used to run the network.

    Z : int
        The size of the latent space.

    classifier_weights : torch.Tensor
        The weights that will be used to compute the loss function.

    classifier : torch.nn._WeightedLoss
        The loss function that will be used to train the network.

    optimizer : torch.optim.Opimizer
        The optimizer that will be used to train the network.
    """

    def __init__(self, dataset, device, hyperparams):
        """Defines and initializes the network.

        Parameters
        ----------
        dataset : datasets.HyperDataset
            The properties of some layers depend on the target dataset.

        device : str
            The computational device that will be used to run the network.

        hyperparams : dict
            Some hyperparameters are used to define the network structure and its behavior.
        """

        super(CNN2D_Residual, self).__init__()

        self.bands = dataset.bands
        self.patch_size = dataset.patch_size
        self.classes_count = dataset.classes_count
        self.device = device
        self.Z = hyperparams["latent_size"]

        self.latent_size = hyperparams["latent_size"]

        # 0. A mapping block changes the number of channels to 16
        self.map_0 = Discriminator_ConvBlock(hyperparams, self.bands, 16, old_norm=True)

        # 1. The first residual stage consists of three residual blocks that use 16 channels
        #    It also downsamples the spatial resolution by a factor of 2
        self.stage_1 = Discriminator_ResStage(hyperparams, 3, 16, 16, strides=(2, 2), old_norm=True)

        # 2. The second residual stage consists of three residual blocks that use 32 channels
        #    It also downsamples the spatial resolution by a factor of 2
        self.stage_2 = Discriminator_ResStage(hyperparams, 3, 16, 32, strides=(2, 2), old_norm=True)

        # 3. The third residual stage consists of three residual blocks that use Z channels
        #    That is the same number of channels as the latent space size used on the generator
        #    It also downsamples the spatial resolution by a factor of 2
        self.stage_3 = Discriminator_ResStage(hyperparams, 3, 32, self.Z, strides=(2, 2), old_norm=True)

        # 4. In order to fuse features from different stages, we need two additional convolutional blocks that map the
        #    low-resolution features to the high-resolution ones
        self.map_stage1 = Discriminator_ConvBlock(hyperparams, 16, self.Z, strides=(4, 4), old_norm=True)
        self.map_stage2 = Discriminator_ConvBlock(hyperparams, 32, self.Z, strides=(2, 2), old_norm=True)

        # 5. Now we apply an average pooling layer to get the final feature vector
        #    Each sample has been downsampled to 4x4 pixels
        self.avg_pool = nn.AvgPool2d((4, 4))

        # Calculamos el tamaño do espazo latente final pasando ruido por todas las capas
        # 5. Now we apply a fully-connected layer to the flattened output of the last convolutional block to get the
        #    final feature vector
        example = torch.randn(1, self.bands, self.patch_size, self.patch_size)
        example = self.map_0(example)
        example1 = self.stage_1(example)
        example2 = self.stage_2(example1)
        example3 = self.stage_3(example2)
        example = self.map_stage1(example1) + self.map_stage2(example2) + example3
        example = self.avg_pool(example)
        out = example.view(example.shape[0], -1).shape[1]

        # 6. The final layer is a fully connected layer that outputs the class probabilities
        self.full = nn.Linear(out, self.classes_count)

        self.apply(self.initialize_weights)

        # From PyTorch docs:
        #   "If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
        #    Parameters of a model after .cuda() will be different objects with those before the call."
        self.to(self.device)

        # Each class is given the same weight in order to point out that we are interested in all of them equally
        self.classifier_weights = torch.ones(self.classes_count, device=self.device)
        self.classifier = nn.CrossEntropyLoss(weight=self.classifier_weights)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=hyperparams["learning_rate"], betas=(0.5, 0.999))

    def forward(self, tensor):
        """Processes a batch of samples.

        Parameters
        ----------
        tensor : torch.Tensor
            The input batch.

        Returns
        -------
        tuple
            The result of applying the network's different transformations over the input batch.

            More specifically, the output is a tuple containing the following tensors:

            - The output of the last layer (class probabilities).
            - The features that are used to compute the output.
        """

        out_start = self.map_0(tensor)

        out_stage1 = self.stage_1(out_start)
        out_stage2 = self.stage_2(out_stage1)
        out_stage3 = self.stage_3(out_stage2)

        out = self.map_stage1(out_stage1) + self.map_stage2(out_stage2) + out_stage3
        out = self.avg_pool(out)

        features = out.view(out.shape[0], -1)
        logits = self.full(features)
        # The softmax function is applied outside

        return (logits, features)

    @staticmethod
    def initialize_weights(child):
        """Receives a layer and initializes its weights and biases if needed.

        Parameters
        ----------
        child : torch.nn.Module
            The layer.
        """

        if isinstance(child, nn.Conv2d):
            nn.init.xavier_normal_(child.weight)
            nn.init.constant_(child.bias, 0)

        elif isinstance(child, nn.Linear):
            nn.init.xavier_normal_(child.weight)
            nn.init.constant_(child.bias, 0)


# --------------------------------------------------------------------------------------------

"""
To avoid importing the dataset class used to train the judge model, we define a mock version of it here. 
The dataset only was used to get the number of bands, patch size, and classes count. Also, we define a
dictionary with the hyperparameters used to create the model.
"""


class DatasetMock:
    def __init__(self, bands=5, patch_size=32, classes_count=10):
        self.bands = bands
        self.patch_size = patch_size
        self.classes_count = classes_count


HYPERPARAMS = {
    "latent_size": 32,
    "activation": "prelu",
    "p_dropout": 0.1,
    "learning_rate": 1e-8,
}
