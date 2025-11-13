from .simple_conv import SimpleConv
import torch.nn as nn
import torch.nn.functional as F

class SimpleConv3Chan(SimpleConv):
    """
    A simple convolutional neural network for CIFAR10 classification tasks based on the simple_conv network
    for mnist classification. See the simple_conv.py file.
    """

    def __init__(self):
        """
        Initializes the SimpleConv model.

        The model architecture includes:
        - Two 2D convolutional layers with ReLU activations and max pooling.
        - A fully connected layer with batch normalization and ReLU activation.
        - A final fully connected layer for classification.
        """
        super().__init__()
        self._c1 = nn.Conv2d(3, 20, kernel_size=5)
        self._c2 = nn.Conv2d(20, 50, kernel_size=5)
        self._f1 = nn.Linear(1250, 50)
        self._f2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Defines the forward pass of the SimpleConv model.

        Args:
        - x (torch.Tensor): Input tensor of shape `(batch_size, 1, H, W)` where
          H and W are the height and width of the input image (e.g., 28x28 for MNIST).

        Returns:
        - torch.Tensor: Output tensor of shape `(batch_size, 10)` containing raw class scores.
        """
        # Apply the first convolutional layer and ReLU activation
        x = F.relu(self._c1(x))

        # Apply max pooling with a 2x2 kernel and stride 2
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Apply the second convolutional layer and ReLU activation
        x = F.relu(self._c2(x))

        # Apply max pooling with a 2x2 kernel and stride 2
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten the feature maps into a vector of size 800
        x = x.view(-1, 1250)

        # Apply the first fully connected layer with batch normalization and ReLU activation
        x = F.relu(self._f1(x))

        # Apply the final fully connected layer for classification
        x = self._f2(x)

        return x
