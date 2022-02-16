import torch


class BypassModel(torch.nn.Module):
    """Bypass mock model.

    Module that returns the same argument that is given.
    """

    def __init__(self) -> None:
        super(BypassModel, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward for bypass model.

        Args:
            x (torch.Tensor): input to the model.

        Returns:
            torch.Tensor: same tensor as the input.
        """
        return x
