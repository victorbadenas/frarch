from typing import List
from typing import Optional

from torch.nn import ModuleDict


class FreezableModuleDict(ModuleDict):
    def __init__(self, freeze: Optional[List[str]] = None, **kwargs):
        """Torch ModuleDict with freeze capabilities.

        Args:
            freeze (Optional[List[str]]): list of names of the parameter's to
                freeze.
            unfreeze (Optional[List[str]]): list of names of the parameter's to
                unfreeze.
            **kwargs: ModuleDict arguments.

        Raises:
            FreezeError: both freeze and unfreeze can't be set simultaneously.
        """
        super().__init__(**kwargs)
        if freeze:
            self.__freeze_module_layers(freeze)

    def __freeze_module_layers(self, freeze: List[str]) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name not in freeze
