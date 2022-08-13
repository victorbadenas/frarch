# Running an experiment

Frarch provides training classes such as [`ClassifierTrainer`](https://victorbadenas.github.io/frarch/source/packages/frarch.train.classifier_trainer.html) which provides methods to train a classifier model.

## Example Python trainer script

In this example we present a sample training script for training the MNIST dataset.

```python
from hyperpyyaml import load_hyperpyyaml

from frarch.parser import parse_arguments
from frarch.utils.data import build_experiment_structure
from frarch.utils.enums.stages import Stage
from frarch.train.classifier_trainer import ClassifierTrainer


class MNISTTrainer(ClassifierTrainer):
    def forward(self, batch, stage):
        inputs, _ = batch
        inputs = inputs.to(self.device)
        return self.modules.model(inputs)

    def compute_loss(self, predictions, batch, stage):
        _, labels = batch
        labels = labels.to(self.device)
        return self.hparams["loss"](predictions, labels)

    def on_stage_end(self, stage, loss=None, epoch=None):
        if stage == Stage.VALID:
            if self.checkpointer is not None:
                self.checkpointer.save(epoch=self.current_epoch, current_step=self.step)


if __name__ == "__main__":
    hparam_file, args = parse_arguments()

    with open(hparam_file, "r") as hparam_file_handler:
        hparams = load_hyperpyyaml(
            hparam_file_handler, args, overrides_must_match=False
        )

    build_experiment_structure(
        hparam_file,
        overrides=args,
        experiment_folder=hparams["experiment_folder"],
        debug=hparams["debug"],
    )

    trainer = MNISTTrainer(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )

    trainer.fit(
        train_set=hparams["train_dataset"],
        valid_set=hparams["valid_dataset"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
```

And the hparams yaml file used to configure the experiment:

```yaml
# seeds
seed: 42
__set_seed: !apply:torch.manual_seed [!ref <seed>]
experiment_name: "mnist"
experiment_folder: "results/mnist_demo/"
device: "cpu"

# data folder
data_folder: /tmp/

# training parameters
epochs: 2
batch_size: 128
shuffle: True
num_clases: 10

transform_tensor: !new:torchvision.transforms.ToTensor
preprocessing: !new:torchvision.transforms.Compose
    transforms: [
        !ref <transform_tensor>,
    ]

# dataset object
train_dataset: !new:torchvision.datasets.MNIST
    root: !ref <data_folder>
    train: true
    download: true
    transform: !ref <preprocessing>

valid_dataset: !new:torchvision.datasets.MNIST
    root: !ref <data_folder>
    train: false
    download: true
    transform: !ref <preprocessing>

# dataloader options
dataloader_options:
    batch_size: !ref <batch_size>
    shuffle: !ref <shuffle>
    num_workers: 8

opt_class: !name:torch.optim.Adam
    lr: 0.001

loss: !new:torch.nn.CrossEntropyLoss

model: !apply:torchvision.models.vgg11
    pretrained: false

modules:
    model: !ref <model>

checkpointer: !new:frarch.modules.checkpointer.Checkpointer
    save_path: !ref <experiment_folder>
    modules: !ref <modules>

```

For the code execution run:

```bash
python train.py mnist.yaml
```

Additional params for the trainer script are as follows:


|                    Argument                    | Description                                                                                                               |
|:----------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------|
|                   -h, --help                   | show this help message and exit                                                                                           |
|                     --debug                    | Run the experiment with only a few batches for all datasets, to ensure code runs without crashing.                        |
|         --debug_batches DEBUG_BATCHES          | Number of batches to run in debug mode.                                                                                   |
|                 --device DEVICE                | The device to run the experiment on (e.g. 'cuda:0' or 'cpu')                                                               |
|                 --noprogressbar                | This flag disables the data loop progressbars.                                                                            |
| --ckpt_interval_minutes CKPT_INTERVAL_MINUTES  | Amount of time between saving intra-epoch checkpoints in minutes. If non-positive, intra-epoch checkpoints are not saved. |
|               --log_file LOG_FILE              | file to save log lines to.                                                                                                |
