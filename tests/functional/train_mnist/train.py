from hyperpyyaml import load_hyperpyyaml

import frarch as fr
from frarch.utils.data import build_experiment_structure
from frarch.utils.stages import Stage


class MNISTTrainer(fr.train.ClassifierTrainer):
    def _forward(self, batch, stage):
        inputs, _ = batch
        inputs = inputs.to(self.device)
        return self.modules.model(inputs)

    def _compute_loss(self, predictions, batch, stage):
        _, labels = batch
        labels = labels.to(self.device)
        return self.hparams["loss"](predictions, labels)

    def _on_stage_end(self, stage, loss=None, epoch=None):
        if stage == Stage.VALID:
            if self.checkpointer is not None:
                self.checkpointer.save(epoch=self.current_epoch, current_step=self.step)


if __name__ == "__main__":
    hparam_file, args = fr.parse_arguments()

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
