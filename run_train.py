import click
import gc
import json
import inspect
from pathlib import Path

import trainer as trainer_lib


def get_trainer_dict():
    elements = inspect.getmembers(trainer_lib, inspect.isclass)
    element_dict = dict(elements)
    trainer_dict = {k: v for k, v in element_dict.items() if k.endswith("Trainer")}
    return trainer_dict


@click.command()
@click.option("--trainer_config_fname", default="./configs/trainer/default.json")
@click.option("--pretrained_checkpoint_path", default=None)
@click.option("--use_tensorboard/--no-use_tensorboard", default=False)
@click.option("--tensorboard_logdir", default=None)
def main(
    trainer_config_fname,
    pretrained_checkpoint_path=None,
    use_tensorboard=False,
    tensorboard_logdir=None,
):
    trainer_dict = get_trainer_dict()
    trainer_configs = json.load(open(trainer_config_fname, "r"))
    if tensorboard_logdir is not None and use_tensorboard:
        if not Path(tensorboard_logdir).exists():
            Path(tensorboard_logdir).mkdir()

    for trainer_id, tc in enumerate(trainer_configs):
        trainer_name = tc["trainer_name"]
        trainer_kwargs = tc["trainer_kwargs"]
        checkpoint_dir = Path(trainer_kwargs.get("checkpoint_dir"))
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir()
        checkpoint_fname = (
            Path(trainer_kwargs.get("checkpoint_dir")) / "trainer_config.json"
        )
        with open(checkpoint_fname, "w") as f:
            json.dump(tc, f, indent=4)

        print(f"start {trainer_id}th training with {trainer_name}")
        assert trainer_name in trainer_dict
        if not use_tensorboard:
            trainer = trainer_dict[trainer_name](**trainer_kwargs)
        else:
            trainer = trainer_lib.get_TensorboardTrainer(
                trainer_name, tensorboard_logdir, **trainer_kwargs
            )

        if pretrained_checkpoint_path is not None:
            trainer.load_checkpoint(pretrained_checkpoint_path)
        trainer.fit()


if __name__ == "__main__":
    main()
