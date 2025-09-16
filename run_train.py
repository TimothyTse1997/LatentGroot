import click
import gc
import json
import inspect
from pathlib import Path

import trainer


def get_trainer_dict():
    elements = inspect.getmembers(trainer, inspect.isclass)
    element_dict = dict(elements)
    trainer_dict = {k: v for k, v in element_dict.items() if k.endswith("Trainer")}
    return trainer_dict


@click.command()
@click.option("--trainer_config_fname", default="./configs/trainer/default.json")
def main(trainer_config_fname):
    trainer_dict = get_trainer_dict()
    trainer_configs = json.load(open(trainer_config_fname, "r"))
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
        trainer = trainer_dict[trainer_name](**trainer_kwargs)
        trainer.fit()


if __name__ == "__main__":
    main()
