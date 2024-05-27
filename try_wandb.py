import wandb

sweep_configuration = {
    "name": "my-awesome-sweep",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "method": "grid",
    "parameters": {"a": {"values": [1, 2, 3, 4]}},
}


def my_train_func():
    # read the current value of parameter "a" from wandb.config
    wandb.init()
    import pdb;pdb.set_trace();
    a = wandb.config.a

    wandb.log({"a": a, "accuracy": a + 1})

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_configuration)

    # run the sweep
    wandb.agent(sweep_id, function=my_train_func)