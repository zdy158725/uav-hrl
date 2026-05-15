from exp_article_test.raytune import smoke_test_hier_env, train_hier_v1


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=Warning)
    kwargs = {
        "seed": 0,
        "num_cpus": 8,
        "num_gpus": 0,
        "num_env_runners": 2,
    }
    print("Smoke test:", smoke_test_hier_env(seed=kwargs["seed"], num_red=1, num_blue=1, rollout_steps=2))
    print("Train kwargs:", kwargs)
    train_hier_v1(**kwargs)
