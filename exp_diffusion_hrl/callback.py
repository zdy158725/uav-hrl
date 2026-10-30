from exp_article_test.callback import AlgoCallbacks


class DiffusionAlgoCallbacks(AlgoCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            env_index=env_index,
            **kwargs,
        )
        stats = base_env.get_sub_environments()[env_index].call_back_stats()
        episode.custom_metrics["prediction_ready_ratio"] = stats["prediction_ready_ratio"]
        episode.custom_metrics["prediction_latency_ms"] = stats["prediction_latency_ms"]
