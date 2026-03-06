__all__ = ["RerankerTrainer", "MultiTaskTrainer"]


def __getattr__(name: str):
    if name == "RerankerTrainer":
        from .train_reranker import RerankerTrainer

        return RerankerTrainer
    if name == "MultiTaskTrainer":
        from .train_multi_task_reranker import MultiTaskTrainer

        return MultiTaskTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
