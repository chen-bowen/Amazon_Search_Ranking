__all__ = ["run_training"]


def __getattr__(name: str):
    if name == "run_training":
        from .train_reranker import run_training
        return run_training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
