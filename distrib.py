import typing as tp


def average_metrics(metrics: tp.Dict[str, float], count: int = 1) -> tp.Dict[str, float]:
    """Average metrics across distributed workers. Single-GPU stub: returns as-is."""
    return metrics
