from typing import Any

class DBSCAN:
    def __init__(
        self,
        eps: float = ...,
        min_samples: int = ...,
        metric: str = ...,
        **kwargs: Any,
    ) -> None: ...
    def fit_predict(self, X: Any) -> Any: ...
