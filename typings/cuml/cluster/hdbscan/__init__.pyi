from typing import Any

class HDBSCAN:
    def __init__(
        self,
        min_cluster_size: int = ...,
        min_samples: int = ...,
        **kwargs: Any,
    ) -> None: ...
    def fit_predict(self, X: Any) -> Any: ...
