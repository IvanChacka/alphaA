from imports import *


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_valid=None, y_valid=None): ...

    @abstractmethod
    def predict(self, X) -> np.ndarray: ...

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path | str):
        return joblib.load(path)

    def get_feature_importance(self): return None
    def get_params(self) -> dict: return self.model.get_params()
