import numpy as np


class Method:
    def __call__(self, x) -> int:
        raise NotImplementedError

    def disable(self, index, x) -> None:
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class Max(Method):
    def __call__(self, x) -> int:
        return x.argmax()

    def disable(self, index, x) -> None:
        x[index] = -np.inf

    def __repr__(self):
        return 'Max'


class Top(Method):
    def __init__(self, n: int = 5):
        self.n = n

    def __call__(self, x) -> int:
        return np.random.choice(np.argpartition(x, -self.n, axis=None)[-self.n:])

    def disable(self, index, x) -> None:
        x[index] = -np.inf

    def __repr__(self):
        return f'Top{self.n}'
