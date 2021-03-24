import numpy as np


class Method:
    def __call__(self, x):
        raise NotImplementedError()

    def disable(self, index, x):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class Max(Method):
    def __call__(self, x):
        return x.argmax()

    def disable(self, index, x):
        x[index] = -np.inf

    def __repr__(self):
        return 'Max'


class Top(Method):
    def __init__(self, n=5):
        self.n = n

    def __call__(self, x):
        return np.random.choice(np.argpartition(x, -self.n, axis=None)[-self.n:])

    def disable(self, index, x):
        x[index] = -np.inf

    def __repr__(self):
        return 'Top' + str(self.n)


class PowerProb(Method):
    def __init__(self, power=4):
        self.power = power

    def __call__(self, x):
        raveled = np.power(np.ravel(x), self.power)
        return np.random.choice(np.arange(x.size), p=(raveled / np.sum(raveled)))

    def disable(self, index, x):
        x[index] = 0.0

    def __repr__(self):
        return 'PowerProb' + str(self.power)


class RandomInference(Method):
    def __call__(self, x):
        return np.random.choice(np.arange(x.size))

    def disable(self, index, x):
        pass

    def __repr__(self):
        return 'RandomInference'
