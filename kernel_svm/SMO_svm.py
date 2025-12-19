import numpy as np
from collections import OrderedDict

class KernelCache:
    def __init__(self, max_size=200):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        return None

    def put(self, idx, col):
        self.cache[idx] = col
        self.cache.move_to_end(idx)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)


class SVM:
    def __init__ (self, kernel = 'rbf', C = 1.0, coef = 0, d = 3, gamma = 1e-2, random_state = 42):
        self.kernel_type = kernel
        self.C = C
        self.w = None
        self.b = 0
        self.alphas = None

        self.coef = coef
        self.d = d
        self.gamma = gamma
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)


    def kernel(self, X, x):
        if self.kernel_type == 'linear':
            return np.dot(X, x)
        elif self.kernel_type == 'sigmoid':
            return np.tanh(self.gamma * np.dot(X,x) + self.coef)
        elif self.kernel_type == 'poly':
            return (np.dot(X, x)*self.gamma + self.coef) ** self.d
        else:
            diff = X - x
            return np.exp(-self.gamma * np.sum(diff*diff, axis=1))
    
    def get_kernel_column(self, X, i):
        col = self.cache.get(i)
        if col is not None:
            return col
        
        col = self.kernel(X, X[i])
        self.cache.put(i, col)
        return col

    def f(self, X, i):
        K = self.get_kernel_column(X, i)
        return np.sum(self.alphas * self.y * K) + self.b
    

    def fit(self, X, y, max_passes = 5, max_iter = 10000, tol = 1e-3, eps = 1e-3, cache_size = 500):
        n, d = X.shape
        self.y = y
        self.X = X

        self.alphas = np.zeros(n)
        self.b = 0
        self.w = np.zeros(d) if self.kernel_type == 'linear' else None

        self.tol = tol
        self.eps = eps

        self.cache = KernelCache(max_size = cache_size)

        E = -y.copy()

        passes = 0
        iters = 0

        while passes < max_passes and iters < max_iter:
            num_changed = 0

            for i2 in range(n):
                E2 = E[i2]
                r2 = E2 * y[i2]

                if ((r2 < -self.tol and self.alphas[i2] < self.C) or
                    (r2 > self.tol and self.alphas[i2] > 0)):

                    non_bound = np.where((self.alphas > 0) & (self.alphas < self.C))[0]
                    if len(non_bound) > 1:
                        i1 = non_bound[np.argmax(np.abs(E[non_bound] - E2))]
                    else:
                        i1 = self.rng.randint(0, n)

                    if i1 == i2:
                        continue

                    alph1_old = self.alphas[i1]
                    alph2_old = self.alphas[i2]
                    y1, y2 = y[i1], y[i2]

                    if y1 != y2:
                        L = max(0, alph2_old - alph1_old)
                        H = min(self.C, self.C + alph2_old - alph1_old)
                    else:
                        L = max(0, alph2_old + alph1_old - self.C)
                        H = min(self.C, alph2_old + alph1_old)

                    if L == H:
                        continue

                    K1 = self.get_kernel_column(X, i1)
                    K2 = self.get_kernel_column(X, i2)
                    K11, K22, K12 = K1[i1], K2[i2], K1[i2]

                    eta = K11 + K22 - 2 * K12

                    E1 = E[i1]

                    if eta > 0:
                        a2 = alph2_old + y2 * (E1 - E2) / eta
                        a2 = np.clip(a2, L, H)
                    else:
                        def obj(a2_test):
                            a1_test = alph1_old + y1 * y2 * (alph2_old - a2_test)
                            return a1_test + a2_test - 0.5 * (K11 * a1_test ** 2 + K22 * a2_test ** 2 + 2 * K12 * a1_test * a2_test)
                        
                        if obj(L) > obj(H) + self.eps:
                            a2 = L
                        elif obj(H) > obj(L) + self.eps:
                            a2 = H
                        else:
                            a2 = alph2_old

                    if abs(a2 - alph2_old) < self.eps * (a2 + alph2_old + self.eps):
                        continue

                    a1 = alph1_old + y1 * y2 * (alph2_old - a2)

                    b1 = self.b - E1 - y1 * (a1 - alph1_old) * K11 - y2 * (a2 - alph2_old) * K12

                    b2 = self.b - E2 - y1 * (a1 - alph1_old) * K12 - y2 * (a2 - alph2_old) * K22

                    b_old = self.b

                    if 0 < a1 < self.C:
                        self.b = b1
                    elif 0 < a2 < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    if (self.kernel_type == 'linear'):
                        self.w += y1 * (a1 - alph1_old) * X[i1] + y2 * (a2 - alph2_old) * X[i2]

                    self.alphas[i1] = a1
                    self.alphas[i2] = a2

                    for i in range(n):
                        E[i] += y1 * (a1 - alph1_old) * K1[i] + y2 * (a2 - alph2_old) * K2[i] + self.b - b_old

                    num_changed += 1
                    iters += 1

            passes = passes + 1 if num_changed == 0 else 0

        return self
    
    def decision_function(self, X):
        if self.kernel_type == 'linear':
            return X @ self.w + self.b
        
        sv = self.alphas > 0
        alphas_sv = self.alphas[sv]
        X_sv = self.X[sv]
        y_sv = self.y[sv]

        scores = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            K = self.kernel(X_sv, x)
            scores[i] = np.sum(alphas_sv * y_sv * K) + self.b

        return scores


    def predict(self, X):
        scores = self.decision_function(X)
        return np.sign(scores)