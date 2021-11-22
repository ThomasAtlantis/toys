import numpy as np


def build_binary_ops(this, that, grad_fn_1, grad_fn_2, values):
    requires_grad = this.requires_grad or that.requires_grad
    dependency = []
    if this.requires_grad:
        dependency.append(dict(tensor=this, grad_fn=grad_fn_1))
    if that.requires_grad:
        dependency.append(dict(tensor=that, grad_fn=grad_fn_2))
    return this.__class__(values, requires_grad, dependency)


def build_unary_ops(this, grad_fn, values):
    dependency = [dict(tensor=this, grad_fn=grad_fn)] if this.requires_grad else []
    return this.__class__(values, this.requires_grad, dependency)


def np_matmul(arr1, arr2):
    if arr1.ndim == 1 and arr2.ndim == 1:
        arr1 = np.mat(arr1).T
        arr2 = np.mat(arr2)
    return arr1 @ arr2


def as_tensor(obj):
    if not isinstance(obj, Tensor):
        obj = Tensor(obj)
    return obj

def register_op(func):
    setattr(Tensor, func.__name__, func)


class Tensor:

    def __init__(self, values, requires_grad=False, dependency=None):
        self._values = np.array(values)
        self.shape = self.values.shape

        self.grad = None
        if requires_grad: self.zero_grad()
        self.requires_grad = requires_grad

        if dependency is None: dependency = []
        self.dependency = dependency

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values):
        self._values = np.array(new_values)
        self.grad = None

    def zero_grad(self):
        self.grad = np.zeros(self.shape)

    def backward(self, grad=None):
        assert self.requires_grad, "Call backward() on a non-requires-grad tensor."
        assert not (grad is None and self.values.size > 1), "grad can be implicitly created only for scalar outputs"
        grad = 1.0 if grad is None else grad
        grad = np.array(grad)
        self.grad = self.grad + grad.astype(np.float)

        for dep in self.dependency:
            grad_for_dep = dep["grad_fn"](grad)
            dep["tensor"].backward(grad_for_dep)

    def __repr__(self):
        return "Tensor" + self._values.__repr__()[5:]

    def __matmul__(self, other):
        """ self @ other """
        return _matmul(self, as_tensor(other))

    def __rmatmul__(self, other):
        """ other @ self """
        return _matmul(as_tensor(other), self)

    def __mul__(self, other):
        """ self * other """
        return _mul(self, as_tensor(other))

    def __rmul__(self, other):
        """ other * self """
        return _mul(as_tensor(other), self)

    def __neg__(self):
        """ -self """
        return _neg(self)

    def __add__(self, other):
        """ self + other """
        return _add(self, as_tensor(other))

    def __radd__(self, other):
        """ other + self """
        return _add(as_tensor(other), self)

    def __sub__(self, other):
        """ self - other """
        return _add(self, -as_tensor(other))

    def __rsub__(self, other):
        """ other - self """
        return _add(-as_tensor(other), self)

    def reduce_mean(self, axis=None):
        """ self.reduce_mean(axis=x) """
        return _reduce_mean(self, axis)

    def reduce_sum(self, axis=None):
        """ self.reduce_sum(axis=x) """
        return _reduce_sum(self, axis)


def _mul(operand_1, operand_2):
    def grad_fn_1(grad):
        grad = grad * operand_2.values
        for _ in range(grad.ndim - operand_1.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(operand_1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def grad_fn_2(grad):
        grad = grad * operand_1.values
        for _ in range(grad.ndim - operand_2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(operand_2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    return build_binary_ops(
        operand_1, operand_2, grad_fn_1, grad_fn_2,
        operand_1.values * operand_2.values)


def _matmul(operand_1, operand_2):
    def grad_fn_1(grad):
        return np_matmul(grad, operand_2.values.T)

    def grad_fn_2(grad):
        return np_matmul(operand_1.values.T, grad)

    return build_binary_ops(
        operand_1, operand_2, grad_fn_1, grad_fn_2,
        np_matmul(operand_1.values, operand_2.values))


def _neg(operand_1):
    return build_unary_ops(operand_1, lambda grad: -grad, -operand_1.values)


def _add(operand_1, operand_2):
    # c = a + b
    # D_c / D_a = 1.0
    # D_c / D_b = 1.0
    def grad_fn_1(grad):
        # handle broadcasting (5, 3) + (3,) -> (5, 3)
        for _ in range(grad.ndim - operand_1.values.ndim):
            grad = grad.sum(axis=0)
        # handle broadcasting (5, 3) + (1, 3) -> (5, 3)
        for i, dim in enumerate(operand_1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def grad_fn_2(grad):
        for _ in range(grad.ndim - operand_2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(operand_2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    return build_binary_ops(
        operand_1, operand_2, grad_fn_1, grad_fn_2,
        operand_1.values + operand_2.values)


# sum函数的axis制定了被reduce的轴，所以需要先expand_dims回来
# 对于mean函数，除以了这个轴上求和的元素总数，即shape[axis]
# x.shape => (2, 3, 4)
# y = x.mean(axis=0)
# y.shape => (3, 4)
# z = np.expand_dims(y, 0)
# z.shape => (1, 3, 4)
# np.repeat(z, 2, 0) => (2, 3, 4)
# 由于y中每个元素是由2个元素求的平均，grad还要除以2
def _reduce_sum(operand_1, axis=None):
    if axis is not None:
        repeat = operand_1.values.shape[axis]

    def grad_fn(grad):
        if axis is None:
            grad = grad * np.ones_like(operand_1.values)
        else:
            grad = np.expand_dims(grad, axis)
            grad = np.repeat(grad, repeat, axis)
            print(grad)
        return grad

    return build_unary_ops(
        operand_1, grad_fn, operand_1.values.sum(axis=axis))


def _reduce_mean(operand_1, axis=None):
    values = operand_1.values.mean(axis=axis)

    if axis is not None:
        repeat = operand_1.values.shape[axis]

    def grad_fn(grad):
        if axis is None:
            grad = grad / operand_1.values.size * np.ones_like(operand_1.values)
        else:
            grad = np.expand_dims(grad / repeat, axis)
            grad = np.repeat(grad, repeat, axis)
        return grad

    return build_unary_ops(operand_1, grad_fn, values)