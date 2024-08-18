import numpy as np


class NoGrad:
    _enabled = False

    def __enter__(self):
        self.prev = NoGrad._enabled
        NoGrad._enabled = True

    def __exit__(self, exc_type, exc_value, traceback):
        NoGrad._enabled = self.prev


def no_grad():
    return NoGrad()


class Tensor:
    def __init__(self, data, _children=(), requires_grad=False, dtype=None):
        if isinstance(data, Tensor):
            data = data.data
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype if dtype else np.float32)
        self.data = data
        self.shape = self.data.shape
        self.dtype = self.data.dtype

        self.requires_grad = requires_grad and not NoGrad._enabled
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._backward = lambda: None
        self._children = set(_children)
        self._id = id(self)
            
    def backward(self) -> None:
        if not self.requires_grad:
            raise RuntimeError(
                "Cannot call backward() on a tensor that does not require gradients."
            )
        self.grad = np.ones_like(self.data)

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for node in reversed(topo):
            node._backward()
            
    def size(self):
        return self.data.size

    def dim(self):
        return len(self.shape)

    def item(self):
        return self.data.item()
    
    def __hash__(self):
        return self._id

    def normal_(self, mean=0, std=1):
        self.data = np.random.normal(mean, std, self.shape)
        return self
    
    def uniform_(self, low=0, high=1):
        self.data = np.random.uniform(low, high, self.shape)
        return self
    
    def fill_(self, value):
        self.data.fill(value)
        return self

    def __repr__(self):
        # numpy representation
        np_str = np.array2string(
            self.data, separator=", ", precision=4, suppress_small=True
        )
        lines = np_str.split("\n")
        # add 'tensor(' at the beginning and extra spacing
        formatted_lines = ["tensor(" + lines[0]] + [
            " " * 8 + line.strip() for line in lines[1:]
        ]
        formatted_lines[-1] += ")"

        tensor_str = "\n".join(formatted_lines)
        return tensor_str + f"       dtype={self.data.dtype}, shape={self.shape}"

    def __getitem__(self, key):
        if isinstance(key, Tensor):
            key = key.data
        sliced_data = self.data[key]
        out = Tensor(sliced_data, _children=(self,), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                grad[key] = out.grad
                self.grad += grad

        out._backward = _backward
        return out

    def gather(self, dim, index):
        # ensure index is a Tensor
        if not isinstance(index, Tensor):
            index = Tensor(index)

        # create a list of slice objects for indexing
        slices = [slice(None)] * self.dim()

        # replace the slice at the specified dimension with the index array
        slices[dim] = index.data

        # use advanced indexing to gather the values
        gathered_data = self.data[tuple(slices)]

        out = Tensor(gathered_data, _children=(self,), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                np.add.at(grad, tuple(slices), out.grad)
                self.grad += grad

        out._backward = _backward
        return out

    def __gt__(self, other):
        assert isinstance(other, (int, float, Tensor))
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data > other.data, dtype=np.float32)
    
    def __eq__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data == other, dtype=np.float32)





