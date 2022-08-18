# Pytorch_spmm_COO
Pytorch CUDA/C++ extension of sparse dense matrix multiplication in COO format. Current implementation only supports GPU. 

Setup:

```
python setup.py install
```
### Sparse Dense Matrix Multiplication

```
spmm_coo_sum(src, row, col, res_dim) -> torch.Tensor
```
Matrix product of a sparse matrix with a dense matrix.

#### Parameters

* **src** *(Tensor)* - The dense matrix.
* **row** *(LongTensor)* - The row tensor of the sparse matrix.
* **col** *(LongTensor)* - The column tensor of the sparse matrix.
* **res_dim** *(int)* - The first dimension of the result matrix.

#### Returns

* **out** *(Tensor)* - The dense output matrix.

#### Example

```python
import torch
from spmm_coo import spmm_coo_sum
row = torch.tensor([0, 0, 1, 2, 2])
col = torch.tensor([0, 2, 1, 0, 1])
src = torch.Tensor([[1, 4], [2, 5], [3, 6]], device="cuda:0", dtype=torch.float)
res_dim = 3
out = spmm_coo_sum(src, row, col, res_dim)
```

```
print(out)
tensor([[7.0, 10.0],
        [5.0, 11.0],
        [1.0, 4.0]])
```
