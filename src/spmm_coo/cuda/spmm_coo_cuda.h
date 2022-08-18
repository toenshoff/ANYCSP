#include <torch/extension.h>

#define CHECK_INPUT_DIM(x) AT_ASSERTM(x, "Input mismatch")
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>> spmm_coo_cuda(
    torch::Tensor src,
    const torch::Tensor edge_start,
    const torch::Tensor edge_end,
    int64_t res_dim,
    std::string reduce);
