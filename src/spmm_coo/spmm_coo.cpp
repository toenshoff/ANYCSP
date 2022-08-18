#include <torch/script.h>

#include "utils.h"
#include "cuda/spmm_coo_cuda.h"

std::tuple<torch::Tensor, torch::optional<torch::Tensor>> spmm_coo_forward(
    const torch::Tensor src,
    const torch::Tensor edge_start,
    const torch::Tensor edge_end,
    int64_t res_dim,
    std::string reduce)
{
  if (src.device().is_cuda())
    return spmm_coo_cuda(src, edge_start, edge_end, res_dim, reduce);
  else
    AT_ERROR("Source Tensor not in GPU!");
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class SpmmCOOMax : public torch::autograd::Function<SpmmCOOMax>
{
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable edge_start,
                               Variable edge_end,
                               int64_t res_dim)
  {
    ctx->saved_data["src_shape"] = src.sizes();
    auto result = spmm_coo_forward(src, edge_start, edge_end, res_dim, "max");
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result).value();
    ctx->save_for_backward({arg_out});
    ctx->mark_non_differentiable({arg_out});
    return {out, arg_out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs)
  {
    auto grad_out = grad_outs[0];
    auto arg_out = ctx->get_saved_variables()[0];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    src_shape[0] += 1;
    auto grad_in = torch::zeros(src_shape, grad_out.options());
    grad_in.scatter_(0, arg_out, grad_out, "add");
    grad_in = grad_in.narrow(0, 0, src_shape[0] - 1);
    return {grad_in, Variable(), Variable(), Variable()};
  }
};

class SpmmCOOSum : public torch::autograd::Function<SpmmCOOSum>
{
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable edge_start,
                               Variable edge_end,
                               int64_t res_dim)
  {
    // ctx->saved_data["dim"] = dim_size;
    ctx->saved_data["src_shape"] = src.sizes();
    auto result = spmm_coo_forward(src, edge_start, edge_end, res_dim, "sum");
    auto out = std::get<0>(result);
    ctx->save_for_backward({edge_start, edge_end});
    ctx->mark_non_differentiable({edge_start, edge_end});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs)
  {
    auto grad_out = grad_outs[0];
    auto edge_start = ctx->get_saved_variables()[0];
    auto edge_end = ctx->get_saved_variables()[1];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    auto result = spmm_coo_forward(grad_out, edge_end, edge_start, src_shape[0], "sum");
    auto grad_in = std::get<0>(result);
    return {grad_in, Variable(), Variable(), Variable()};
  }
};

class SpmmCOOMean : public torch::autograd::Function<SpmmCOOMean>
{
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable edge_start,
                               Variable edge_end,
                               int64_t res_dim)
  {
    ctx->saved_data["src_shape"] = src.sizes();
    auto result = spmm_coo_forward(src, edge_start, edge_end, res_dim, "sum");
    auto out = std::get<0>(result);
    // compute degree of elements in result tensor
    auto ones = torch::ones(size(src,0), src.options());
    result = spmm_coo_forward(ones, edge_start, edge_end, res_dim, "sum");
    auto degree = std::get<0>(result);
    degree.masked_fill_(degree < 1, 1);
    // divide result tensor by degree
    degree = broadcast(degree, out, 0);
    if (out.is_floating_point())
      out.true_divide_(degree);
    else
      out.div_(degree, "floor");
    ctx->save_for_backward({edge_start, edge_end, degree});
    ctx->mark_non_differentiable({edge_start, edge_end, degree});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs)
  {
    auto grad_out = grad_outs[0].clone();
    auto saved = ctx->get_saved_variables();
    auto edge_start = saved[0];
    auto edge_end = saved[1];
    auto degree = saved[2];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    grad_out.true_divide_(degree);
    auto result = spmm_coo_forward(grad_out, edge_end, edge_start, src_shape[0], "sum");
    auto grad_in = std::get<0>(result);
    return {grad_in, Variable(), Variable(), Variable(), Variable()};
  }
};

std::tuple<torch::Tensor, torch::Tensor> spmm_coo_max(const torch::Tensor src,
                                                          const torch::Tensor edge_start,
                                                          const torch::Tensor edge_end,
                                                          int64_t res_dim)
{
  auto result = SpmmCOOMax::apply(src, edge_start, edge_end, res_dim);

  return std::make_tuple(result[0], result[1]);
}

torch::Tensor spmm_coo_sum(const torch::Tensor src,
                               const torch::Tensor edge_start,
                               const torch::Tensor edge_end,
                               int64_t res_dim)
{

  return SpmmCOOSum::apply(src, edge_start, edge_end, res_dim)[0];
}

torch::Tensor spmm_coo_mean(const torch::Tensor src,
                                const torch::Tensor edge_start,
                                const torch::Tensor edge_end,
                                int64_t res_dim)
{

  return SpmmCOOMean::apply(src, edge_start, edge_end, res_dim)[0];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("spmm_coo_sum", &spmm_coo_sum, "Sum Sparse Mul forward");
  m.def("spmm_coo_max", &spmm_coo_max, "Max Sparse Mul forward");
  m.def("spmm_coo_mean", &spmm_coo_mean, "Mean Sparse Mul forward");
}