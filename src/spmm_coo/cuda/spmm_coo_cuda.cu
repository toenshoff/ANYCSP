#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

#include "spmm_coo_cuda.h"
#include "reducer.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

template <typename scalar_t, ReductionType REDUCE>
__global__ void spmm_coo_kernel(
    const scalar_t* __restrict__ src,
    const int64_t* __restrict__ row, 
    const int64_t* __restrict__ col,
    scalar_t* __restrict__ res,
    int64_t* __restrict__ arg_out,
    size_t hidden_dim,
    size_t N)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_id < N){
        int edge_index = thread_id / hidden_dim;
        int hidden_dim_index = thread_id % hidden_dim;

        int edge_start = __ldg(row + edge_index);
        int edge_end = __ldg(col + edge_index);
        scalar_t write_val = __ldg(src + edge_start*hidden_dim + hidden_dim_index);
        int res_index = edge_end*hidden_dim + hidden_dim_index;

        Reducer<scalar_t, REDUCE>::atomic_write(
                res + res_index, 
                write_val);

        //compute arg out tensor
        if(REDUCE == MIN || REDUCE == MAX){
            __syncthreads();
            if(res[res_index] == write_val)
                arg_out[res_index] = edge_start;
        }
    }            
}

std::tuple<torch::Tensor,torch::optional<torch::Tensor>> spmm_coo_cuda(
    torch::Tensor src, 
    const torch::Tensor edge_start, 
    const torch::Tensor edge_end,
    int64_t res_dim,
    std::string reduce)
{
    //check input
    CHECK_INPUT(src);
    CHECK_INPUT_DIM(edge_start.size(0) == edge_end.size(0));
    CHECK_INPUT(edge_start);
    CHECK_INPUT(edge_end);
    src = src.contiguous();
    
    size_t hidden_dim = 1;
    if(src.dim() == 2)
        hidden_dim = size(src, 1);
    size_t N = edge_end.numel()*hidden_dim;

    //create out and arg_out Tensor with given out_dim
    auto res_dims = src.sizes().vec();
    res_dims[0] = res_dim;
    torch::Tensor res = torch::empty(res_dims, src.options());
    torch::optional<torch::Tensor> arg_out = torch::nullopt;
    //torch::Tensor arg_out = torch::empty(0, src.options());
    int64_t *arg_out_data = nullptr;
    if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
        arg_out = torch::full_like(res,src.size(0),edge_start.options());
        arg_out_data = arg_out.value().data_ptr<int64_t>();
      }
    
    AT_DISPATCH_FLOATING_TYPES(src.type(), "_", [&] {
        auto src_data = src.data_ptr<scalar_t>();
        auto res_data = res.data_ptr<scalar_t>();
        auto edge_start_data = edge_start.data_ptr<int64_t>();
        auto edge_end_data = edge_end.data_ptr<int64_t>();

        AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
            res.fill_(Reducer<scalar_t, REDUCE>::init());

            spmm_coo_kernel<scalar_t, REDUCE><<<BLOCKS(N), THREADS>>>(
                src_data,
                edge_start_data,
                edge_end_data,
                res_data,
                arg_out_data,
                hidden_dim,
                N);
   
            res.masked_fill_(res == Reducer<scalar_t, REDUCE>::init(), (scalar_t)0);
        });     
    });

    checkCuda(cudaGetLastError());
    
    return std::make_tuple(res,arg_out);   
}
