#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

template <typename scalar_t>
__global__ void count_valid_kernel(
    const scalar_t* values,
    int64_t rows,
    int64_t top_k,
    double threshold,
    int64_t* counts) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int64_t base = row * top_k;
    int64_t count = 0;
    for (int64_t k = 0; k < top_k; ++k) {
        if (static_cast<double>(values[base + k]) >= threshold) {
            ++count;
        }
    }
    counts[row] = count;
}

template <typename scalar_t>
__global__ void scatter_valid_kernel(
    const scalar_t* values,
    const int64_t* indices,
    int64_t rows,
    int64_t seq_len,
    int64_t top_k,
    double threshold,
    const int64_t* offsets,
    int64_t* batch_out,
    int64_t* pos_out,
    int64_t* feat_out,
    scalar_t* value_out) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int64_t base = row * top_k;
    int64_t out = offsets[row];
    const int64_t batch_idx = row / seq_len;
    const int64_t pos_idx = row % seq_len;

    for (int64_t k = 0; k < top_k; ++k) {
        const scalar_t value = values[base + k];
        if (static_cast<double>(value) < threshold) {
            continue;
        }
        batch_out[out] = batch_idx;
        pos_out[out] = pos_idx;
        feat_out[out] = indices[base + k];
        value_out[out] = value;
        ++out;
    }
}

void check_cuda_launch(const char* kernel_name) {
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, kernel_name, " failed: ", cudaGetErrorString(error));
}

}  // namespace

std::vector<torch::Tensor> compact_topk_threshold_cuda(
    torch::Tensor top_vals,
    torch::Tensor top_idx,
    double threshold) {
    TORCH_CHECK(top_vals.is_cuda(), "compact_topk_threshold_cuda expects CUDA tensors");
    TORCH_CHECK(top_idx.is_cuda(), "compact_topk_threshold_cuda expects CUDA tensors");

    auto values = top_vals.contiguous();
    auto indices = top_idx.contiguous();

    const auto batch = values.size(0);
    const auto seq_len = values.size(1);
    const auto top_k = values.size(2);
    const auto rows = batch * seq_len;

    auto long_options = indices.options().dtype(torch::kLong);
    auto counts = torch::zeros({rows}, long_options);

    if (rows == 0) {
        auto empty_long = torch::empty({0}, long_options);
        auto empty_values = torch::empty({0}, values.options());
        return {empty_long, empty_long.clone(), empty_long.clone(), empty_values};
    }

    const int threads = 256;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "compact_topk_threshold_cuda_count", [&] {
        count_valid_kernel<scalar_t><<<blocks, threads>>>(
            values.data_ptr<scalar_t>(),
            rows,
            top_k,
            threshold,
            counts.data_ptr<int64_t>()
        );
    });
    check_cuda_launch("count_valid_kernel");

    auto cumsum_counts = counts.cumsum(0, torch::kLong);
    const int64_t valid_count = cumsum_counts[rows - 1].item<int64_t>();

    auto offsets = torch::zeros({rows}, long_options);
    if (rows > 1) {
        offsets.slice(0, 1).copy_(cumsum_counts.slice(0, 0, rows - 1));
    }

    auto batch_out = torch::empty({valid_count}, long_options);
    auto pos_out = torch::empty({valid_count}, long_options);
    auto feat_out = torch::empty({valid_count}, long_options);
    auto value_out = torch::empty({valid_count}, values.options());

    if (valid_count == 0) {
        return {batch_out, pos_out, feat_out, value_out};
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "compact_topk_threshold_cuda_scatter", [&] {
        scatter_valid_kernel<scalar_t><<<blocks, threads>>>(
            values.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            rows,
            seq_len,
            top_k,
            threshold,
            offsets.data_ptr<int64_t>(),
            batch_out.data_ptr<int64_t>(),
            pos_out.data_ptr<int64_t>(),
            feat_out.data_ptr<int64_t>(),
            value_out.data_ptr<scalar_t>()
        );
    });
    check_cuda_launch("scatter_valid_kernel");

    return {batch_out, pos_out, feat_out, value_out};
}
