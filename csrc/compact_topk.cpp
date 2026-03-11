#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <vector>

std::vector<torch::Tensor> compact_topk_threshold_cpu(
    torch::Tensor top_vals,
    torch::Tensor top_idx,
    double threshold);

#ifdef WITH_CUDA
std::vector<torch::Tensor> compact_topk_threshold_cuda(
    torch::Tensor top_vals,
    torch::Tensor top_idx,
    double threshold);
#endif

namespace {

void validate_inputs(const torch::Tensor& top_vals, const torch::Tensor& top_idx) {
    TORCH_CHECK(top_vals.dim() == 3, "top_vals must have shape [batch, seq_len, top_k]");
    TORCH_CHECK(top_vals.sizes() == top_idx.sizes(), "top_vals and top_idx must have the same shape");
    TORCH_CHECK(top_idx.scalar_type() == torch::kLong, "top_idx must have dtype torch.int64");
    TORCH_CHECK(top_vals.device() == top_idx.device(), "top_vals and top_idx must be on the same device");
}

}  // namespace

std::vector<torch::Tensor> compact_topk_threshold_cpu(
    torch::Tensor top_vals,
    torch::Tensor top_idx,
    double threshold) {
    validate_inputs(top_vals, top_idx);
    TORCH_CHECK(!top_vals.is_cuda(), "compact_topk_threshold_cpu expects CPU tensors");

    auto values = top_vals.contiguous();
    auto indices = top_idx.contiguous();

    const auto batch = values.size(0);
    const auto seq_len = values.size(1);
    const auto top_k = values.size(2);
    const auto total = values.numel();

    int64_t valid_count = 0;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "compact_topk_threshold_cpu_count", [&] {
        const auto* value_ptr = values.data_ptr<scalar_t>();
        for (int64_t linear = 0; linear < total; ++linear) {
            if (static_cast<double>(value_ptr[linear]) >= threshold) {
                ++valid_count;
            }
        }
    });

    auto long_options = indices.options().dtype(torch::kLong);
    auto batch_out = torch::empty({valid_count}, long_options);
    auto pos_out = torch::empty({valid_count}, long_options);
    auto feat_out = torch::empty({valid_count}, long_options);
    auto value_out = torch::empty({valid_count}, values.options());

    auto* batch_ptr = batch_out.data_ptr<int64_t>();
    auto* pos_ptr = pos_out.data_ptr<int64_t>();
    auto* feat_ptr = feat_out.data_ptr<int64_t>();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "compact_topk_threshold_cpu_fill", [&] {
        const auto* value_ptr = values.data_ptr<scalar_t>();
        const auto* index_ptr = indices.data_ptr<int64_t>();
        auto* value_out_ptr = value_out.data_ptr<scalar_t>();

        int64_t out = 0;
        for (int64_t linear = 0; linear < total; ++linear) {
            const auto value = value_ptr[linear];
            if (static_cast<double>(value) < threshold) {
                continue;
            }
            const int64_t row = linear / top_k;
            const int64_t batch_idx = row / seq_len;
            const int64_t pos_idx = row % seq_len;
            batch_ptr[out] = batch_idx;
            pos_ptr[out] = pos_idx;
            feat_ptr[out] = index_ptr[linear];
            value_out_ptr[out] = value;
            ++out;
        }
    });

    return {batch_out, pos_out, feat_out, value_out};
}

std::vector<torch::Tensor> compact_topk_threshold(
    torch::Tensor top_vals,
    torch::Tensor top_idx,
    double threshold) {
    validate_inputs(top_vals, top_idx);
    if (top_vals.is_cuda()) {
#ifdef WITH_CUDA
        return compact_topk_threshold_cuda(top_vals, top_idx, threshold);
#else
        TORCH_CHECK(false, "This build does not include CUDA kernels.");
#endif
    }
    return compact_topk_threshold_cpu(top_vals, top_idx, threshold);
}

bool has_cuda() {
#ifdef WITH_CUDA
    return true;
#else
    return false;
#endif
}

std::string build_variant() {
#ifdef WITH_CUDA
    return "cuda";
#else
    return "cpu";
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "compact_topk_threshold",
        &compact_topk_threshold,
        "Fuse thresholding and compaction over top-k outputs"
    );
    m.def("has_cuda", &has_cuda, "Whether the extension was compiled with CUDA support");
    m.def("build_variant", &build_variant, "Build variant for the native extension");
}
