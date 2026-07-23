#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <ATen/Parallel.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

namespace {

double benchmark_aoti_ms(
    torch::inductor::AOTIModelPackageLoader& loader,
    const torch::Tensor& x,
    int rounds,
    int warmup_rounds
) {
    for (int i = 0; i < warmup_rounds; ++i) {
        loader.run({x});
    }

    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < rounds; ++i) {
        loader.run({x});
    }
    const auto ended = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = ended - started;
    return elapsed.count() / rounds;
}

void parallel_series(
    int64_t total_series,
    int num_threads,
    const std::function<void(int64_t, int64_t)>& run_range
) {
    if (num_threads <= 1 || total_series <= 1) {
        run_range(0, total_series);
        return;
    }
    at::parallel_for(0, total_series, 1, run_range);
}

torch::Tensor manual_combined(const torch::Tensor& x, int num_threads) {
    const auto contiguous = x.contiguous();
    const int64_t batch = contiguous.size(0);
    const int64_t states = contiguous.size(1);
    const int64_t timesteps = contiguous.size(2);
    const int64_t total_series = batch * states;
    auto out = torch::empty({batch, states, 3}, contiguous.options());

    const float* in_ptr = contiguous.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    parallel_series(total_series, num_threads, [&](int64_t start, int64_t end) {
        for (int64_t series = start; series < end; ++series) {
            const int64_t series_offset = series * timesteps;
            float sum = 0.0F;
            float maximum = -std::numeric_limits<float>::infinity();
            float minimum = std::numeric_limits<float>::infinity();
            for (int64_t t = 0; t < timesteps; ++t) {
                const float value = in_ptr[series_offset + t];
                sum += value;
                maximum = value > maximum ? value : maximum;
                minimum = value < minimum ? value : minimum;
            }
            const int64_t out_offset = series * 3;
            out_ptr[out_offset] = sum / static_cast<float>(timesteps);
            out_ptr[out_offset + 1] = maximum;
            out_ptr[out_offset + 2] = minimum;
        }
    });
    return out;
}

torch::Tensor manual_one_feature(const torch::Tensor& x, int num_threads, const std::string& name) {
    const auto contiguous = x.contiguous();
    const int64_t batch = contiguous.size(0);
    const int64_t states = contiguous.size(1);
    const int64_t timesteps = contiguous.size(2);
    const int64_t total_series = batch * states;
    auto out = torch::empty({batch, states, 1}, contiguous.options());

    const float* in_ptr = contiguous.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    parallel_series(total_series, num_threads, [&](int64_t start, int64_t end) {
        for (int64_t series = start; series < end; ++series) {
            const int64_t series_offset = series * timesteps;
            float value_out = 0.0F;
            if (name == "mean") {
                float sum = 0.0F;
                for (int64_t t = 0; t < timesteps; ++t) {
                    sum += in_ptr[series_offset + t];
                }
                value_out = sum / static_cast<float>(timesteps);
            } else if (name == "maximum") {
                float maximum = -std::numeric_limits<float>::infinity();
                for (int64_t t = 0; t < timesteps; ++t) {
                    const float value = in_ptr[series_offset + t];
                    maximum = value > maximum ? value : maximum;
                }
                value_out = maximum;
            } else {
                float minimum = std::numeric_limits<float>::infinity();
                for (int64_t t = 0; t < timesteps; ++t) {
                    const float value = in_ptr[series_offset + t];
                    minimum = value < minimum ? value : minimum;
                }
                value_out = minimum;
            }
            out_ptr[series] = value_out;
        }
    });
    return out;
}

double benchmark_manual_combined_ms(
    const torch::Tensor& x,
    int rounds,
    int warmup_rounds,
    int num_threads
) {
    for (int i = 0; i < warmup_rounds; ++i) {
        manual_combined(x, num_threads);
    }

    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < rounds; ++i) {
        manual_combined(x, num_threads);
    }
    const auto ended = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = ended - started;
    return elapsed.count() / rounds;
}

double benchmark_manual_one_feature_ms(
    const torch::Tensor& x,
    int rounds,
    int warmup_rounds,
    int num_threads,
    const std::string& name
) {
    for (int i = 0; i < warmup_rounds; ++i) {
        manual_one_feature(x, num_threads, name);
    }

    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < rounds; ++i) {
        manual_one_feature(x, num_threads, name);
    }
    const auto ended = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = ended - started;
    return elapsed.count() / rounds;
}

int parse_int_arg(char** argv, int index, int fallback) {
    if (argv[index] == nullptr) {
        return fallback;
    }
    return std::atoi(argv[index]);
}

void print_timing(
    const std::string& group,
    const std::string& path,
    const torch::Tensor& x,
    double median_ms
) {
    std::cout << "TIMING\t" << group << "\t" << path << "\t" << x.size(0) << "\t"
              << x.size(1) << "\t" << x.size(2) << "\t" << median_ms << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0]
                  << " <extractor_kernel.pt2> <mean.pt2> <maximum.pt2> <minimum.pt2> "
                     "[large_batch] [states] [timesteps] [rounds] [threads]\n";
        return 2;
    }

    const std::string extractor_package_path = argv[1];
    const std::string mean_package_path = argv[2];
    const std::string maximum_package_path = argv[3];
    const std::string minimum_package_path = argv[4];
    const int large_batch = argc > 5 ? parse_int_arg(argv, 5, 10000) : 10000;
    const int states = argc > 6 ? parse_int_arg(argv, 6, 2) : 2;
    const int timesteps = argc > 7 ? parse_int_arg(argv, 7, 1000) : 1000;
    const int rounds = argc > 8 ? parse_int_arg(argv, 8, 100) : 100;
    const int detected_threads = static_cast<int>(std::thread::hardware_concurrency());
    const int fallback_threads = detected_threads > 0 ? detected_threads : 1;
    const int threads = argc > 9 ? parse_int_arg(argv, 9, fallback_threads) : fallback_threads;

    c10::InferenceMode inference_mode;
    at::set_num_threads(threads);
    torch::manual_seed(0);

    torch::inductor::AOTIModelPackageLoader extractor_loader(extractor_package_path);
    torch::inductor::AOTIModelPackageLoader mean_loader(mean_package_path);
    torch::inductor::AOTIModelPackageLoader maximum_loader(maximum_package_path);
    torch::inductor::AOTIModelPackageLoader minimum_loader(minimum_package_path);

    const std::vector<torch::Tensor> inputs = {
        torch::randn({1, states, timesteps}, torch::kFloat32),
        torch::randn({large_batch, states, timesteps}, torch::kFloat32),
    };

    std::cout << "THREADS\t" << threads << "\n";
    for (const auto& x : inputs) {
        const int manual_threads = x.size(0) == 1 ? 1 : threads;
        const auto combined_manual = manual_combined(x, manual_threads);
        const auto extractor_out = extractor_loader.run({x})[0];
        std::cout << "DIFF\textractor_aoti_vs_manual_combined_batch_" << x.size(0) << "\t"
                  << (extractor_out - combined_manual).abs().max().item<float>() << "\n";

        print_timing(
            "combined",
            "TorchFeatureExtractor generated kernel, AOTI C++",
            x,
            benchmark_aoti_ms(extractor_loader, x, rounds, 10)
        );
        print_timing(
            "combined",
            x.size(0) == 1 ? "manual C++ one-pass" : "manual C++ one-pass threaded",
            x,
            benchmark_manual_combined_ms(x, rounds, 10, manual_threads)
        );

        const std::vector<std::pair<std::string, torch::inductor::AOTIModelPackageLoader*>>
            loaders = {
                {"mean", &mean_loader},
                {"maximum", &maximum_loader},
                {"minimum", &minimum_loader},
            };
        for (const auto& [name, loader] : loaders) {
            const auto manual = manual_one_feature(x, manual_threads, name);
            const auto aoti_out = loader->run({x})[0];
            std::cout << "DIFF\t" << name << "_aoti_vs_manual_batch_" << x.size(0) << "\t"
                      << (aoti_out - manual).abs().max().item<float>() << "\n";
            print_timing(
                "separated",
                name + " FeatureClass, AOTI C++",
                x,
                benchmark_aoti_ms(*loader, x, rounds, 10)
            );
            print_timing(
                "separated",
                x.size(0) == 1 ? name + " manual C++ loop"
                               : name + " manual C++ loop threaded",
                x,
                benchmark_manual_one_feature_ms(x, rounds, 10, manual_threads, name)
            );
        }
    }
    return 0;
}
