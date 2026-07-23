#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/program.h>

namespace runtime = executorch::runtime;
namespace extension = executorch::extension;

class VectorDataLoader final : public runtime::DataLoader {
 public:
  explicit VectorDataLoader(std::vector<uint8_t> data) : data_(std::move(data)) {}

  runtime::Result<runtime::FreeableBuffer> load(
      size_t offset,
      size_t size,
      const SegmentInfo&) const override {
    if (offset > data_.size() || size > data_.size() - offset) {
      return runtime::Error::InvalidArgument;
    }
    auto* copy = static_cast<uint8_t*>(std::malloc(size));
    if (copy == nullptr) {
      return runtime::Error::MemoryAllocationFailed;
    }
    std::memcpy(copy, data_.data() + offset, size);
    return runtime::FreeableBuffer(
        copy,
        size,
        [](void*, void* data, size_t) { std::free(data); });
  }

  runtime::Error load_into(
      size_t offset,
      size_t size,
      const SegmentInfo&,
      void* buffer) const override {
    if (offset > data_.size() || size > data_.size() - offset) {
      return runtime::Error::InvalidArgument;
    }
    std::memcpy(buffer, data_.data() + offset, size);
    return runtime::Error::Ok;
  }

  runtime::Result<size_t> size() const override {
    return data_.size();
  }

 private:
  std::vector<uint8_t> data_;
};

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    std::cerr << "failed to open " << path << "\n";
    std::exit(2);
  }
  return std::vector<uint8_t>(
      std::istreambuf_iterator<char>(stream),
      std::istreambuf_iterator<char>());
}

std::vector<float> random_input(int batch, int states, int timesteps) {
  std::mt19937 rng(12345);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> values(static_cast<size_t>(batch) * states * timesteps);
  for (float& value : values) {
    value = dist(rng);
  }
  return values;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  if (values.size() % 2 == 0) {
    return (values[middle - 1] + values[middle]) / 2.0;
  }
  return values[middle];
}

runtime::Method load_method(
    const runtime::Program& program,
    runtime::MemoryManager& memory_manager) {
  auto method_result = program.load_method("forward", &memory_manager);
  if (!method_result.ok()) {
    std::cerr << "load_method failed: " << static_cast<int>(method_result.error()) << "\n";
    std::exit(3);
  }
  return std::move(method_result.get());
}

void run_batch(
    const runtime::Program& program,
    int batch,
    int states,
    int timesteps,
    int rounds,
    int warmup_rounds) {
  constexpr size_t allocator_size = 64 * 1024 * 1024;
  std::vector<uint8_t> method_memory(allocator_size);
  std::vector<uint8_t> temp_memory(allocator_size);
  runtime::MemoryAllocator method_allocator(
      method_memory.size(),
      method_memory.data());
  runtime::MemoryAllocator temp_allocator(temp_memory.size(), temp_memory.data());

  auto meta_result = program.method_meta("forward");
  if (!meta_result.ok()) {
    std::cerr << "method_meta failed: " << static_cast<int>(meta_result.error()) << "\n";
    std::exit(4);
  }
  const runtime::MethodMeta& meta = meta_result.get();
  std::vector<std::vector<uint8_t>> planned_storage(meta.num_memory_planned_buffers());
  std::vector<runtime::Span<uint8_t>> planned_spans;
  planned_spans.reserve(planned_storage.size());
  for (size_t index = 0; index < planned_storage.size(); ++index) {
    const auto size_result = meta.memory_planned_buffer_size(index);
    if (!size_result.ok()) {
      std::cerr << "memory_planned_buffer_size failed: "
                << static_cast<int>(size_result.error()) << "\n";
      std::exit(5);
    }
    planned_storage[index].resize(size_result.get());
    planned_spans.emplace_back(planned_storage[index].data(), planned_storage[index].size());
  }
  runtime::HierarchicalAllocator planned_memory(
      runtime::Span<runtime::Span<uint8_t>>(planned_spans.data(), planned_spans.size()));
  runtime::MemoryManager memory_manager(
      &method_allocator,
      planned_spans.empty() ? nullptr : &planned_memory,
      &temp_allocator);
  runtime::Method method = load_method(program, memory_manager);

  std::vector<float> input = random_input(batch, states, timesteps);
  extension::TensorPtr input_tensor = extension::from_blob(
      input.data(),
      {
          static_cast<executorch::aten::SizesType>(batch),
          static_cast<executorch::aten::SizesType>(states),
          static_cast<executorch::aten::SizesType>(timesteps),
      },
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND);
  runtime::EValue input_value(*input_tensor);

  auto execute_once = [&]() {
    runtime::Error error = method.set_input(input_value, 0);
    if (error != runtime::Error::Ok) {
      std::cerr << "set_input failed: " << static_cast<int>(error) << "\n";
      std::exit(6);
    }
    error = method.execute();
    if (error != runtime::Error::Ok) {
      std::cerr << "execute failed: " << static_cast<int>(error) << "\n";
      std::exit(7);
    }
    runtime::EValue output;
    error = method.get_outputs(&output, 1);
    if (error != runtime::Error::Ok || !output.isTensor()) {
      std::cerr << "get_outputs failed: " << static_cast<int>(error) << "\n";
      std::exit(8);
    }
    volatile float sink = output.toTensor().const_data_ptr<float>()[0];
    (void)sink;
  };

  for (int i = 0; i < warmup_rounds; ++i) {
    execute_once();
  }

  std::vector<double> timings;
  timings.reserve(rounds);
  for (int i = 0; i < rounds; ++i) {
    const auto started = std::chrono::steady_clock::now();
    execute_once();
    const auto stopped = std::chrono::steady_clock::now();
    timings.push_back(
        std::chrono::duration<double, std::milli>(stopped - started).count());
  }

  std::cout << "TIMING\texecutorch_cpp\tTorchFeatureExtractor generated kernel, "
               "ExecuTorch C++ runtime\t"
            << batch << "\t" << states << "\t" << timesteps << "\t" << median(timings)
            << "\n";
}

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage: " << argv[0]
              << " model.pte large_batch states timesteps rounds warmup_rounds\n";
    return 1;
  }

  const std::string model_path = argv[1];
  const int large_batch = std::stoi(argv[2]);
  const int states = std::stoi(argv[3]);
  const int timesteps = std::stoi(argv[4]);
  const int rounds = std::stoi(argv[5]);
  const int warmup_rounds = std::stoi(argv[6]);

  VectorDataLoader loader(read_file(model_path));
  auto program_result = runtime::Program::load(&loader, runtime::Program::Verification::Minimal);
  if (!program_result.ok()) {
    std::cerr << "Program::load failed: " << static_cast<int>(program_result.error()) << "\n";
    return 2;
  }
  runtime::Program program = std::move(program_result.get());

  run_batch(program, 1, states, timesteps, rounds, warmup_rounds);
  run_batch(program, large_batch, states, timesteps, rounds, warmup_rounds);
  return 0;
}
