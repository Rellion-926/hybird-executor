#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace hyexec {

template <typename T, std::size_t Rank>
struct RankedMemRef {
  T* allocated;
  T* aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

template <typename T, std::size_t Rank>
RankedMemRef<T, Rank> makeContiguousMemRef(T* data,
                                           const int64_t (&sizes)[Rank]) {
  RankedMemRef<T, Rank> result{};
  result.allocated = data;
  result.aligned = data;
  result.offset = 0;
  int64_t stride = 1;
  for (std::size_t i = Rank; i-- > 0;) {
    result.sizes[i] = sizes[i];
    result.strides[i] = stride;
    stride *= sizes[i];
  }
  return result;
}

static_assert(std::is_standard_layout<RankedMemRef<float, 1>>::value,
              "MLIR memref descriptor must be standard-layout");

}  // namespace hyexec
