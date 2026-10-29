#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"
#include "hybrid_executor/memref_descriptor.hpp"

#include <cstdint>

BEGIN_PKG_OP_DEFINITION(PKG_VectorAddMlir);
template <typename TensorType>
GraphStatus vectorAddMlirImpl(TensorType& out, const TensorType& lhs, const TensorType& rhs);
DEF_PACKAGE_OP((vectorAddMlirImpl<Tensor>), "VectorAddMlir")
DEF_TENSOR_PROPERTIES(Op("VectorAddMlir", "in0", "in1"), Flat("*"), MainMemory("*"))

#if defined(__hexagon__)
extern "C" void vector_add_mlir(int64_t, hyexec::RankedMemRef<float, 1>*,
                                 int64_t, hyexec::RankedMemRef<float, 1>*,
                                 int64_t, hyexec::RankedMemRef<float, 1>*,
                                 int, int, int, int, int, int);
#endif

template <typename TensorType>
GraphStatus vectorAddMlirImpl(TensorType& out, const TensorType& lhs, const TensorType& rhs) {
  constexpr int64_t kElements = 1024;
  out.set_dims(lhs);
  auto* output = static_cast<float*>(out.raw_data());
  auto* left = const_cast<float*>(static_cast<const float*>(lhs.raw_data_const()));
  auto* right = const_cast<float*>(static_cast<const float*>(rhs.raw_data_const()));
#if defined(__hexagon__)
  const int64_t shape[1] = {kElements};
  auto lhs_desc = hyexec::makeContiguousMemRef(left, shape);
  auto rhs_desc = hyexec::makeContiguousMemRef(right, shape);
  auto out_desc = hyexec::makeContiguousMemRef(output, shape);
  vector_add_mlir(1, &lhs_desc, 1, &rhs_desc, 1, &out_desc, 1, 1, 1, 0, 0, 0);
#else
  for (int64_t i = 0; i < kElements; ++i) output[i] = left[i] + right[i];
#endif
  return GraphStatus::Success;
}
END_PKG_OP_DEFINITION(PKG_VectorAddMlir);
