#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

#include <cstddef>

BEGIN_PKG_OP_DEFINITION(PKG_CustomAdd);

template <typename TensorType>
GraphStatus customAddImpl(TensorType& out, const TensorType& lhs,
                          const TensorType& rhs);

// Generic Tensor registration permits the HTP tiler to select compatible
// main-memory or TCM tensor implementations.
DEF_PACKAGE_OP((customAddImpl<Tensor>), "CustomAdd")
DEF_TENSOR_PROPERTIES(Op("CustomAdd", "in0", "in1"), Flat("*"), MainMemory("*"))

template <typename TensorType>
GraphStatus customAddImpl(TensorType& out, const TensorType& lhs,
                          const TensorType& rhs) {
  out.set_dims(lhs);
  auto* output = static_cast<float*>(out.raw_data());
  const auto* left = static_cast<const float*>(lhs.raw_data_const());
  const auto* right = static_cast<const float*>(rhs.raw_data_const());
  // `Tensor` exposes storage counts rather than a STL-style length().  The
  // valid count follows the runtime shape and excludes any padded capacity.
  const std::size_t elements = lhs.valid_storage_elements();

  // Correctness-first implementation. Replace only this loop with an HVX
  // routine; package registration and graph contracts remain unchanged.
  for (std::size_t i = 0; i < elements; ++i) output[i] = left[i] + right[i];
  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_CustomAdd);
