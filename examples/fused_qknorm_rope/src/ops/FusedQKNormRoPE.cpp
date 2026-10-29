#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

#include <cstddef>
#include <cstdint>

BEGIN_PKG_OP_DEFINITION(PKG_FusedQKNormRoPE);

template <typename TensorType>
GraphStatus fusedQKNormRoPEImpl(TensorType &combinedOut,
                                const TensorType &q,
                                const TensorType &k,
                                const TensorType &gammaQ,
                                const TensorType &gammaK,
                                const TensorType &sin,
                                const TensorType &cos);

DEF_PACKAGE_OP((fusedQKNormRoPEImpl<Tensor>), "FusedQKNormRoPE")

namespace {

constexpr size_t kQHeads = 32;
constexpr size_t kKHeads = 8;
constexpr size_t kTotalHeads = kQHeads + kKHeads;
constexpr size_t kHeadDim = 128;
constexpr size_t kRotaryHalf = 64;

template <typename T, int Rank>
struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

#if defined(__hexagon__)
extern "C" void fused_qknorm_rope(
    int64_t, MemRefDescriptor<_Float16, 4> *,
    int64_t, MemRefDescriptor<_Float16, 4> *,
    int64_t, MemRefDescriptor<_Float16, 1> *,
    int64_t, MemRefDescriptor<_Float16, 1> *,
    int64_t, MemRefDescriptor<float, 2> *,
    int64_t, MemRefDescriptor<float, 2> *,
    int64_t, MemRefDescriptor<_Float16, 4> *,
    int, int, int, int, int, int);
#endif

}  // namespace

template <typename TensorType>
GraphStatus fusedQKNormRoPEImpl(TensorType &combinedOut,
                                const TensorType &q,
                                const TensorType &k,
                                const TensorType &gammaQ,
                                const TensorType &gammaK,
                                const TensorType &sin,
                                const TensorType &cos) {
  auto [batch, sequence, qHeads, headDim] = q.dims();
  size_t outDims[4] = {batch, sequence, kTotalHeads, headDim};
  combinedOut.set_dims(outDims);

#if defined(__hexagon__)
  auto *qData = const_cast<_Float16 *>(
      static_cast<const _Float16 *>(q.raw_data_const()));
  auto *kData = const_cast<_Float16 *>(
      static_cast<const _Float16 *>(k.raw_data_const()));
  auto *gammaQData = const_cast<_Float16 *>(
      static_cast<const _Float16 *>(gammaQ.raw_data_const()));
  auto *gammaKData = const_cast<_Float16 *>(
      static_cast<const _Float16 *>(gammaK.raw_data_const()));
  auto *sinData = const_cast<float *>(
      static_cast<const float *>(sin.raw_data_const()));
  auto *cosData = const_cast<float *>(
      static_cast<const float *>(cos.raw_data_const()));
  auto *outData = static_cast<_Float16 *>(combinedOut.raw_data());

  MemRefDescriptor<_Float16, 4> qDesc{
      qData, qData, 0,
      {static_cast<int64_t>(batch), static_cast<int64_t>(sequence),
       static_cast<int64_t>(qHeads), static_cast<int64_t>(headDim)},
      {static_cast<int64_t>(sequence * qHeads * headDim),
       static_cast<int64_t>(qHeads * headDim), static_cast<int64_t>(headDim), 1}};
  MemRefDescriptor<_Float16, 4> kDesc{
      kData, kData, 0,
      {static_cast<int64_t>(batch), static_cast<int64_t>(sequence),
       static_cast<int64_t>(kKHeads), static_cast<int64_t>(headDim)},
      {static_cast<int64_t>(sequence * kKHeads * headDim),
       static_cast<int64_t>(kKHeads * headDim), static_cast<int64_t>(headDim), 1}};
  MemRefDescriptor<_Float16, 1> gammaQDesc{
      gammaQData, gammaQData, 0, {static_cast<int64_t>(kHeadDim)}, {1}};
  MemRefDescriptor<_Float16, 1> gammaKDesc{
      gammaKData, gammaKData, 0, {static_cast<int64_t>(kHeadDim)}, {1}};
  MemRefDescriptor<float, 2> sinDesc{
      sinData, sinData, 0,
      {static_cast<int64_t>(sequence), static_cast<int64_t>(kRotaryHalf)},
      {static_cast<int64_t>(kRotaryHalf), 1}};
  MemRefDescriptor<float, 2> cosDesc{
      cosData, cosData, 0,
      {static_cast<int64_t>(sequence), static_cast<int64_t>(kRotaryHalf)},
      {static_cast<int64_t>(kRotaryHalf), 1}};
  MemRefDescriptor<_Float16, 4> outDesc{
      outData, outData, 0,
      {static_cast<int64_t>(batch), static_cast<int64_t>(sequence),
       static_cast<int64_t>(kTotalHeads), static_cast<int64_t>(headDim)},
      {static_cast<int64_t>(sequence * kTotalHeads * headDim),
       static_cast<int64_t>(kTotalHeads * headDim), static_cast<int64_t>(headDim), 1}};

  for (int token = 0; token < static_cast<int>(sequence); ++token) {
    for (int head = 0; head < static_cast<int>(kTotalHeads); ++head) {
      fused_qknorm_rope(4, &qDesc, 4, &kDesc, 1, &gammaQDesc, 1, &gammaKDesc,
                        2, &sinDesc, 2, &cosDesc, 4, &outDesc,
                        static_cast<int>(sequence), static_cast<int>(kTotalHeads),
                        1, token, head, 0);
    }
  }
#else
  (void)k;
  (void)gammaQ;
  (void)gammaK;
  (void)sin;
  (void)cos;
#endif
  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_FusedQKNormRoPE);
