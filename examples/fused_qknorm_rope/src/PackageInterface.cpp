#include <array>
#include <cstring>

#include "HTP/QnnHtpCommon.h"
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "HTP/core/unique_types.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"

DEFINE_UNIQ_TY()
BEGIN_PKG_OPS_OPTS_LIST()
DECLARE_PKG_OPS_OPTS_LIST(PKG_FusedQKNormRoPE)
END_PKG_OPS_OPTS_LIST()

static constexpr auto sg_packageName = THIS_PKG_NAME_STR;
static std::array<const char *, 1> sg_opNames{{"FusedQKNormRoPE"}};
static Qnn_ApiVersion_t sg_sdkApiVersion = QNN_HTP_API_VERSION_INIT;
static QnnOpPackage_Info_t sg_packageInfo = QNN_OP_PACKAGE_INFO_INIT;
static bool sg_initialized = false;

INIT_PACKAGE_OP_DEF()
INIT_PACKAGE_OPTIMIZATION_DEF()
INIT_PACKAGE_PARAM_ORDER_DEF()
INIT_PKG_CORE_INIT_FUNC()

Qnn_ErrorHandle_t packageInit(QnnOpPackage_GlobalInfrastructure_t) {
  if (sg_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  REGISTER_PACKAGE_PARAM_ORDERS()
  REGISTER_PACKAGE_AXIS_PARAMS()
  REGISTER_PACKAGE_PER_CHANNEL_QUANTIZED_OPS()
  sg_initialized = true;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t packageGetInfo(const QnnOpPackage_Info_t **info) {
  if (!sg_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  if (info == nullptr) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  sg_packageInfo = QNN_OP_PACKAGE_INFO_INIT;
  sg_packageInfo.packageName = sg_packageName;
  sg_packageInfo.operationNames = sg_opNames.data();
  sg_packageInfo.numOperations = sg_opNames.size();
  sg_packageInfo.sdkBuildId = QNN_SDK_BUILD_ID;
  sg_packageInfo.sdkApiVersion = &sg_sdkApiVersion;
  *info = &sg_packageInfo;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t packageValidate(Qnn_OpConfig_t op) {
  if (op.v1.packageName == nullptr || op.v1.typeName == nullptr ||
      std::strcmp(op.v1.packageName, sg_packageName) != 0 ||
      std::strcmp(op.v1.typeName, "FusedQKNormRoPE") != 0 ||
      op.v1.numOfParams != 0 || op.v1.numOfInputs != 6 ||
      op.v1.numOfOutputs != 1) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t packageTerminate() {
  if (!sg_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  sg_initialized = false;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t logInitialize(QnnLog_Callback_t callback, QnnLog_Level_t level) {
  return callback != nullptr && level >= QNN_LOG_LEVEL_ERROR
             ? QNN_SUCCESS
             : QNN_LOG_ERROR_INVALID_ARGUMENT;
}
Qnn_ErrorHandle_t logSetLevel(QnnLog_Level_t level) {
  return level >= QNN_LOG_LEVEL_ERROR ? QNN_SUCCESS
                                      : QNN_LOG_ERROR_INVALID_ARGUMENT;
}
Qnn_ErrorHandle_t logTerminate() { return QNN_SUCCESS; }

extern "C" Qnn_ErrorHandle_t FusedMlirPackageInterfaceProvider(
    QnnOpPackage_Interface_t *interface) {
  if (interface == nullptr) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  interface->interfaceVersion = {1, 4, 0};
  interface->v1_4.init = packageInit;
  interface->v1_4.terminate = packageTerminate;
  interface->v1_4.getInfo = packageGetInfo;
  interface->v1_4.validateOpConfig = packageValidate;
  interface->v1_4.createOpImpl = nullptr;
  interface->v1_4.freeOpImpl = nullptr;
  interface->v1_4.logInitialize = logInitialize;
  interface->v1_4.logSetLevel = logSetLevel;
  interface->v1_4.logTerminate = logTerminate;
  return QNN_SUCCESS;
}
