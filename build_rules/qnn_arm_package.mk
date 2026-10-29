ifndef QNN_INCLUDE
$(error QNN_INCLUDE must point to QAIRT include/QNN)
endif
ifndef QNN_TARGET_LIB
$(error QNN_TARGET_LIB must point to QAIRT lib/aarch64-android)
endif
ifndef HEXAGON_SDK_ROOT
$(error HEXAGON_SDK_ROOT is required for libnative headers)
endif
ifndef HEXAGON_TOOLS_ROOT
$(error HEXAGON_TOOLS_ROOT is required for libnative headers)
endif
ifndef ANDROID_NDK_ROOT
$(error ANDROID_NDK_ROOT is required)
endif
ifndef PACKAGE_NAME
$(error PACKAGE_NAME is required)
endif

BUILD_DIR ?= build/aarch64-android
OUTPUT ?= $(BUILD_DIR)/libQnn$(PACKAGE_NAME).so
INTERFACE_SOURCE ?= src/PackageInterface.cpp
OP_SOURCE ?= src/ops/Op.cpp
PROJECT_INCLUDE ?= include
NDK_HOST_TAG ?= linux-x86_64

ANDROID_CXX := $(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/$(NDK_HOST_TAG)/bin/clang++
SYSROOT := $(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/$(NDK_HOST_TAG)/sysroot
LIBNATIVE := $(HEXAGON_TOOLS_ROOT)/libnative
COMMON_FLAGS := --target=aarch64-none-linux-android21 --sysroot=$(SYSROOT) \
  -std=c++17 -fPIC -Wall -Wno-missing-braces -Wno-unused-function \
  -Wno-unused-command-line-argument -Wno-format -Wno-unused-variable \
  -Wno-unused-parameter -fvisibility=default -stdlib=libc++ \
  -DANDROID -DUSE_OS_LINUX -DREFERENCE_OP \
  -DQNN_API="__attribute__((visibility(\"default\")))" \
  -D__QAIC_HEADER_EXPORT="__attribute__((visibility(\"default\")))" \
  -DTHIS_PKG_NAME=$(PACKAGE_NAME) \
  -I$(QNN_INCLUDE) -I$(LIBNATIVE)/include -I$(PROJECT_INCLUDE)

INTERFACE_OBJ := $(BUILD_DIR)/PackageInterface.o
OP_OBJ := $(BUILD_DIR)/Op.o

.PHONY: all
all: $(OUTPUT)

$(BUILD_DIR):
	mkdir -p $@

$(INTERFACE_OBJ): $(INTERFACE_SOURCE) | $(BUILD_DIR)
	$(ANDROID_CXX) $(COMMON_FLAGS) -c $< -o $@

$(OP_OBJ): $(OP_SOURCE) | $(BUILD_DIR)
	$(ANDROID_CXX) $(COMMON_FLAGS) -c $< -o $@

$(OUTPUT): $(INTERFACE_OBJ) $(OP_OBJ)
	$(ANDROID_CXX) --target=aarch64-none-linux-android21 --sysroot=$(SYSROOT) \
	  -stdlib=libc++ -static-libstdc++ -shared -o $@ $^ \
	  -L$(QNN_TARGET_LIB) -lQnnHtp -lQnnHtpPrepare
