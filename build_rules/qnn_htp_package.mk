ifndef QNN_INCLUDE
$(error QNN_INCLUDE must point to QAIRT include/QNN)
endif
ifndef HEXAGON_SDK_ROOT
$(error HEXAGON_SDK_ROOT is required)
endif
ifndef HEXAGON_TOOLS_ROOT
$(error HEXAGON_TOOLS_ROOT is required)
endif
ifndef PACKAGE_NAME
$(error PACKAGE_NAME is required)
endif
ifndef OP_SOURCE
$(error OP_SOURCE is required)
endif
ifndef INTERFACE_SOURCE
$(error INTERFACE_SOURCE is required)
endif

BUILD_DIR ?= build/hexagon-v69
OUTPUT ?= $(BUILD_DIR)/libQnn$(PACKAGE_NAME).so
PROJECT_INCLUDE ?= include
KERNEL_OBJECT ?=

HEXAGON_CXX := $(HEXAGON_TOOLS_ROOT)/bin/hexagon-clang++
QURT_INCLUDE := $(HEXAGON_SDK_ROOT)/rtos/qurt/computev69/include
COMMON_FLAGS := -std=c++17 -fPIC -Wall -Wno-missing-braces \
  -Wno-unused-function -Wno-unused-command-line-argument -Wno-format \
  -fvisibility=default -stdlib=libc++ \
  -DQNN_API="__attribute__((visibility(\"default\")))" \
  -D__QAIC_HEADER_EXPORT="__attribute__((visibility(\"default\")))" \
  -DTHIS_PKG_NAME=$(PACKAGE_NAME)
HEXAGON_FLAGS := $(COMMON_FLAGS) -mv69 -mhvx -mhvx-length=128B -mhmx \
  -DUSE_OS_QURT -DPREPARE_DISABLED -O2 \
  -I$(QNN_INCLUDE) -I$(QURT_INCLUDE)/qurt -I$(QURT_INCLUDE)/posix \
  -I$(HEXAGON_SDK_ROOT)/incs -I$(HEXAGON_SDK_ROOT)/incs/stddef \
  -I$(PROJECT_INCLUDE)

INTERFACE_OBJ := $(BUILD_DIR)/PackageInterface.o
OP_OBJ := $(BUILD_DIR)/Op.o

.PHONY: all
all: $(OUTPUT)

$(BUILD_DIR):
	mkdir -p $@

$(INTERFACE_OBJ): $(INTERFACE_SOURCE) | $(BUILD_DIR)
	$(HEXAGON_CXX) $(HEXAGON_FLAGS) -c $< -o $@

$(OP_OBJ): $(OP_SOURCE) | $(BUILD_DIR)
	$(HEXAGON_CXX) $(HEXAGON_FLAGS) -c $< -o $@

$(OUTPUT): $(INTERFACE_OBJ) $(OP_OBJ) $(KERNEL_OBJECT)
	$(HEXAGON_CXX) -mv69 -fPIC -std=c++17 -g -shared -o $@ $^
