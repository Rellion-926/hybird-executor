ifndef QAIRT_SDK_ROOT
$(error QAIRT_SDK_ROOT is required)
endif
ifndef ANDROID_NDK_ROOT
$(error ANDROID_NDK_ROOT is required)
endif
ifndef MODEL_CPP
$(error MODEL_CPP is required)
endif
ifndef MODEL_BIN
MODEL_BIN := $(basename $(MODEL_CPP)).bin
endif
ifndef MODEL_NAME
$(error MODEL_NAME is required)
endif

OUTPUT_DIR ?= build/model

.PHONY: all
all:
	$(QAIRT_SDK_ROOT)/bin/x86_64-linux-clang/qnn-model-lib-generator \
	  -c $(MODEL_CPP) -b $(MODEL_BIN) -o $(OUTPUT_DIR) -t aarch64-android \
	  --model_name $(MODEL_NAME) --backend libQnnHtp.so \
	  --ndk_root $(ANDROID_NDK_ROOT)
