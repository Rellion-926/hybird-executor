PYTHON ?= python3
TARGET ?= v69
KERNEL_MODE ?= prebuilt
PLUGIN ?= vector_add_mlir

.PHONY: test list doctor build bundle deploy run audit

test:
	$(PYTHON) -m unittest discover -s tests -v

list:
	./hyexec list

doctor:
	./hyexec doctor --kernel-mode $(KERNEL_MODE)

build:
	./hyexec build $(PLUGIN) --target $(TARGET) --kernel-mode $(KERNEL_MODE)

bundle:
	./hyexec bundle $(PLUGIN) --target $(TARGET)

deploy:
	./hyexec deploy $(PLUGIN) --target $(TARGET)

run:
	./hyexec run $(PLUGIN) --target $(TARGET)

audit:
	$(PYTHON) tools/audit_repository.py
