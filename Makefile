# Compiler settings
NVCC = nvcc
NVCC_FLAGS = 

# Directories
SRC_DIR = src
BIN_DIR = bin

# Source files
SOURCES_V0 = $(SRC_DIR)/bitonic_sortV0.cu
SOURCES_V1 = $(SRC_DIR)/bitonic_sortV1.cu
SOURCES_V2 = $(SRC_DIR)/bitonic_sortV2.cu
SOURCES_V3 = $(SRC_DIR)/bitonic_sortV3.cu
SOURCES_RADIX = $(SRC_DIR)/radix_sort.cu

# Executables
EXEC_V0 = $(BIN_DIR)/bitonic_v0
EXEC_V1 = $(BIN_DIR)/bitonic_v1
EXEC_V2 = $(BIN_DIR)/bitonic_v2
EXEC_V3 = $(BIN_DIR)/bitonic_v3
EXEC_RADIX = $(BIN_DIR)/radix_sort

# Default target: prints help
.DEFAULT_GOAL := help

# Help message
help:
	@echo "Usage:"
	@echo "  make bitonic v=[0|1|2|3]              : Build bitonic sort version 0, 1, 2, or 3"
	@echo "  make radix                            : Build radix sort"
	@echo "  make all                              : Build all sorting implementations"
	@echo "  make run v=[0|1|2|3|radix] q=[value]  : Run sorting algorithm (builds if needed)"
	@echo "  make clean                            : Remove all compiled files"

# Build bitonic sort executables into bin folder
$(BIN_DIR)/bitonic_v%: $(SRC_DIR)/bitonic_sortV%.cu
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

bitonic:
	@if [ -z "$(v)" ]; then \
		echo "Usage: make bitonic v=[0|1|2|3]"; exit 1; \
	elif [ "$(v)" = "0" -o "$(v)" = "1" -o "$(v)" = "2" -o "$(v)" = "3" ]; then \
		$(MAKE) $(BIN_DIR)/bitonic_v$(v); \
	else \
		echo "Error: Invalid VERSION (v). Use 0, 1, 2, or 3"; exit 1; \
	fi

# Radix sort target
$(EXEC_RADIX): $(SOURCES_RADIX)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(SOURCES_RADIX) -o $@


radix:
	$(MAKE) $(EXEC_RADIX)

# Build all sorting implementations
all: $(EXEC_V0) $(EXEC_V1) $(EXEC_V2) $(EXEC_V3) $(EXEC_RADIX)

# Run target with checks
run:
	@if [ -z "$(v)" ] || [ -z "$(q)" ]; then \
		echo "Usage: make run v=[0|1|2|3|radix] q=[value]"; exit 1; \
	else \
		case "$(v)" in \
			0|1|2|3) $(MAKE) bitonic_v$(v) && ./bitonic_v$(v) $(q) ;; \
			radix) $(MAKE) $(EXEC_RADIX) && ./$(EXEC_RADIX) $(q) ;; \
			*) echo "Error: Invalid version (v). Use 0, 1, 2, 3, or radix"; exit 1 ;; \
		esac; \
	fi

# Clean
clean:
	rm -rf $(BIN_DIR)
	
# Catch invalid targets
%:
	@echo "Error: Unknown target '$@'"
	@$(MAKE) help

.PHONY: all help bitonic radix clean run