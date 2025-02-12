# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -O3

# Source files
SOURCES_V0 = bitonic_sortV0.cu
SOURCES_V1 = bitonic_sortV1.cu
SOURCES_V2 = bitonic_sortV2.cu
SOURCES_RADIX = radix.cu

# Executables
EXEC_V0 = bitonic_v0
EXEC_V1 = bitonic_v1
EXEC_V2 = bitonic_v2
EXEC_RADIX = radix_sort

# Default target: prints help
.DEFAULT_GOAL := help

# Help message
help:
	@echo "Usage:"
	@echo "  make bitonic v=[0|1|2]        : Build bitonic sort version (v) 0, 1, or 2"
	@echo "  make radix                    : Build radix sort"
	@echo "  make all                      : Build all sorting implementations"
	@echo "  make run v=[0|1|2|radix] q=[value]  : Run sorting algorithm (builds if needed)"
	@echo "  make clean                    : Remove all compiled files"

# Build bitonic sort executables
bitonic_v%: bitonic_sortV%.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

bitonic:
	@if [ -z "$(v)" ]; then \
		echo "Usage: make bitonic v=[0|1|2]"; exit 1; \
	elif [ "$(v)" = "0" -o "$(v)" = "1" -o "$(v)" = "2" ]; then \
		$(MAKE) bitonic_v$(v); \
	else \
		echo "Error: Invalid VERSION (v). Use 0, 1, or 2"; exit 1; \
	fi

# Radix sort target
$(EXEC_RADIX): $(SOURCES_RADIX)
	$(NVCC) $(NVCC_FLAGS) $(SOURCES_RADIX) -o $@

radix:
	$(MAKE) $(EXEC_RADIX)

# Build all sorting implementations
all: $(EXEC_V0) $(EXEC_V1) $(EXEC_V2) $(EXEC_RADIX)

# Run target with checks
run:
	@if [ -z "$(v)" ] || [ -z "$(q)" ]; then \
		echo "Usage: make run v=[0|1|2|radix] q=[value]"; exit 1; \
	else \
		case "$(v)" in \
			0|1|2) $(MAKE) bitonic_v$(v) && ./bitonic_v$(v) $(q) ;; \
			radix) $(MAKE) $(EXEC_RADIX) && ./$(EXEC_RADIX) $(q) ;; \
			*) echo "Error: Invalid version (v). Use 0, 1, 2, or radix"; exit 1 ;; \
		esac; \
	fi

# Clean
clean:
	rm -f $(EXEC_V0) $(EXEC_V1) $(EXEC_V2) $(EXEC_RADIX)

# Catch invalid targets
%:
	@echo "Error: Unknown target '$@'"
	@$(MAKE) help

.PHONY: all help bitonic radix clean run