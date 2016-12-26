NVCC_FLAGS := -std=c++11 --default-stream per-thread --expt-extended-lambda --expt-relaxed-constexpr
programs = old_async old_thread cpp11async cpp11thread

$(programs): %: %.cu
	nvcc $(NVCC_FLAGS) $^ -o $@

.PHONY: all clean

all: $(programs)

clean:
	rm -f $(programs)
