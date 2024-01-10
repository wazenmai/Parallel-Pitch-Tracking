NVCCFLAGS  = -O3 -std=c++11 -Xptxas=-v -arch=sm_61
NVCC     = nvcc 
LDFLAGS = -lpng -lz

TARGETS = cuda_sharememory

.PHONY: all
all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS)