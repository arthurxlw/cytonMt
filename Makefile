CUDA=/usr/local/cuda
CXX=$(CUDA)/bin/nvcc
CUFLAGS=--relocatable-device-code=true -arch=sm_30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=sm_62 -gencode=arch=compute_70,code=sm_70 

FLAGS = -Isrc/cytonLib/ -I$(CUDA)/include -O3 -std=c++11 --compile $(CUFLAGS)  -x cu 

LDFLAGS = --cudart static $(CUFLAGS) -link  -L$(CUDA)/lib64 -lcudnn -lcublas -lcurand

print-%  : ; @echo $* = $($*)

SRC = $(wildcard src/*/[a-zA-Z]*.cu)
OBJa = $(SRC:.cu=.o)
OBJ = $(addprefix build/,$(OBJa))

bin/cytonMt: $(OBJ)
	mkdir -p bin
	$(CXX) $(LDFLAGS) $(OBJ)  -o bin/cytonMt 

build/%.o: %.cu
	@mkdir -p $(@D)
	$(CXX) -c $(FLAGS) $< -o $@ 

clean:
	rm -rf build bin
	rm -rf data/model
