# Defining the variables
finalname = Reform

objects = Parallel_Reform.o GraphicsCardFunctions.o CPUFunctions.o

NVCC = nvcc
NVFLAGS = -arch=sm_30 '-std=c++11'

main: $(objects)
	$(NVCC) $(NVFLAGS) -o $(finalname) $(objects)
	rm $(objects)

GraphicsCardFunctions.o: Source/GraphicsCardFunctions.cu
	$(NVCC) $(NVFLAGS) -c Source/GraphicsCardFunctions.cu

CPUFunctions.o: Source/CPUFunctions.cu
	$(NVCC) $(NVFLAGS) -c Source/CPUFunctions.cu

Parallel_Reform.o: Source/Parallel_Reform.cu
	$(NVCC) $(NVFLAGS) -c Source/Parallel_Reform.cu
