CXX = g++
NVCC = nvcc
arch ?= sm_50
device ?= 0
program_name ?= my_program
program_arguments ?=
verbosity ?= 0

get_alloc_info:
	-@$(NVCC) -arch=$(arch) -g get_alloc_info.cu -o get_alloc_info
	-@gdb -batch-silent -iex "set auto-load safe-path /" --args get_alloc_info $(device) 2>/dev/null

show_alloc_info: get_alloc_info
	-@$(CXX) AllocatorInfo.cpp show_alloc_info.cpp -o show_alloc_info

preload_library.so:
	-@$(CXX) -I/usr/local/cuda/include -shared -fPIC MemoryPool.cpp SizeClass.cpp AllocatorInfo.cpp AllocatorStatus.cpp preload_library.cpp -o preload_library.so -ldl

memory_analysis: get_alloc_info preload_library.so
	-@$(NVCC) -arch=$(arch) -cudart shared $(program_name).cu -o $(program_name)
	-@VERBOSITY=$(verbosity) LD_PRELOAD="preload_library.so" ./$(program_name) $(program_arguments) 2>$(program_name).txt
	-@cat $(program_name).txt

.PHONY: run clean

run: show_alloc_info
	-@./show_alloc_info

clean:
	-@rm -f get_alloc_info show_alloc_info info.cfg preload_library.so
