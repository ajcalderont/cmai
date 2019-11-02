# CUDA Memory Allocator Inspector

## Introduction

CUDA Memory Allocator Inspector is a tool that can be executed in a Linux system with an NVIDIA GPU to extract its memory allocator properties. It can be used in two ways: In the first one it uses reverse engineering techniques to extract the memory allocator properties of a target GPU. With this information it generates a configuration file which can be used to visualize the extracted properties of the memory allocator. The second way of using this tool consists on a preload library which can be used to automatically compute the real GPU memory consumption of a target application based on the memory allocator properties stored in the configuration file.

## How to use the tool

The easiest way to use the tool is through the commands defined in the Makefile.

### Extracting the memory allocator properties

To extract the memory allocator properties and show the results, run the command:

```
make run
```

If you have more than one GPU in the same computer, you can specify the one you are interested with the parameter *device*:

```
make run device=1
```

### Analyzing the real memory usage of a target CUDA application

To compute the real memory usage of a target CUDA application, based on the extracted properties of the memory allocator, you can use the command:

```
make memory_analysis program_name=my_program
```
The tool will show the maximum memory usage of the target application and will also generate a txt file with the results (for example, my_program.txt). Note that you must specify the program name without the .cu extension. If your program receives arguments, you can specify them using the *program_arguments* parameter:

```
make memory_analysis program_name=my_program program_arguments=my_args
```
By default, the tool only shows the maximum memory usage of the target application. If you want to see the evolution of the memory usage after every CUDA memory call in your program, you should use the *verbosity* parameter:

```
make memory_analysis program_name=my_program verbosity=1
```
