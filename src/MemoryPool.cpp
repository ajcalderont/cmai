#include "MemoryPool.h"

MemoryPool::MemoryPool(char *baseAddress, int size)
{
    this->baseAddress = baseAddress;
    this->size = size;
    this->memoryUsed = 0;
}

void MemoryPool::addUsed(int allocationSize)
{
    this->memoryUsed += allocationSize;
}

void MemoryPool::freeUsed(int allocationSize)
{
    this->memoryUsed -= allocationSize;
}

int MemoryPool::getMemoryUsed()
{
    return this->memoryUsed;
}

bool MemoryPool::includesAddress(char *address)
{
    return ((address >= this->baseAddress) && (address < (this->baseAddress + this->size)));
}