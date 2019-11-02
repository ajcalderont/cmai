#include "SizeClass.h"

SizeClass::SizeClass(int classId, int minBytes, int maxBytes, int poolSize, int granularity)
{
    this->classId = classId;
    this->minBytes = minBytes;
    this->maxBytes = maxBytes;
    this->poolSize = poolSize;
    this->granularity = granularity;
}

void SizeClass::addAllocation(char *address, int allocationSize)
{
    int realSize = ((allocationSize + this->granularity - 1) / this->granularity) * this->granularity;
    bool newPool = true;
    for(list<MemoryPool>::iterator i = this->memoryPools.begin(); i != this->memoryPools.end(); ++i)
    {
        if(i->includesAddress(address))
        {
            i->addUsed(realSize);
            newPool = false;
            break;
        }
    }

    if(newPool)
    {
        MemoryPool pool(address, this->poolSize);
        pool.addUsed(realSize);
        this->memoryPools.push_front(pool);
    }
}

void SizeClass::removeAllocation(char *address, int allocationSize)
{
    int realSize = ((allocationSize + this->granularity - 1) / this->granularity) * this->granularity;
    for(list<MemoryPool>::iterator i = this->memoryPools.begin(); i != this->memoryPools.end(); ++i)
    {
        if(i->includesAddress(address))
        {
            i->freeUsed(realSize);
            if(i->getMemoryUsed() == 0)
            {
                this->memoryPools.erase(i);
            }
            break;
        }
    }
}

bool SizeClass::includesSize(int allocationSize)
{
    return ((allocationSize >= this->minBytes) && (allocationSize <= this->maxBytes));
}

int SizeClass::getClassId()
{
    return this->classId;
}

int SizeClass::getMemoryReserved()
{
    return (this->memoryPools.size() * this->poolSize);
}

int SizeClass::getMemoryUsed()
{
    int memoryUsed = 0;
    for(list<MemoryPool>::iterator i = this->memoryPools.begin(); i != this->memoryPools.end(); ++i)
    {
        memoryUsed += i->getMemoryUsed();
    }
    return memoryUsed;
}

void SizeClass::setMinBytes(int minBytes)
{
    this->minBytes = minBytes;
}

void SizeClass::setMaxBytes(int maxBytes)
{
    this->maxBytes = maxBytes;
}