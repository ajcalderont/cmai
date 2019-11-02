#include <stdio.h>
#include "AllocatorStatus.h"

AllocatorStatus::AllocatorStatus(AllocatorInfo info)
{
    this->poolSize = info.poolSize;
    this->granularity = info.granularity;
    this->largeGranularity = info.largerMultiple;
    this->totalUserMemory = 0;
    this->totalLargeMemory = 0;
    this->totalLargeMemoryUsed = 0;
    this->maximumMemoryReserved = 0;
    for(list<SizeClassInfo>::iterator i = info.sizeClassesInfo.begin(); i != info.sizeClassesInfo.end(); ++i)
    {
        SizeClass sizeClass(i->classId, i->minBytes, i->maxBytes, this->poolSize, this->granularity);
        this->sizeClasses.push_back(sizeClass);
    }
    this->sizeClasses.back().setMaxBytes(this->poolSize);
}

void AllocatorStatus::addAllocation(void *address, int size)
{
    this->allocations.insert(pair<void*,int>(address, size));
    this->totalUserMemory += size;
    if(size > this->poolSize)
    {
        int realSize = ((size + this->largeGranularity - 1) / this->largeGranularity) * this->largeGranularity;
        this->totalLargeMemory += realSize;
        this->totalLargeMemoryUsed += size;
    }
    else
    {
         for(list<SizeClass>::iterator i = this->sizeClasses.begin(); i != this->sizeClasses.end(); ++i)
         {
             if(i->includesSize(size))
             {
                 i->addAllocation(static_cast<char*>(address), size);
                 break;
             }
         }
    }
    int currentTotal = this->getTotalMemoryReserved();
    if(currentTotal >= this->maximumMemoryReserved)
    {
        this->maximumMemoryReserved = currentTotal;
        this->maximumUserMemory = this->totalUserMemory;
    }
}

void AllocatorStatus::removeAllocation(void *address)
{
    int size = this->allocations[address];
    this->totalUserMemory -= size;
    if(size > this->poolSize)
    {
        int realSize = ((size + this->largeGranularity - 1) / this->largeGranularity) * this->largeGranularity;
        this->totalLargeMemory -= realSize;
        this->totalLargeMemoryUsed -= size;
    }
    else
    {
        for(list<SizeClass>::iterator i = this->sizeClasses.begin(); i != this->sizeClasses.end(); ++i)
         {
             if(i->includesSize(size))
             {
                 i->removeAllocation(static_cast<char*>(address), size);
                 break;
             }
         }
    }
    this->allocations.erase(address);
}

int AllocatorStatus::getTotalMemoryReserved()
{
    int totalMemoryReserved = this->totalLargeMemory;
    for(list<SizeClass>::iterator i = this->sizeClasses.begin(); i != this->sizeClasses.end(); ++i)
    {
        totalMemoryReserved += i->getMemoryReserved();
    }
    return totalMemoryReserved;
}

int AllocatorStatus::getMaximumMemoryReserved()
{
    return this->maximumMemoryReserved;
}

int AllocatorStatus::getMaximumUserMemory()
{
    return this->maximumUserMemory;
}

void AllocatorStatus::printStatus()
{
    fprintf(stderr, "Total memory reserved by user: %d bytes\n", this->totalUserMemory);
    fprintf(stderr, "Real memory reserved by allocator:\n");
    int totalMemoryUsed = this->totalLargeMemoryUsed;
    for(list<SizeClass>::iterator i = this->sizeClasses.begin(); i != this->sizeClasses.end(); ++i)
    {
        int memoryUsed = i->getMemoryUsed();
        fprintf(stderr, "Class %d: %d bytes (%d bytes used)\n", i->getClassId(), i->getMemoryReserved(), memoryUsed);
        totalMemoryUsed += memoryUsed;
    }
    int totalMemoryReserved = this->getTotalMemoryReserved();
    float percentageUsed = totalMemoryReserved > 0? (totalMemoryUsed/(float)totalMemoryReserved) * 100 : 0;
    float factor = this->totalUserMemory > 0? (totalMemoryReserved / (float)this->totalUserMemory) : 0;
    fprintf(stderr, "Larger allocations: %d bytes\n", this->totalLargeMemory);
    fprintf(stderr, "Total memory reserved by allocator: %d bytes (%.1fx the amount requested by user)\n", totalMemoryReserved, factor);
    fprintf(stderr, "Total memory used : %d bytes (%.1f%% of the memory reserved by allocator)\n\n", totalMemoryUsed, percentageUsed);
}