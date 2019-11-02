#ifndef ALLOCATORSTATUS_H
#define ALLOCATORSTATUS_H

#include "AllocatorInfo.h"
#include "SizeClass.h"
#include <list>
#include <map>

using namespace std;

class AllocatorStatus
{
    public:
        AllocatorStatus(AllocatorInfo info);
        void addAllocation(void *address, int size);
        void removeAllocation(void *address);
        int getTotalMemoryReserved();
        int getMaximumMemoryReserved();
        int getMaximumUserMemory();
        void printStatus();

    private:
        int poolSize;
        int granularity;
        int largeGranularity;
        int totalUserMemory;
        int totalLargeMemory;
        int totalLargeMemoryUsed;
        int maximumMemoryReserved;
        int maximumUserMemory;
        list<SizeClass> sizeClasses;
        map<void*,int> allocations;
};

#endif
