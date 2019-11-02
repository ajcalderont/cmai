#ifndef SIZECLASS_H
#define SIZECLASS_H

#include <list>
#include "MemoryPool.h"

using namespace std;

class SizeClass
{
    public:
        SizeClass(int classId, int minBytes, int maxBytes, int poolSize, int granularity);
        void addAllocation(char *address, int allocationSize);
        void removeAllocation(char *address, int allocationSize);
        bool includesSize(int allocationSize);
        int getClassId();
        int getMemoryReserved();
        int getMemoryUsed();
        void setMinBytes(int minBytes);
        void setMaxBytes(int maxBytes);

    private:
        int classId;
        int minBytes;
        int maxBytes;
        int poolSize;
        int granularity;
        list<MemoryPool> memoryPools;
};

#endif
