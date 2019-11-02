#ifndef ALLOCATORINFO_H
#define ALLOCATORINFO_H

#include <list>

using namespace std;

class SizeClassInfo
{
    public:
        SizeClassInfo(int classId, int minBlocks, int maxBlocks, int minBytes, int maxBytes);
        int classId;
        int minBlocks;
        int maxBlocks;
        int minBytes;
        int maxBytes;
};

class AllocatorInfo
{
    public:
        AllocatorInfo(const char* fileName);
        char deviceName[20];
        char computeCapability[10];
        char runtimeVersion[10];
        char driverVersion[10];
        char policy[20];
        char expansionPolicy[20];
        char poolUsage[20];
        char shrinking[30];
        int poolSize;
        int granularity;
        int largerMultiple;
        int coalescing;
        int splitting;
        list<SizeClassInfo> sizeClassesInfo;
};

#endif