#include <stdio.h>
#include "AllocatorInfo.h"

int main()
{
    AllocatorInfo info("info.cfg");
    printf("Device name: %s\n", info.deviceName);
    printf("Compute capability: %s\n", info.computeCapability);
	printf("CUDA runtime version: %s\n", info.runtimeVersion);
    printf("CUDA driver version: %s\n\n", info.driverVersion);
    printf("Pool size: %d bytes\n", info.poolSize);
    printf("Granularity: %d bytes\n\n", info.granularity);
    printf("Size classes\n");
    int granularity = info.granularity;
    for(std::list<SizeClassInfo>::iterator i = info.sizeClassesInfo.begin(); i != info.sizeClassesInfo.end(); ++i)
    {
        printf("Class %d: from %-5d to %-5d blocks of %d bytes [%-8d to %-8d bytes]\n", i->classId, i->minBlocks, i->maxBlocks, granularity, i->minBytes, i->maxBytes);
    }
    printf("Larger allocations: mmap rounded to next %d bytes multiple\n\n", info.largerMultiple);
    printf("Allocator policy: %s\n", info.policy);
    printf("Coalescing support: %s\n", info.coalescing?"Yes":"No");
    printf("Splitting support: %s\n", info.splitting?"Yes":"No");
    printf("Expansion policy: %s\n", info.expansionPolicy);
    printf("Pool usage: %s\n", info.poolUsage);
    printf("Shrinking support: %s\n", info.shrinking);
    return 0;
}