#ifndef MEMORYPOOL_H
#define MEMORYPOOL_H

class MemoryPool
{
    public:
        MemoryPool(char *baseAddress, int size);
        void addUsed(int allocationSize);
        void freeUsed(int allocationSize);
        int getMemoryUsed();
        bool includesAddress(char *address);

    private:
        char *baseAddress;
        int size;
        int memoryUsed;
};

#endif
