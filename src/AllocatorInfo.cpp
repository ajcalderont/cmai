#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "AllocatorInfo.h"

SizeClassInfo::SizeClassInfo(int classId, int minBlocks, int maxBlocks, int minBytes, int maxBytes)
{
	this->classId = classId;
    this->minBlocks = minBlocks;
    this->maxBlocks = maxBlocks;
    this->minBytes = minBytes;
    this->maxBytes = maxBytes;
}

AllocatorInfo::AllocatorInfo(const char* fileName)
{
	FILE *configFile = fopen(fileName, "r");
	char line[50], param[5];
	
	while(fgets(line, sizeof line, configFile) != NULL)
	{
		sscanf(line, "%3s", param);
		if(strcmp(param, "DEV") == 0)
		{
			sscanf(line, "%3s %20[^\n]", param, this->deviceName);
		} 
		else if(strcmp(param, "CAP") == 0)
		{
			sscanf(line, "%3s %10s", param, this->computeCapability);
		}
		else if(strcmp(param, "RTV") == 0)
		{
			sscanf(line, "%3s %10s", param, this->runtimeVersion);
		}
		else if(strcmp(param, "DRV") == 0)
		{
			sscanf(line, "%3s %10s", param, this->driverVersion);
		}
		else if(strcmp(param, "PSZ") == 0)
		{
			sscanf(line, "%3s %d", param, &this->poolSize);
		}
		else if(strcmp(param, "GRN") == 0)
		{
			sscanf(line, "%3s %d", param, &this->granularity);
		}
		else if(strcmp(param, "SCL") == 0)
		{
			int classId, minBlocks, maxBlocks, minBytes, maxBytes;
			sscanf(line, "%3s %d %d %d %d %d", param, &classId, &minBlocks, &maxBlocks, &minBytes, &maxBytes);
			SizeClassInfo sizeClassInfo(classId, minBlocks, maxBlocks, minBytes, maxBytes);
			this->sizeClassesInfo.push_back(sizeClassInfo);
		}
		else if(strcmp(param, "LGM") == 0)
		{
			sscanf(line, "%3s %d", param, &this->largerMultiple);
		}
		else if(strcmp(param, "POL") == 0)
		{
			sscanf(line, "%3s %20[^\n]", param, this->policy);
		}
		else if(strcmp(param, "CLS") == 0)
		{
			sscanf(line, "%3s %d", param, &this->coalescing);
		}
		else if(strcmp(param, "SPL") == 0)
		{
			sscanf(line, "%3s %d", param, &this->splitting);
		}
		else if(strcmp(param, "EXP") == 0)
		{
			sscanf(line, "%3s %20[^\n]", param, this->expansionPolicy);
		}
		else if(strcmp(param, "USG") == 0)
		{
			sscanf(line, "%3s %20[^\n]", param, this->poolUsage);
		}
		else if(strcmp(param, "SHR") == 0)
		{
			sscanf(line, "%3s %30[^\n]", param, this->shrinking);
		}
	}

	fclose(configFile);
}