set logging file info.cfg

# Device properties and pool size extraction
break get_alloc_info.cu:main:break_01
run
break mmap
continue
set logging overwrite on
set logging on
printf "DEV %s\n", main::properties.name
printf "CAP %d.%d\n", main::properties.major, main::properties.minor
printf "RTV %d.%d\n", main::runtime_version/1000, (main::runtime_version%1000)/10
printf "DRV %d.%d\n", main::driver_version/1000, (main::driver_version%1000)/10
printf "PSZ %d\n", len
set logging off
set logging overwrite off
set variable main::pool_size = len
delete

# Granularity extraction
break get_alloc_info.cu:main:break_02
continue
break mmap
continue
set logging on
set variable main::granularity = main::pool_size/main::iteration
printf "GRN %d\n", main::granularity
set logging off
set variable main::flag = 1
delete

# Size classes extraction
break get_alloc_info.cu:main:break_03
continue
break mmap
set variable $class = 0
set variable main::finished = 0
continue
while main::finished == 0
	continue
	continue
	set variable main::class_finished = 1
	if main::inf_size == (main::sup_size - main::granularity)
		set variable main::finished = 1
	else
		set variable $class = $class + 1
		set variable $min_blocks = main::inf_size/main::granularity
		set variable $max_blocks = (main::sup_size - main::granularity)/main::granularity
		set variable $min_bytes =  ($min_blocks - 1) * main::granularity + 1
		set variable $max_bytes =  $max_blocks * main::granularity
		set logging on
		printf "SCL %d %d %d %d %d\n", $class, $min_blocks, $max_blocks, $min_bytes, $max_bytes
		set logging off
	end
	continue
	continue
end
delete

# Larger allocations multiple
break get_alloc_info.cu:main:break_04
continue
break mmap
continue
set logging on
printf "LGM %d\n", len - main::pool_size
set logging off
delete

# Allocator policy detection
break get_alloc_info.cu:main:break_05
continue
set logging on
if main::chunk_7 == main::chunk_5
	printf "POL Best fit\n"
else 
	if main::chunk_7 == main::chunk_6 + main::granularity
		printf "POL Next fit\n"
	else
		if main::chunk_7 == main::chunk_1
			if main::chunk_8 == main::chunk_1 + main::granularity
				printf "POL First fit\n"
			else
				if main::chunk_8 == main::chunk_3
					printf "POL Worst fit\n"
				else
					printf "POL Unrecognized policy\n"
				end
			end
		end
	end
end
set logging off
delete

# Coalescing support
break get_alloc_info.cu:main:break_06
continue
set logging on
if main::chunk_4 == main::chunk_1
	printf "CLS %d\n", 1
else 
	printf "CLS %d\n", 0
end
set logging off
delete

# Splitting support
break get_alloc_info.cu:main:break_07
continue
set logging on
if main::chunk_3 == main::chunk_1
	printf "SPL %d\n", 1
else 
	printf "SPL %d\n", 0
end
set logging off
delete

# Expansion policy
break get_alloc_info.cu:main:break_08
continue
break mmap
continue
set logging on
if main::index == main::max_allocations - 1
	printf "EXP When full\n"
else
	printf "EXP Treshold %d\n", main::index * main::granularity
end
set logging off
delete

# Pool usage
break get_alloc_info.cu:main:break_09
continue
set logging on
if main::chunk_10 == main::chunk_1
	printf "USG First available\n"
else
	if  main::chunk_10 == main::chunk_5
		printf "USG Best available\n"
	else
		if main::chunk_10 == main::chunk_9 + main::quarter
			printf "USG Last created\n"
		else
			printf "USG Unrecognized\n"
		end
	end
end
set logging off
delete

# Shrinking support
break get_alloc_info.cu:main:break_10
continue
break mmap
continue
set logging on
if  main::flag == 0
	printf "SHR Yes. Any pool deleted\n"
else
	if main::flag == 1
		printf "SHR Yes. Last pool deleted\n"
	else
		printf "SHR No. Pools deleted at the end\n"
	end
end
set logging off
delete

# Finalization
continue
quit