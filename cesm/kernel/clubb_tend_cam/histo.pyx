[readdata]


model_diffs = []

with open("model.ini") as f:

	etimesec = "[elapsedtime.elapsedtime]"
	sec = None

	for line in f:
		if line.startswith("["):
			if line.startswith(etimesec):
				sec = line
			else:
				sec = None	
		elif sec is not None:	
			if not line: continue
			splitted = [l.strip() for l in line.split("=")]
			if len(splitted) == 2:
				mpi, omp, invoke = [int(n) for n in splitted[0].split()]
				start, stop = [float(f) for f in splitted[1].split(",")]
				diff = stop - start
				if diff < 0.006:
					model_diffs.append(diff)

kernel_etimes = []

with open("kernel_2ranks.txt") as f:
#with open("kernel_16ranks.txt") as f:
#with open("kernel_32ranks.txt") as f:

    niter, nranks, ncache_pollution = [v.strip() for v in f.readline().split()]

    for line in f:
        splitted = line.split()
        kernel_etimes.append(float(splitted[-1].strip())*1E-6)

[plotdata]

import matplotlib.pyplot as plt
#import numpy as np
#from matplotlib import colors
#from matplotlib.ticker import PercentFormatter

n_bins = 100

# Generate a normal distribution, center at x=0 and y=5

#plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(16,8))

TITLESIZE = 18
TEXTSIZE = 12

# We can set the number of bins with the `bins` kwarg
axs[0].hist(model_diffs, bins=n_bins)
#axs[0].hist([], bins=n_bins)
axs[0].set_title("CLUBB Timing From CESM", fontsize=TITLESIZE)
axs[0].set_xlabel("Elapsed time(sec)", fontsize=TEXTSIZE)
axs[0].set_ylabel("Frequency", fontsize=TEXTSIZE)
#axs[1].hist(kernel_etimes, bins=5, fontsize=TEXTSIZE)
axs[1].hist(kernel_etimes, bins=10)
axs[1].set_title("CLUBB Timing From Kernel\n(NITER=%s, NRANKS=%s, CACHE_POLLUTION=%s)"%(niter, nranks, ncache_pollution), fontsize=TITLESIZE)
axs[1].set_xlabel("Elapsed time(sec)", fontsize=TEXTSIZE)

plt.savefig("clubb_timing_niter%s_nranks%s_ncache%s.pdf"%(niter, nranks, ncache_pollution))

plt.show()
