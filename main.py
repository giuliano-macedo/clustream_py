import numpy as np
import pandas as pd
from tqdm import tqdm

from CluStream import CluStream
import cProfile, pstats, io



def profile(fnc):
	
	"""A decorator that uses cProfile to profile a function"""
	
	def inner(*args, **kwargs):
		
		pr = cProfile.Profile()
		pr.enable()
		retval = fnc(*args, **kwargs)
		pr.disable()
		s = io.StringIO()
		sortby = 'cumulative'
		ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
		ps.print_stats()
		print(s.getvalue())
		return retval

	return inner
# @profile
from time import time
def main():
	model=CluStream(m=2000,h=1000)
	t=0
	total_time=0
	for chunk in tqdm(pd.read_csv("only_numerical.csv",chunksize=2000,dtype=np.float32),total=4898431//2000,disable=True):
		
		for datapoint in tqdm(chunk.values,disable=True):
			start=time()
			model.offline_cluster(datapoint,t)
			if t>model.m:
				total_time+=1//(time()-start)
			t+=1
		if t==4000:
			break
	print(total_time/((t-model.m)),"avg pps")
		
main()