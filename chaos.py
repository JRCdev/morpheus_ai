import random as r
from iris import iris_search as search


vector_len = 384
r.seed()

arr = [r.random()*2-1 for _ in range(vector_len)]
print(arr)
#print(search("Why",1))
print(search(arr,1))