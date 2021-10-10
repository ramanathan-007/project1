import numpy
from scipy import stats
import matplotlib.pyplot as plt
dataset = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x=numpy.mean(dataset)
print("Mean:",x)

y=numpy.median(dataset)
print("Median:",y)

z=stats.mode(dataset)
print("Mode:",z)

s=numpy.std(dataset)
print("Standard Deviation:",s)

v=numpy.var(dataset)
print("Variance:",v)

p = numpy.percentile(dataset, 75)
print("Percentile:",p)

plt.hist(dataset,100)
plt.show()

m = [5,7,8,7,2,17,2,9,4,11,12,9,6]

plt.scatter(dataset,m)
plt.show()

