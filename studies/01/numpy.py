import numpy as np

# tasks:

# create arrays with 0's, 1's and arbitrary numbers with size of 7

zeroes = np.zeros(7)
ones = np.ones(7)
fours = np.full(7, 4.0)

print("zeroes ", zeroes)
print("ones ", ones)
print("fours ", fours)

# create array from list

arr = np.array([0, 1, 2, 5, 7, 12])

# create array with range; change number on some index

range1 = np.arange(9)
range2 = np.arange(2, 9)

range1[1] = 9

print("range1 ", range1)

# create array in range 0-100 with a step of 10

linspace = np.linspace(0, 100, 11)

print("linspace ", linspace)

# ------ Multi-dimensional arrays

# create multi-dimensional arrays with 0's, 1's and arbitrary number; create from python's list of lists

m_zeroes = np.zeros((2, 3))
m_ones = np.ones((2, 3))
m_fours = np.full((2, 3), 4)

# access & change element, row and column

m_zeroes[0, 2] = 10
m_zeroes[0] = [1, 2, 3]
m_zeroes[:, 2] = [3, 4]

print("m_zeroes (alt) ", m_zeroes)
print("m_ones ", m_ones)
print("m_fours ", m_fours)

# ------ Random

# create an array with random numbers 0-1, with size (2, 3); make results reproducible;

np.random.seed(4)
rand_arr = np.random.rand(2, 3)

# create random array with values 0-100; create same with integers;

rand_arr1 = 100 * np.random.rand(2, 3)
rand_arr2 = 100 * np.random.randint(low=0, high=100, size=(2, 3))

# create random array with normal distribution;

rand_arr3 = np.random.randn(2, 3)

print("rand_arr ", rand_arr)
print("rand_arr1 ", rand_arr1)
print("rand_arr2 ", rand_arr2)
print("rand_arr3 ", rand_arr3)

# ------ Operations

# create any array, make element-wise operations on it with +, -, *, /, **

arr = np.arange(0, 6)

arr = arr + 1
arr = ((arr ** 2) + 8) / 2

print("arr (0-6 with operations) ", arr)

# make element-wise operations with another array

arr1 = np.array([0, 40, 3, 5, 1, 4])

print("arr + arr1 ", arr + arr1)

# ------ Comparisons

# compare array with number, with another array

print("arr > 20 ", arr > 20)
print("arr > arr1 ", arr > arr1)

# get an array of elements that satisfy a comparison

print("arr[arr > arr1] ", arr[arr > arr1])

# ------ min, max, sum, mean, std (standard deviation)

print("np.min(arr) ", np.min(arr))
print("np.max(arr) ", np.max(arr))
print("np.sum(arr) ", np.sum(arr))
print("np.mean(arr) ", arr.mean()) # <-- do this instead of np....

# same with multidimensional