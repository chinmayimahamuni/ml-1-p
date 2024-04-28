#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mrjob.job import MRJob

class WordCount(MRJob):

    def mapper(self, _, line):
        for word in line.split():
            yield word, 1

    def reducer(self, key, values):
        yield key, sum(values)

# Create an instance of the WordCount job
job = WordCount()

# Run the job with inline input data
input_data = ["hello world hello", "world world"]
output_data = job.run_job(input_data=input_data)

# Print the output
for line in output_data:
    print(line)


# In[2]:


from mrjob.job import MRJob

class WordCount(MRJob):

    def mapper(self, _, line):
        for word in line.split():
            yield word, 1

    def reducer(self, key, values):
        yield key, sum(values)

# Create an instance of the WordCount job
job = WordCount()

# Remove unrecognized arguments before running the job
import sys
args = sys.argv[1:]  # Remove the first argument (which is the script name)
if '-f' in args:
    args.remove('-f')

# Run the job with the modified arguments
job.run(args=args)


# In[3]:


from mrjob.job import MRJob
from mrjob.compat import get_jobconf_value

class WordCount(MRJob):

    def mapper(self, _, line):
        for word in line.split():
            yield word, 1

    def reducer(self, key, values):
        yield key, sum(values)

# Create an instance of the WordCount job
job = WordCount()

# Get command-line arguments excluding Jupyter-specific arguments
args = job.get_sys_argv()[1:]

# Run the job with the modified arguments
job.run(args=args)


# In[4]:


def matrix_multiply(A, B):
    # Check if matrices can be multiplied
    if len(A[0]) != len(B):
        raise ValueError("Matrix A's columns must match Matrix B's rows.")

    # Map phase: Calculate products for each cell of the result matrix
    def map_phase():
        # Generate tuples (i, k, value) where i, k are indices of the result matrix cell
        # value is A[i][j] * B[j][k]
        products = []
        for i in range(len(A)):
            for k in range(len(B[0])):
                for j in range(len(B)):
                    products.append((i, k, A[i][j] * B[j][k]))
        return products

    # Reduce phase: Sum all products corresponding to each cell in the result matrix
    def reduce_phase(products):
        result = [[0]*len(B[0]) for _ in range(len(A))]
        for i, k, product in products:
            result[i][k] += product
        return result

    # Execute phases
    products = map_phase()
    result = reduce_phase(products)
    return result

# Example matrices
A = [[1, 2], [3, 4]]
B = [[2, 0], [1, 2]]

# Multiply matrices
result = matrix_multiply(A, B)

# Print result
print("Result of Matrix Multiplication:")
for row in result:
    print(row)


# In[ ]:




