#!/usr/bin/env python
# coding: utf-8

# In[1]:


# cl3 4 fuzzy set
def fuzzy_union(A, B):
    return {x: max(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_intersection(A, B):
    return {x: min(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_complement(A):
    return {x: 1 - A[x] for x in A}

def fuzzy_difference(A, B):
    return {x: min(A.get(x, 0), 1 - B.get(x, 0)) for x in set(A) | set(B)}

def cartesian_product(A, B):
    return {(x, y): min(A[x], B[y]) for x in A for y in B}

def max_min_composition(R, S):
    # Ensure that the second element of R and the first element of S match
    result = {}
    for (x, y), r_val in R.items():
        for (y_prime, z), s_val in S.items():
            if y == y_prime:
                if (x, z) not in result:
                    result[(x, z)] = r_val if r_val < s_val else s_val
                else:
                    result[(x, z)] = max(result[(x, z)], min(r_val, s_val))
    return result

# Define fuzzy sets
A = {'a': 0.5, 'b': 0.8}
B = {'b': 0.6, 'c': 0.9}

# Operations on fuzzy sets
print("Union:", fuzzy_union(A, B))
print("Intersection:", fuzzy_intersection(A, B))
print("Complement of A:", fuzzy_complement(A))
print("Difference A - B:", fuzzy_difference(A, B))

# # Define fuzzy relations
R = {('x1', 'y1'): 0.2, ('x1', 'y2'): 0.5, ('x2', 'y1'): 0.8}
S = {('y1', 'z1'): 0.3, ('y2', 'z1'): 0.6, ('y1', 'z2'): 0.4}

R2 = {('x1', 'y1'): 0.7, ('x1', 'y2'): 0.5, ('x2', 'y1'): 0.8, ('x2', 'y2'): 0.4 }
S2 = {('y1', 'z1'): 0.9, ('y1', 'z2'): 0.6, ('y2', 'z1'): 0.1, ('y2', 'z2'): 0.7 }
# # Cartesian product and max-min composition
print("Cartesian Product of A and B:", cartesian_product(A, B))
# print("Max-Min Composition of R and S:", max_min_composition(R, S))
print("Max-Min Composition of R and S:", max_min_composition(R2, S2))


# In[2]:


# To apply the artificial immune patter recognition to perform a task of structure damage
# Classification.
# cl3 7 immune
import numpy as np

def generate_dummy_data(samples=100, features=10):
    """Generate random data and binary labels."""
    data = np.random.rand(samples, features)
    labels = np.random.randint(0, 2, size=samples)
    return data, labels

class AIRS:
    def __init__(self, num_detectors=5):
        """Initialize AIRS with a specified number of detectors."""
        self.num_detectors = num_detectors
        self.detectors = None

    def train(self, X):
        """Select a subset of data as detectors randomly."""
        indices = np.random.choice(len(X), self.num_detectors, replace=False)
        self.detectors = X[indices]

    def predict(self, X):
        """Predict labels based on nearest detector using Euclidean distance."""
        predictions = []
        for sample in X:
            distances = np.linalg.norm(self.detectors - sample, axis=1)
            prediction = np.argmin(distances)
            predictions.append(prediction)
        return predictions

# Create dummy data and split it
data, labels = generate_dummy_data(samples=50, features=5)  # Smaller dataset for simplicity
split_index = int(len(data) * 0.8)
train_data, test_data = data[:split_index], data[split_index:]

# Initialize, train, and test AIRS
airs = AIRS(num_detectors=3)
airs.train(train_data)
predictions = airs.predict(test_data)

# Calculate and print accuracy
accuracy = np.mean(predictions == labels[split_index:])
print(f"Accuracy: {accuracy:.2f}")


# In[ ]:




