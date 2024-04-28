#!/usr/bin/env python
# coding: utf-8

# In[1]:


def map_phase(data):
    # Map phase: create key-value pairs for each record
    # Key: (gender, survived), Value: (age, count)
    result = []
    for record in data:
        key = (record['gender'], record['survived'])
        value = (record['age'], 1)
        result.append((key, value))
    return result


def reduce_phase(mapped_data):
    # Reduce phase: aggregate data by key
    reduced = {}
    for key, value in mapped_data:
        if key not in reduced:
            reduced[key] = {'total_age': 0, 'count': 0}
        reduced[key]['total_age'] += value[0]
        reduced[key]['count'] += value[1]
    return reduced

def calculate_results(reduced_data):
    # Calculate average age and counts
    results = {}
    for key, value in reduced_data.items():
        if key[1] == 0:  # Only for those who died
            average_age = value['total_age'] / value['count']
            results[f"Average age of died {key[0]}s"] = average_age
        if key[1] == 1:  # Count survivors by gender
            results[f"Number of survived {key[0]}s"] = value['count']
    return results

data = [
    {'gender': 'male', 'age': 22, 'survived': 0},
    {'gender': 'female', 'age': 28, 'survived': 0},
    {'gender': 'male', 'age': 42, 'survived': 0},
    {'gender': 'female', 'age': 36, 'survived': 1},
    {'gender': 'male', 'age': 30, 'survived': 1},
    {'gender': 'female', 'age': 24, 'survived': 1},
]

# Perform MapReduce operations
mapped_data = map_phase(data)
reduced_data = reduce_phase(mapped_data)
results = calculate_results(reduced_data)

# Print the results
for result, value in results.items():
    print(f"{result}: {value:.2f}" if isinstance(value, float) else f"{result}: {value}")


# In[ ]:




