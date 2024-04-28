#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict

def mapper(text):
    for word in text.split():
        yield (word, 1)

def reducer(mapped_values):
    reduction = defaultdict(int)
    for key, value in mapped_values:
        reduction[key] += value
    return reduction

def main(text):
    mapped_values = list(mapper(text))
    reduced_values = reducer(mapped_values)

    for word, count in sorted(reduced_values.items()):
        print(f'{word}: {count}')

if __name__ == '__main__':
    # Provide the text input directly within the notebook
    text = input("Enter text: ")
    main(text)


# In[ ]:




