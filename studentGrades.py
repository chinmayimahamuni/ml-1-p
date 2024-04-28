import numpy as np
from mrjob.job import MRJob

class MRStudentGrades(MRJob):

    def mapper(self, _, line):
        # Assuming each line in the file is in the format "StudentID,Grade"
        parts = line.split(',')
        student_id = parts[0].strip()
        grade = parts[1].strip()
        yield student_id, grade

    def reducer(self, key, values):
        # Collect all grades for each student
        grades = list(values)
        m = grades
        yield key, grades

if __name__ == '__main__':
    MRStudentGrades.run()
