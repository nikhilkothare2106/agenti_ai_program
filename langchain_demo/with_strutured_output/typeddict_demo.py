from typing import TypedDict

class Student(TypedDict):
    name: str
    age: int

stu1: Student = {'name':'abc','age':22}
print(stu1)