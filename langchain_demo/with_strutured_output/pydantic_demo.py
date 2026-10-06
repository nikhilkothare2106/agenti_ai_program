from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = "nikhil"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(
        gt=0,
        lt=10,
        default=7,
        description="A decimal value representing the cgpa of the student",
    )


stu1 = {"name": "nk", "age": 3, "email": "abc@gmail.com", "cgpa": 5}
# stu1 = {"name": "nk", "age": '3'} # pydantic conver str to int
student = Student(**stu1)
print(student)
print(type(student))
print(dict(student))
student_json = student.model_dump_json()
print(student_json)
