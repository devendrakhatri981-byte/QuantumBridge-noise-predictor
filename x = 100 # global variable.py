import pickle

# Object to serialize (can be any Python object)
student = {
    "name": "Aryan",
    "age": 19,
    "branch": "AI/ML",
    "grades": [85, 90, 78, 95]
}

# Serialization — saving object to binary file
with open("student.pkl", "wb") as f:   # wb = write binary
    pickle.dump(student, f)

print("Object serialized and saved to student.pkl")
print("Original object:", student)