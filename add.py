y= int(input("enter the no. 2:"))
x= int(input("enter the no. 1:"))
choice = input(" enter your choice(a/b/c/d)")
a= x+y
b= x-y
c= x*y
d= x/y
if choice== 'a':
 print("result:", x+y)
elif choice== 'b':
 print("result:", x-y)
elif choice== 'c':
 print("result:", x*y)
elif choice== 'b':
 if  y!= 0:
  print("result:", x/y)
 else:
  print("error! division by zero is not alloweed.")
else:
 print("invalid input")