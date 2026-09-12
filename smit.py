largest = 0 
for i in range(3):
    num = int(input("Enter a number: "))
    if num > largest:
        largest = num
print("The largest number is:", largest)