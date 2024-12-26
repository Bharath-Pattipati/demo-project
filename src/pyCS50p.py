# %% Import libraries

# %% Functions and Variables Examples
"""# combined functions (readability, length of line, simplify, etc.)
name = input("What is your name? \n").strip().title()  # input expects a string

# split into first and last name
first_name, last_name = name.split(" ")  # assign the first and last name to variables

print("Hello, %s!" % first_name)  # say hello to the user
print("Hello, {}!".format(first_name))  # another way to say hello to the user
print("Hello, " + first_name + "!")  # yet another way to say hello to the user
print(
    f"Hello, {first_name}! (most common way)"
)  # yet another way to say hello to the user"""

# %% Calculator
""" x = float(input("What's x? "))
y = float(input("What's y? "))
z = round(x + y, 4)  # round to the nearest whole number
print(f"Sum = {z:,}")  # print the sum with commas
print(f"Sum = {z:.2f}")  # print the sum with 2 decimal place """


# %% functions
""" def main():
    name = input("What is your name? \n").strip().title()  # input expects a string
    hello(name)  # call the function with the name as an argument


def hello(to="world"):  # default value for name
    print(f"Hello, {to}!")


main()  # call the main function """

# %% Compare
""" x = int(input("What's x? "))
y = int(input("What's y? "))

if x > y:
    print(f"{x} is greater than {y}")
elif x < y:
    print(f"{x} is less than {y}")
else:
    print(f"{x} is equal to {y}") """

# %% Grade
""" score = float(input("What's your score? "))

if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
elif score >= 70:
    print("Grade: C")
elif score >= 60:
    print("Grade: D")
else:
    print("Grade: F") """


# %% Parity
""" def main():
    x = int(input("What's x? "))
    parity(x)


def parity(x):
    if is_even(x):
        print(f"{x} is even")
    else:
        print(f"{x} is odd")


def is_even(n):
    return n % 2 == 0


main()  # call the main function """


# %% House
""" def main():
    name = input("What's your name? ").strip().title()
    house(name)


def house(person):
    match person:  # pattern matching
        case "Harry" | "Hermione" | "Ron":
            print("Gryffindor")
        case "Draco":
            print("Slytherin")
        case "Luna":
            print("Ravenclaw")
        case "Cedric":
            print("Hufflepuff")
        case _:
            print("Muggle")


main()  # call the main function """

# %% Loops example (CAT)
""" i = 0
while i < 3:
    print("Meow!")
    i += 1 """

""" n = int(input("How many times should the CAT talk? "))
for _ in range(n):
    print("Meow!")

print("Purr!\n" * n, end="") """

# %% Hogwarts Example (LOOPS, LIST, DICTIONARY)
""" students = ["Harry", "Ron", "Hermione", "Draco", "Luna", "Cedric"]
houses = [
    "Gryffindor",
    "Gryffindor",
    "Gryffindor",
    "Slytherin",
    "Ravenclaw",
    "Hufflepuff",
]

for student in students:  # anything that is iterable
    print(f"{student} belongs to {houses[students.index(student)]}") """

# Dictionary: key-value pairs
""" students = {
    "Harry": "Gryffindor",
    "Ron": "Gryffindor",
    "Hermione": "Gryffindor",
    "Draco": "Slytherin",
    "Luna": "Ravenclaw",
    "Cedric": "Hufflepuff",
}

# print(students["Harry"])  # access the value using the key

# Dictionary loops over the keys by default
for student, house in students.items():  # iterate over the dictionary
    print(f"{student} belongs to {house}") """

# List of dictionaries
""" students = [
    {"name": "Harry", "house": "Gryffindor", "patronus": "stag"},
    {"name": "Ron", "house": "Gryffindor", "patronus": "Jack Russell Terrier"},
    {"name": "Hermione", "house": "Gryffindor", "patronus": "otter"},
    {"name": "Draco", "house": "Slytherin", "patronus": "dragon"},
    {"name": "Luna", "house": "Ravenclaw", "patronus": "hare"},
    {"name": "Cedric", "house": "Hufflepuff", "patronus": "badger"},
]

for student in students:
    print(f"{student['name']}, {student['house']}, {student['patronus']}")
 """


# %% MARIO GAME
def main():
    height = get_height()
    #    draw_pyramid(height)
    print_square(height)


def get_height():
    while True:
        try:
            height = int(input("Height (1 to 10): "))
            if 1 <= height <= 10:
                return height
        except ValueError:
            pass


""" def draw_pyramid(height):
    for i in range(height):
        print(" " * (height - i - 1) + "#" * (i + 1)) """


def print_square(size):
    for i in range(size):
        print("#" * size)


main()  # call the main function
