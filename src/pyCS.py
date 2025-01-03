"""
CS50p Course: Introduction to Programming with Python
"""

# %% Import libraries
# from random import choice, randint, shuffle
# import statistics as stats
# import sys
# import json

# import cowsay
# import requests

# import csv
# import sys
# from PIL import Image

import re

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
""" def main():
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


"""
""" def draw_pyramid(height):
    for i in range(height):
        print(" " * (height - i - 1) + "#" * (i + 1)) 


def print_square(size):
    for i in range(size):
        print("#" * size)


main()  # call the main function """

# %% Exception Handling
""" while True:  # run infinite loop until a valid number is entered
    try:
        x = int(input("Enter a number: "))
    except ValueError:
        print("x is not an Integer")
    else:  # executes if no exception is raised
        break

print(f"x = {x}") """

# %% Generate

""" coin = choice(["heads", "tails"])  # randomly select an item from the list
print(coin)

dice = randint(1, 10)  # randomly select an integer between 1 and 6
print(dice)

cards = ["jack", "queen", "king"]  # create a list of 52 cards
shuffle(cards)  # shuffle the list
print(cards)  # print cards

print(stats.mean([1, 2, 3, 4, 5]))  # calculate the mean of the list """

# %% System Module
# Execute in terminal using: python src/pyCS50p.py "Name"
# print("Hello, my name is", sys.argv[0])  # print the name of the script

""" if len(sys.argv) < 2:
    sys.exit("Too few arguments")

for arg in sys.argv[1:]:  # slice the list to exclude the script name
    print(f"Hello, {arg}!") """

# %% SAY
# Terminal command: python src/pyCS50p.py Moooo
""" if len(sys.argv) == 2:
    cowsay.cow("Hello, " + sys.argv[1]) """

# %% Website Requests: iTUNES API
# JSON: JAvascript Object Notation
# API: Application Programming Interface
# URL: Uniform Resource Locator
# URI: Uniform Resource Identifier

""" if len(sys.argv) != 2:
    sys.exit("Usage in Terminal: python src/pyCS50p.py <artist>")

response = requests.get(f"https://itunes.apple.com/search?term={sys.argv[1]}&limit=10")
# print(json.dumps(response.json(), indent=2))

o = response.json()
for i in o["results"]:
    print(i["trackName"], i["releaseDate"]) """


# %% Unit Testing for Calculator written in testCalculator.py
""" def main():
    x = int(input("What's x? "))
    print("x squared is", square(x))


def square(n):
    return n * n


if __name__ == "__main__":
    main()  # call the main function """

# %% File IO
# names = []
""" for _ in range(3):
    names.append(input("What's your Name?"))

with open("names.txt", "a") as file:
    for name in names:
        file.write(name + "\n") """

""" with open("names.txt", "r") as file:
    for line in file:
        names.append(line.strip())

for name in sorted(names, reverse=True):
    print(name)  # print the sorted names """

""" with open("src/students.csv") as file:
    for line in file:
        name, house = line.strip().split(",")
        print(f"{name} is in {house}") """

# students = []

""" with open("src/students.csv") as file:
    for line in file:
        name, house = line.strip().split(",")
        students.append({"name": name, "house": house}) """


""" def get_name(student):
    return student["name"] """

""" for student in sorted(students, key=get_name, reverse=True):
    print(f"{student['name']} is in {student['house']}")  # print the sorted names """

""" for student in sorted(
    students, key=lambda student: student["name"]
):  # lambda anonymous function
    print(f"{student['name']} is in {student['home']}")  # print the sorted names
 """

""" # Writing to file
name = input("What's your name? ")
home = input("What's your home? ")

with open("src/students.csv", "a") as file:
    writer = csv.DictWriter(file, fieldnames=["name", "home"])
    writer.writerow(
        {"name": name, "home": home}
    )  # write the name and house to the file """

# %% Images
""" images = []

for arg in sys.argv[1:]:  # slice the list to exclude the script name
    image = Image.open(arg)
    images.append(image)

# images[0].show()
images[0].save(
    "output.gif", save_all=True, append_images=images[1:], duration=200, loop=0
) """
# %% Regular Expressions: EMAIL
# email = input("What's your email? ").strip()
""" if "@" in email and "." in email:
    print("Valid email")
else:
    print("Invalid email") """

""" username, domain = email.split("@")

if username and domain.endswith(".com"):
    print("Valid email")
else:
    print("Invalid email")
 """


# if re.search(r"^\w+@(\w+\.)?\w+\.(com|edu|gov|net|org)$", email, re.IGNORECASE):
#    print("Valid email")
# else:
#    print("Invalid email")


# %% Regular Expressions: FORMAT
""" if "," in name:
    last, first = name.split(",")
    print(f"Hello, {first.strip()} {last.strip()}!") """


# name = input("What's your name? ").strip()
# re.search(r"^\w+$", name)
""" if matches := re.search(r"^(.+), *(.+)$", name):
    name = matches.group(2) + " " + matches.group(1)  # location 0
print(f"Hello, {name}!") """


# Twitter: https://x.com/davidjmalan?mx=2
url = input("URL: ").strip()

# username = re.sub(r"^(https?://)?(www\.)?x.com/", "", url)
# print(f"Username: {username}")

""" if matches:
    print(f"Username: {matches.group('username')}")
else:
    print("Invalid URL") """

if matches := re.search(r"^(https?://)?(www\.)?x.com/(?P<username>\w+)", url, re.I):
    print(f"Username: {matches.group('username')}")
else:
    print("Invalid URL")
