"""
CS50p Course: Introduction to Programming with Python
"""

# %% Import libraries
# from math import sqrt  # import specific function from a library

# import this # The Zen of Python, by Tim Peters

# from random import choice, randint, shuffle
# import statistics as stats
# import random
# import sys
# import json

# import cowsay
# import requests

# import csv
# import sys
# from PIL import Image

# import re
# import argparse

# %% Functions and Variables Examples
# combined functions (readability, length of line, simplify, etc.)
""" name = input("What is your name? \n").strip().title()  # input expects a string

# split into first and last name
first_name, last_name = name.split(" ")  # assign the first and last name to variables

print("Hello, %s!" % first_name)  # say hello to the user
print("Hello, {}!".format(first_name))  # another way to say hello to the user
print("Hello, " + first_name + "!")  # yet another way to say hello to the user
print(
    f"Hello, {first_name}! (most common way)"
)  # yet another way to say hello to the user

place = input("what place to do live? \n").strip().title()
print(f"{first_name} lives in {place}") """

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
    print(f"{student['name']}, {student['house']}, {student['patronus'].capitalize()}") """


# %% MARIO GAME
""" def main():
    height = get_height()
    draw_pyramid(height)
    # print_square(height)


def get_height():
    while True:
        try:
            height = int(input("Height (1 to 10): "))
            if 1 <= height <= 10:
                return height
        except ValueError:
            pass


def draw_pyramid(height):
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
# url = input("URL: ").strip()

# username = re.sub(r"^(https?://)?(www\.)?x.com/", "", url)
# print(f"Username: {username}")

""" if matches:
    print(f"Username: {matches.group('username')}")
else:
    print("Invalid URL") """

# if matches := re.search(r"^(https?://)?(www\.)?x.com/(?P<username>\w+)", url, re.I):
#   print(f"Username: {matches.group('username')}")
# else:
#    print("Invalid URL")


# %% Object-Oriented Programming (OOP)
""" class Student:  # Class is mutable but can be made immutable by using dataclasses
    def __init__(  # default function that will always run when the class is called
        self, name, house, patronous=None
    ):  # initialize objects of the class, adding instance variables to objects.
        self.name = name  # self is used to store the instance variables
        self.house = house  # goes through getter and setter
        self.patronus = patronous

    def __str__(
        self,
    ):  # python will automatically call this function when another function calls this object as a string
        return f"{self.name} is from {self.house}"

    def __repr__(self):  # representation of the object, more for developers, debugging
        return f"Student({self.name}, {self.house})"

    # Getter
    @property
    def house(self):
        return self._house

    # Setter
    @house.setter
    def house(self, house):
        if house not in ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]:
            raise ValueError("Invalid house!")
        self._house = house

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if not name:
            raise ValueError("Missing name!")
        self._name = name

    def charm(self):
        match self.patronus:
            case "Stag":
                return "🐴"
            case "Otter":
                return "🦦"
            case "Dragon":
                return "🐉"
            case "Jack Russell Terrier":
                return "🐶"
            case _:  # default case
                return "🐱"

    @classmethod
    def get(cls):
        name = input("Name: ")
        house = input("House: ")
        patronus = input("Patronus: ")
        return cls(name, house, patronus) """


""" def get_student():  # can return tuple (), list [] or dictionary {}, you can also create and return at the same time
    return Student(
        input("Name: "), input("House: "), input("Patronus: ")
    )  # create object of class Student i.e. instance of the class, Constructor Call. """


""" def main():
    student = Student.get()
    print(student)
    print("Expecto Patronum: ", student.charm())


if __name__ == "__main__":
    main()  # call the main function """

# %% OOP Types
""" print(type(50.0))
print(type("Hello, World!"))
print(type([]))
print(type({}))
print(type(())) """


# %% OOP Sorting Hat
""" class Hat:
    houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]

    @classmethod
    def sort(cls, name):
        print(f"{name} belongs to {random.choice(cls.houses)}")


hat = Hat()  # instantiating object of certain class
hat.sort("Harry") """


# %% OOP Inheritance
""" class Wizard:
    def __init__(self, name):
        if not name:
            raise ValueError("Missing name!")
        self.name = name


class Student(Wizard):
    def __init__(self, name, house):
        super().__init__(name)
        self.house = house

    ...


class Professor(Wizard):
    def __init__(self, name, subject):
        super().__init__(name)
        self.subject = subject

    ...


wizard = Wizard("Dumbledore")
student = Student("Harry", "Gryffindor")
professor = Professor("Snape", "Potions")
print(wizard.name)
print(student.name, student.house)
print(professor.name, professor.subject) """


# %% OOP: Operator Overloading
""" class Vault:
    def __init__(self, galleons=0, sickles=0, knuts=0):
        self.galleons = galleons
        self.sickles = sickles
        self.knuts = knuts

    def __str__(self):
        return f"{self.galleons} galleons, {self.sickles} sickles, {self.knuts} knuts"

    def __add__(self, other):  # overloading the + operator
        galleons = self.galleons + other.galleons
        sickles = self.sickles + other.sickles
        knuts = self.knuts + other.knuts
        return Vault(galleons, sickles, knuts)


potters = Vault(7, 21, 42)
print(potters)

weasleys = Vault(3, 7, 21)
print(weasleys)

total = potters + weasleys
print(total) """

# %% Miscellaneous concepts
""" students = [
    {"name": "Harry", "house": "Gryffindor"},
    {"name": "Ron", "house": "Gryffindor"},
    {"name": "Hermione", "house": "Gryffindor"},
    {"name": "Draco", "house": "Slytherin"},
    {"name": "Luna", "house": "Ravenclaw"},
    {"name": "Cedric", "house": "Hufflepuff"},
]

houses = (
    set()
)  # set is a collection of unique elements, duplicates are automatically removed
for student in students:
    houses.add(student["house"])

for house in sorted(houses):
    print(house) """

# %% Bank
""" balance = 0


def main():
    print("Balance: ", balance)
    deposit(100)
    withdraw(50)
    print("Balance: ", balance)


def deposit(amount):
    global balance
    balance += amount


def withdraw(amount):
    global balance
    balance -= amount


if __name__ == "__main__":
    main()  # call the main function """


# %% Bank Account: class instance variables and methods
""" class Account:
    def __init__(self):
        self._balance = 0  # special parameter self is used to store the instance variables and make it accessible to all functions within class

    @property  # getter
    def balance(self):
        return self._balance

    def __str__(self):
        return f"Balance: {self.balance}"

    def deposit(self, n):
        self._balance += n

    def withdraw(self, n):
        self._balance -= n


def main():
    account = Account()
    print(account)
    account.deposit(100)
    account.withdraw(50)
    print(account)


if __name__ == "__main__":
    main()  # call the main function """


# %% Constants
""" class Cat:
    MEOWS = 3  # class variable, shared by all instances of the class

    def meow(self):
        print("Meow!\n" * Cat.MEOWS)


cat = Cat()
cat.meow() """


# def meow(n: int) -> str:  # indicate arg type (int) and return type (str)
#    """Meow n times.

#    Args:
#        n (int): _description_

#    Returns:
#        str: _description_
#    """
#    return "meow\n" * n


# number: int = int(input("Number: "))  # run mypy src/pyCS.py to check correct types
# meows: str = meow(number)
# print(meows)

# %% Argument Parsing
""" if len(sys.argv) == 1:
    print("meow")
elif len(sys.argv) == 3 and sys.argv[1] == "-n":  # python src/pyCS.py -n 5
    n = int(sys.argv[2])
    print("meow\n" * n)
else:
    print("usage: meows.py") """

""" parser = argparse.ArgumentParser(description="Meow like a cat")
parser.add_argument("-n", default=1, help="number of times to meow", type=int)
args = parser.parse_args()

for _ in range(int(args.n)):
    print("meow") """

# %% Unpacking
# first, last = input("What's your full name? ").split()
# print(f"Hello, {first}")


""" def total(galleons=0, sickles=0, knuts=0):
    return (galleons * 17 + sickles) * 29 + knuts """


# coins = [100, 50, 25] # List
# print(total(*coins), "Knuts") # List unpacking

""" coins = {"galleons": 100, "sickles": 50, "knuts": 25}
print(total(**coins), "Knuts")  # unpack dictionary """


""" def f(*args, **kwargs):
    print(f"Positional arguments: {args} of {type(args)}")  # Tuple
    print(f"Positional aruments: {kwargs} of {type(kwargs)}")  # Tuple


f(100, 50, 25, 5)
f(galleons=100, sickles=50, knuts=25) """


# %% MAP & List Comprehenson
""" def main():
    yell("This", "is", "CS50!")


def yell(*words):
    uppercased = map(
        str.upper, words
    )  # map(function, iterable), iterate over each word/sequence and apply function, functional programming
    print(*uppercased)
    uppercase = [word.upper() for word in words]  # list comprehension
    print(*uppercase)


if __name__ == "__main__":
    main() """

# %% Gryffindors: List and Dict comprehension, FILTER function
""" students = [
    {"name": "Harry", "house": "Gryffindor"},
    {"name": "Ron", "house": "Gryffindor"},
    {"name": "Hermione", "house": "Gryffindor"},
    {"name": "Draco", "house": "Slytherin"},
    {"name": "Luna", "house": "Ravenclaw"},
    {"name": "Cedric", "house": "Hufflepuff"},
] """

""" gryffindors = [
    student["name"] for student in students if student["house"] == "Gryffindor"
]

for gryffindor in sorted(gryffindors):
    print(gryffindor) """


""" def is_gryffindor(s):
    return s["house"] == "Gryffindor"


gryffindors = filter(is_gryffindor, students)

for gryffindor in sorted(gryffindors, key=lambda s: s["name"]):
    print(gryffindor["name"]) """


# students = ["Harry", "Hermoine", "Ron"]

""" gryffindors = [
    {"name": student, "house": "Gryffindor"} for student in students
]  # Dict comprehension

print(gryffindors)

gryffindors = {student: "Gryffindor" for student in students}
print(gryffindors) """

""" for i, student in enumerate(students):
    print(i + 1, student) """


# %% SLEEP: Generators with YIELD Keyword
""" def main():
    n = int(input("What's n: "))
    for s in sheep(n):
        print(s)


def sheep(n):
    for i in range(n):
        yield "🐱" * i


if __name__ == "__main__":
    main() """


# %% Recursion
""" def multiply(a, b):
    if b == 0:
        return 0
    return a + multiply(a, b - 1)


print(multiply(3, 4))  # `12` """

# %% Stripping Whitespace and displaying large numbers
""" text = input("Text: ")
print(text.strip())

largeNum = 14_000_000_000
print(largeNum) """

# %% List Examples
""" bicycles = ["Trek", "Cannondale", "Redline", "Specialized"]
print(bicycles[-1])  # print last item in the list

bicycles[0] = "Giant"  # change first item in the list
print(bicycles)

bicycles.append("Schwinn")  # add item to the end of the list
print(bicycles) """

""" magicians = ["alice", "david", "carolina", "zeus"]
for magician in sorted(magicians):
    print(magician.title() + ", that was a great trick!")

magicians.pop()
print(magicians) """

# %% List comprehension and slicing
""" squares = [value**2 for value in range(1, 11)]
print(f"Squares: {squares}")

sqrtValues = [sqrt(value) for value in range(1, 11)]
print(f"Sum of sqrt: {sum(sqrtValues)}")

cubes = [v**3 for v in range(1, 11)]
print(f"Cubes: {cubes[:5]}") """

# %% Dictionary Examples

""" # dictionary: 3 major rivers and countires in which they flow
rivers = {
    "nile": "egypt",
    "amazon": "brazil",
    "yangtze": "china",
}

for r in rivers:
    print(f"The {r.title()} runs through {rivers[r].title()}")

for r in rivers.keys():
    print(r)

for r in rivers.values():
    print(r)

# Make aliens dictionary
aliens = []

for a in range(30):
    new_alien = {"color": "green", "points": 5, "speed": "slow"}
    aliens.append(new_alien)

for alien in aliens[:5]:
    print(alien)
print("...............") """

# %% Interacting with user
""" age = input("How old are you? ")

if int(age) >= 18:
    print("You are old enough to vote!")
else:
    print("You are not old enough to vote!") """

""" cnt = 0
while cnt < 3:
    print(f"Count: {cnt}")
    cnt += 1
print("Done") """

""" active = True
while active:
    message = input("Tell me something: ")
    if message == "quit":
        active = False
    else:
        print(message) """

# filling dictionary using while loop user input
responses = {}
polling_active = True
while polling_active:
    name = input("what is your name? ")
    response = input("which city would you like to visit? ")

    responses[name] = response

    repeat = input("Would you like to let another person respond? (yes/no) ")
    if repeat == "no" or repeat == "n":
        polling_active = False

print("\n--- Poll Results ---")
for name, response in responses.items():
    print(f"{name} would like to visit {response}.")

# %%
