# import libraries
import random

# create dictionary of questions and answers
python_qna = {
    "What is Python?": "Language",
    "How do you declare a variable?": "Assignment",
    "What data type is `[1, 2, 3]`?": "List",
    "What is `(1, 2, 3)` called?": "Tuple",
    "What is `{‘a’: 1, ‘b’: 2}`?": "Dictionary",
    "Which loop runs indefinitely?": "While",
    "What keyword defines a function?": "def",
    "What keyword creates a class?": "class",
    "What is `True` and `False`?": "Boolean",
    "How do you start a comment?": "Hash",
    "What keyword exits a loop?": "break",
    "What keyword skips an iteration?": "continue",
    "What is used for string formatting?": "f-string",
    "What operator checks equality?": "==",
    "What keyword handles exceptions?": "try",
}


# create function to ask 5 random questions and check answers
def ask_questions(k):
    score = 0
    for index, question in enumerate(random.sample(list(python_qna), k)):
        print(f"{index + 1}. {question}")
        answer = input("Answer: ").lower().strip()
        if answer == python_qna[question].lower():
            score += 1
        else:
            print(f"Wrong answer. Correct answer is {python_qna[question]}")
    return score


if __name__ == "__main__":
    k = 5
    score = ask_questions(k)
    print(f"Score = {score / k * 100:.2f}%")
# print score at end of game
