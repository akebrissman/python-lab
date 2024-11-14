import random

def task_1():
    # Skriv ett program som emot en sträng som input och skriver ut längden på strängen.
    # Exempel-input: "thisIsAString" Förväntad output: 13
    input_string = input("Write a string: ")
    print(f'Length: {len(input_string)}')


def task_2():
    # Skriv ett program som skriver ut frekvensen av tecken i en given sträng.
    # Exempel-input: "banana" Förväntad output: {"b":1, "a":3, "n":2}
    input_string = input("Write a string: ")


def task_3():
    # Skriv ett program som för en given sträng skriver ut de två första och de två sista tecknen i strängen (på valfritt format)
    # Exempel-input: "banana" Förväntad output: "ba na"
    input_string = input("Write a string: ")
    first_two = input_string[:2]
    last_two = input_string[-2:]
    print(f"first two: {first_two}, last two: {last_two}")


def task_4():
    # Skriv ett program som tar två strängar som input och skapar EN ny sträng där de två första tecken i varje sträng bytts ut.
    # Exempel-input: "abc", "xyz" Förväntad output: "xyc abz"
    input_string_a = "abcdef" # input("Write a string: ")
    input_string_b = "xyzåäö"  # input("Write a string: ")
    result_string_a = input_string_b[:2] + input_string_a[2:]
    result_string_b = input_string_a[:2] + input_string_b[2:]
    print(f"String 1: {result_string_a}, String 2: {result_string_b}")


def task_5():
    # Skriv ett program som lägger till "ing" i slutet av en given sträng, om strängen är kortare än 3 tecken ska den lämnas ofärndrad.
    # Expempel-input: "Python" Förväntad output: "Pythoning"
    input_string = input("Write a string: ")
    if len(input_string) < 3:
        result_string = input_string
    else:
        result_string = input_string + "ing"
    print(result_string)


def task_6():
    # Skriv ett program som först tar bort all whitespace (mellanslag, tab (\t), newline(\n)), och sedan även tar bort alla tecken på ojämna indexvärden, från given sträng.
    # Exempel-input: "a string with spaces and a newline character\n" Förväntad output: "atigihpcsnaelncaatr"
    input_string = "a string with spaces and a newline character\n" # input("Write a string: ")
    result_string_a = input_string.strip(" ") .replace(" ", "")
    result_string_b = ""
    for i in range(len(result_string_a)):
        if i % 2 == 0:
            result_string_b += result_string_a[i]

    print(result_string_b)


def task_7():
    # Skriv ett program som tar en komma-separerad sekvens av ord och skriver ut de unika orden i alfabetisk ordning.
    # Exempel-input: "red, white, black, red, green, black" Förväntad output: "black, green, red, white"
    input_string = "red, white, black, red, green,black"  # input("Write a string: ")
    result_list = input_string.replace(" ", "").split(",")
    result_set = set(result_list)
    result_list = list(result_set)
    result_list.sort()
    print(result_list)


def task_8():
    # Skriv en funktion som konverterar en given sträng till versaler (uppercase) om den innehåller minst 2 versaler bland de 4 första tecknen.
    input_string = "HelHaaaaH"  # input("Write a string: ")

    count = 0
    for char in input_string[:4]:
        if char.isupper():
            count += 1

    if count >= 2:
        result_string = input_string.upper()
    else:
        result_string = input_string
    print(result_string)


def task_9():
    # Skriv en funktion som vänder (reverse) på en sträng om dess längd är en multipel av 4.
    input_string = "abcdefghijkl"  # input("Write a string: ")

    count = 0
    for char in input_string[:4]:
        if char.isupper():
            count += 1

    if len(input_string) % 4 == 0:
        result_string = input_string[::-1]
    else:
        result_string = input_string
    print(result_string)


def task_10():
    # Skriv en funktion som skapar en ny sträng bestående av 4 kopior av de två sista tecken i en given sträng.
    # Exempel-input: "Python" Förväntad output: "onononon"
    input_string = "Python"  # input("Write a string: ")
    result_string = ""

    for i in range(4):
        result_string = result_string + input_string[-2:]

    print(result_string)


def task_11():
    # Skriv en funktion som tar emot en lista med ord och returnerar det längsta ordet samt dess längd
    input_list = ["aaaa", "bb", "ccccccc", "ddd"]
    index = -1
    max_len = 0
    result = ""
    for i in range(len(input_list)):
        if len(input_list[i]) > max_len:
            index = i
            max_len = len(input_list[i])
    print(input_list[index])

    max_len = 0
    for i in input_list:
        if len(i) > max_len:
            result = i
            max_len = len(i)

    print(result)


def task_12():
    # Skriv ett program som genererar en enkel multiplikationsmodell för tal 1-10.
    # Hur snyggt kan du få tabellen? Läs på om sträng-formattering i Python.
    for i in range(1, 10):
        for j in range(1, 10):
            print(f" {i * j}")


    # formatted_string = ", ".join(my_list)


def task_13():
    # Skriv en funktion som beräknar fakulteten av ett givet tal
    input_number = 13
    result = 1

    for i in range(1, input_number + 1):
        result = result * i

    print(f"Factorial of {input_number} is: {result}")


def task_14():
    # Skapa ett enkelt gissningsspel där datorn väljer ett slumpmässigt tal mellan 1-100 (eller annat intervall), och låt användaren gissa tills de hittar rätt nummer. För varje felaktig gissning berättar datorn om det rätta svaret är högre eller lägre än spelarens gissning.
    pass


def task_15():
    # Skriv ett program som kontrollerar om ett givet ord är ett palindrom (läses likadant framifrån som bakifrån).
    input_string = "abccba"
    if input_string == input_string[::-1]:
        print("This is palindrom")
    else:
        print("Not palindrom")


def task_16():
    # Skriv ett python program som itererar mellan 1 och 50,
    # om talet är delbart med 3 printar den "fizz"
    # om talet är delbart med 5 printar den "buzz",
    # om talet är delbart med både 3 och 5 så printar den "FizzBuzz"
    # annars printar den bara ut talet
    random_int = random.randint(0, 5)
    print(random_int)


def task_17():
    class Vehicle(object):
        def __init__(self, color):
            self.color = color

    class Car(Vehicle):
        def __init__(self):
            # self.color = "Blue"
            super().__init__("Red")
            pass

    car = Car()
    print(car.color)


def task_18():
    pass


if __name__ == '__main__':
    # task_1()
    # task_2()
    # task_3()
    # task_4()
    # task_5()
    # task_6()
    # task_7()
    # task_8()
    # task_9()
    task_10()
    task_11()
    task_12()
    task_13()
    task_14()
    task_15()
    task_16()
    task_17()
