import random


class Hangman(object):
    def __init__(self, words):
        random_value = random.randint(0, len(words) - 1)
        self.word = words[random_value]
        self.matching_chars = []
        self.number_of_tries = 0
        self.number_of_errors = 0
        self.number_of_matching_chars = 0
        self._errors_until_fail = 5

    def print_output_string(self):
        # Denna funktion loopar över index i listan
        my_string = ""
        for i in range(len(self.word)):
            if self.word[i].upper() in self.matching_chars:
                my_string = my_string + self.word[i].upper()
            else:
                my_string = my_string + "_"
        print(my_string)

    def check_char_in_word(self, char):
        # Denna funktion loopar tecken för tecken i strängen
        if char.upper() not in self.matching_chars:
            upper_char = char.upper()
            for char in self.word:
                if char.upper() == upper_char:
                    self.matching_chars.append(upper_char)

    def is_valid_input(self, char):
        # Denna funktion validerar input
        valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ"
        upper_char = char.upper()
        if len(char) == 1 and upper_char in valid_chars:
            return True
        else:
            return False

    def add_char(self, input_char):
        if self.is_valid_input(input_char):
            self.number_of_tries += 1
            number_of_matching_chars = len(self.matching_chars)
            self.check_char_in_word(input_char)

            if len(self.matching_chars) > number_of_matching_chars:
                return True
            else:
                self.number_of_errors += 1
                return False

    def draw_man(self, full=False):
        errors = self.number_of_errors if full is False else self._errors_until_fail
        lines = list()
        lines.append("__________")
        lines.append("|/" if errors < 1 else "|/       |")
        lines.append("|" if errors < 2 else "|        0")
        lines.append("|" if errors < 3 else "|       \\|/")
        lines.append("|" if errors < 4 else "|        |")
        lines.append("|" if errors < 5 else "|       / \\")
        lines.append("|")

        for line in lines:
            print(line)

    def end_of_game_fail(self):
        return self.number_of_errors >= self._errors_until_fail

    def end_of_game_success(self):
        return len(self.matching_chars) == len(self.word)


def continue_playing():
    input_string = input("Om du vill spela igen skriv ja och i annat fall nej: ")
    if input_string.lower() == "ja":
        return True
    else:
        return False


def run():
    debug = False
    words = ["skärm", "lampa", "skrivbord", "stol", "dator", "tangentbord"]
    hangman = Hangman(words)
    hangman.draw_man(True)
    hint = f"({hangman.word})" if debug is True else ""

    print(f"Gissa ordet. Du får ange an bokstav i taget mellan A-Ö. Gemener eller versaler spelar ingen roll")
    print(f"Ordet har {len(hangman.word)} bokstäver {hint}")

    playing = True
    while playing:
        hangman.print_output_string()
        input_char = input("Ange en bokstav: ")

        if hangman.is_valid_input(input_char):
            if hangman.add_char(input_char):
                # print("Rätt")
                if hangman.end_of_game_success():
                    print(f"Grattis du lyckades på {hangman.number_of_tries} försök")
                    if continue_playing():
                        hangman = Hangman(words)
                        hangman.draw_man(True)
                    else:
                        playing = False
            else:
                # print("Fel gissa igen")
                hangman.draw_man()
                if hangman.end_of_game_fail():
                    print(f"Sorry, du misslyckades på {hangman.number_of_tries} försök")
                    if continue_playing():
                        hangman = Hangman(words)
                        hangman.draw_man(True)
                    else:
                        playing = False
                pass

        else:
            print("Ogiltig inmatning försök igen")


if __name__ == '__main__':
    run()
