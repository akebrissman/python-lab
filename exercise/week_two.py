from functools import reduce


class Book(object):
    def __init__(self, title, author, year, genre, borrowed):
        self.title = title
        self.author = title
        self.author = author
        self.year = year
        self.genre = genre
        self.is_borrowed = borrowed

    def __str__(self):
        return f"{self.title} by {self.author} ({self.year}) Borrowed {self.is_borrowed}"

    def borrow(self):
        if not self.is_borrowed:
            self.is_borrowed = True
            return True
        else:
            return False

    def return_book(self):
        self.is_borrowed = False


class Library(object):
    def __init__(self):
        self.books = []

    def add_book(self, book):
        self.books.append(book)

    def remove_book(self, book):
        self.books.remove(book)

    def available_books(self):
        return list(filter(lambda book: not book.is_borrowed, self.books))

    def list_books(self):
        return list(map(str, self.books))

    def find_book(self, title):
        return next((book for book in self.books if book.title.lower() == title.lower()), None)

    def get_pages(self):
        return reduce (lambda x, y: x + y, map(lambda book: book.pages, self.books), 0)

    def group_by_genre(self):
        return {genre: list(books) for genre, books in
                itertools.groupby(sorted(self.books, key=lambda x: x.genre), key=lambda x: x.genre)}


library = Library()
library.add_book(Book("A", "B", 1950, "Fiction", False))
library.add_book(Book("C", "D", 1950, "Fiction", False))
a = library.find_book("C")
print(f"Hittad bok: {a}")
print(library.list_books())
list(map(print, library.list_books()))

# books = list()
# book = Book("A", "B", 1950, "Fiction", False)
# books.append(book)
# book = Book("C", "D", 1950, "Fiction", False)
# books.append(book)
#
# for book in books:
#     print(book)
#
# print(book)
# print(book.is_borrowed)
# print(book)
# book.borrow()
# print(book.is_borrowed)
# print(book)
# book.return_book()
# print(book.is_borrowed)
# print(book)
