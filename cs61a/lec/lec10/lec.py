def pair(a, b):
    return 2**a * 3**b


def left(p):
    return multiplicity(2, p)


def right(p):
    return multiplicity(3, p)


def multiplicity(factor, n):
    if n % factor != 0:
        return 0
    return multiplicity(factor, n / 2) + 1


x = pair(2, 3)
print(left(x))
print(right(x))
