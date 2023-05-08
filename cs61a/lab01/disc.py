def is_prime(n):
    """
    >>> is_prime(10)
    False
    >>> is_prime(7)
    True
    """
    "*** YOUR CODE HERE ***"
    k=1
    while k < n:
      r = n%k
      if r == 0:
        if k != 1:
          if k != n:
            return False
      k += 1
    return True

def fizzbuzz(n):
    """
    >>> result = fizzbuzz(16)
    1
    2
    fizz
    4
    buzz
    fizz
    7
    8
    fizz
    buzz
    11
    fizz
    13
    14
    fizzbuzz
    16
    >>> result == None
    True
    """

    k=1
    while k <= n:
      if k%3 == 0 and k%5 ==0:
        print('fizzbuzz')
      elif k%3 == 0:
         print('fizz')
      elif k%5 == 0:
         print("buzz")
      else:
         print(k)
      k += 1
