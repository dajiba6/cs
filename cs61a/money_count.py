def fuli2(start, goal, income, rate, count=0):
    print("cash :", start, "years :", count / 12)
    if start >= goal:
        return count
    return fuli2(start * rate + income, goal, income, rate, count + 1)


start = 10000
income = 6000
rate = 1.01
goal = 300000

print(
    "==================\n",
    "start = ",
    start,
    "\n",
    "months needed = ",
    fuli2(start, goal, income, rate),
    "\n",
    "years needed = ",
    fuli2(start, goal, income, rate) / 12,
)
