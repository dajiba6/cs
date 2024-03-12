class Money_cont:
    def __init__(self, init_money, income_per_month, rate_invest_return):
        self.init_money = init_money
        self.income_per_month = income_per_month
        self.rate_invest_return = rate_invest_return

        pass

    def HowMuchYouWant(self, total_amout):
        current_money = self.init_money
        month_count = 0
        while current_money < total_amout:
            month_count += 1
            current_money += (
                current_money * self.rate_invest_return + self.income_per_month
            )
        print(
            f"Target Money: {total_amout}\nTime Required: {month_count} months = {month_count/12} years\n"
        )

    def HowMuchYouGet(self, total_month):
        month_count = 0
        current_money = self.init_money
        while month_count < total_month:
            month_count += 1
            current_money += (
                current_money * self.rate_invest_return + self.income_per_month
            )
        print(
            f"Target Months: {total_month}\nMoney you get: {current_money} = {round(current_money/1e4,3)} W\n"
        )


def main():
    print(f"Welcome my friend! Let's see how much you get!")
    init_money = float(input("How much money you have? "))
    income_per_month = float(input("How much money you make per month? "))
    rate_invest_return = float(input("What's the rate of your investment return? "))
    print("")
    target_money = float(input("How much money you want? "))
    target_month = float(input("How many months you want to save? "))

    mymoney = Money_cont(init_money, income_per_month, rate_invest_return)
    print("============= $ ===============")
    mymoney.HowMuchYouWant(target_money)
    mymoney.HowMuchYouGet(target_month)


if __name__ == "__main__":
    main()
