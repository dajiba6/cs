#include <iostream>
#include <vector>
#include <cmath>
#include "hw01.h"

using namespace std;

int main()
{
  double input;
  vector<double> numbers;
  char continueInput;
  do
  {
    std::cout << "push one number to input: ";
    std::cin >> input;
    numbers.push_back(input);
    std::cout << "continue pushing?(y/n): ";
    std::cin >> continueInput;

  } while (continueInput == 'y' || continueInput == 'Y');

  Rosenbrock r(numbers);

  double res = r.Calculate(numbers);
  cout << "Init result: " << res << endl;
  r.Minimization();
}