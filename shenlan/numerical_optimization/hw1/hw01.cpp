#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include "hw01.h"
using namespace std;

int main()
{
  double input;
  vector<double> numbers;
  string inputstring;
  cout << "Enter numbers separated by space:";
  getline(cin, inputstring);
  istringstream iss(inputstring);

  while (iss >> input)
  {
    numbers.push_back(input);
  }

  Rosenbrock r(numbers);

  double res = r.Calculate(numbers);
  cout << "Init result: " << res << endl;
  r.Minimization();
}