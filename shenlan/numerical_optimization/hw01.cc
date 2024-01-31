#include <iostream>
#include <vector>
#include <cmath>
#include "hw01.h"

using namespace std;

int main()
{

  vector<double> test = {1, 2, 3, 4};
  Rosenbrock r(test);

  double res = r.Calculate(test);
  cout << "Init result: " << res << endl;
  r.Minimization(10);
}