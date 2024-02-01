#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include "hw01.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/LU>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
int main()
{
  cout << "Enter number to select mode, 0:Steepest graident; 1:Newton's method:";
  int mode;
  cin >> mode;
  cin.ignore();
  double input;
  string inputstring;
  cout << "Enter numbers separated by space:";
  getline(cin, inputstring);
  istringstream iss(inputstring);
  vector<double> numbersIn;
  while (iss >> input)
  {
    numbersIn.push_back(input);
  }

  Eigen::Map<VectorXd> numbers(numbersIn.data(), numbersIn.size());
  Rosenbrock r(numbers, mode);

  double res = r.Calculate(numbers);
  cout << "Init result: " << res << endl;
  r.Minimization();
}