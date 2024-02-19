#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/LU>
#include <fstream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class Rosenbrock
{
public:
  Rosenbrock(VectorXd data) : input_data_(data)
  {
    cout << "Input data loaded" << endl;
  }
  // part of Rosenbrock func
  double Fx(double x1, double x2)
  {
    return 100 * pow(x1 * x1 - x2, 2) + pow(x1 - 1, 2);
  }
  // derivative of func with respect to x1
  double DotFxX1(double x1, double x2) { return -2 * (1 - x1) + 200 * (x2 - x1 * x1) * (-2 * x1); }
  // derivative of func with respect to x2
  double DotFxx2(double x1, double x2) { return -200 * (x1 * x1 - x2); }

  /**
   * @brief calulate the output value of Rosenbrock function
   * @param {VectorXd} input
   * @return {*}
   */
  double Calculate(VectorXd input)
  {
    if (input.size() % 2 != 0)
    {
      cout << "wrong data size!";
      return -1;
    }
    double sum = 0;
    for (int i = 0; i < input.size(); i = i + 2)
    {
      sum += Fx(input(i), input(i + 1));
    }
    return sum;
  }

  /**
   * @brief calculate the gradient of Rosenbrock function
   * @param {VectorXd} input
   * @return {*}
   */
  VectorXd Gradient(VectorXd input)
  {
    VectorXd direct = VectorXd::Zero(input.size());
    for (int i = 0; i < input.size(); i = i + 2)
    {
      direct(i) = DotFxX1(input[i], input[i + 1]);
      direct(i + 1) = DotFxx2(input[i], input[i + 1]);
    }
    return direct;
  }

private:
  VectorXd input_data_;
};