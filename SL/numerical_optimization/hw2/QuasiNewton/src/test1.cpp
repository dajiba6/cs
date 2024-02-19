#include "QuasiNewton.h"
#include "rosenbrck.h"

using namespace std;

int main()
{
  VectorXd x1(2);
  x1 << 0.2, 0.1;

  Rosenbrock targetFunc1(x1);
  auto costFunc = [&targetFunc1](Eigen::VectorXd input)
  {
    return targetFunc1.Calculate(input);
  };

  auto gradientFunc = [&targetFunc1](Eigen::VectorXd input)
  {
    return targetFunc1.Gradient(input);
  };

  QuasiNewton qua1(costFunc, gradientFunc, x1);
  qua1.Minimization();
}