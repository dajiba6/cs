#include "QuasiNewton.h"
#include "rosenbrck.h"

using namespace std;

int main()
{
  VectorXd x1(4);
  x1 << -3, -5, -100, 1;

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
  qua1.ResultInfo();
}