#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
class Rosenbrock
{
public:
  Rosenbrock(int variableNumber) : variableNumber_(variableNumber)
  {
    if (variableNumber_ % 2 != 0)
    {
      variableNumber_++;
    }
    cout << variableNumber_ << " Parameters loaded." << endl;
  }

  double Fx(double x1, double x2)
  {
    return 100 * pow(x1 * x1 - x2, 2) + pow(x1 - 1, 2);
  }

  double DotFxX1(double x1, double x2)
  {
    return 100 * (4 * x1 * (x1 * x1 - x2)) + 2 * (x1 - 1);
  }

  double DotFxx2(double x1, double x2)
  {
    return -100 * (x1 * x1 - x2);
  }

  double Calculate(vector<double> input)
  {
    double sum = 0;
    for (int i = 0; i < variableNumber_ / 2; i = i + 2)
    {
      sum += Fx(input[i], input[i + 1]);
    }
    return sum;
  }

  vector<double> Gradient(vector<double> input)
  {
    vector<double> direct = {0, 0};
    for (int i = 0; i < variableNumber_ / 2; i = i + 2)
    {
      direct[0] += DotFxX1(input[i], input[i + 1]);
      direct[1] += DotFxx2(input[i], input[i + 1]);
    }
    return direct;
  }

  double Minimization(int times, vector<double> &input)
  {

    for (int i = 0; i < times; i++)
    {
      vector<double> direct = Gradient(input);
      for (int i = 0; i < variableNumber_; i = i + 2)
      {
        input[i] -= direct[0];
        input[i + 1] -= direct[1];
      }
    }
    double res = Calculate(input);
    cout << "After " << times << " iteration, result = " << res;
  }

  double length1(double length)
  {
    count_++;
    return length / count_;
  }

private:
  double count_ = 0;
  int variableNumber_ = 0;
};

int main()
{
  Rosenbrock r(4);

  double res = r.Calculate({100, 100, 100, 100});
  r.Minimization(10, )
          cout
      << res << endl;
}