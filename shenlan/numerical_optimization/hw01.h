#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
class Rosenbrock
{

private:
  double count_ = 0;
  vector<double> data_;
  int dataLength_;

public:
  Rosenbrock(vector<double> data) : data_(data), dataLength_(data.size())
  {
    if (dataLength_ % 2 != 0)
    {
      cout << "Wrong data size!";
    }
    else
      cout << dataLength_ << " Parameters loaded." << endl;
  }

  double Fx(double x1, double x2)
  {
    return 100 * pow(x1 * x1 - x2, 2) + pow(x1 - 1, 2);
  }

  double DotFxX1(double x1, double x2) { return 100 * (4 * x1 * (x1 * x1 - x2)) + 2 * (x1 - 1); }

  double DotFxx2(double x1, double x2) { return -100 * (x1 * x1 - x2); }

  double Calculate(vector<double> input)
  {
    double sum = 0;
    for (int i = 0; i < dataLength_ / 2; i = i + 2)
    {
      sum += Fx(input[i], input[i + 1]);
    }
    return sum;
  }

  vector<double> Gradient(vector<double> input)
  {
    vector<double> direct(input.size(), 0);
    for (int i = 0; i < dataLength_ / 2; i = i + 2)
    {
      direct[i] -= DotFxX1(input[i], input[i + 1]);
      direct[i + 1] -= DotFxx2(input[i], input[i + 1]);
    }
    return direct;
  }

  double Minimization(int times)
  {
    vector<double> input = data_;
    for (int i = 0; i < times; i++)
    {
      vector<double> direct = Gradient(input);
      double cos = direct[];
      double sin = 0;
      for (int i = 0; i < dataLength_; i = i + 2)
      {
        input[i] -= direct[0];
        input[i + 1] -= direct[1];
      }
    }
    double res = Calculate(input);
    cout << "After " << times << " iteration, result = " << res << endl;
    cout << "Last input:";
    for (auto num : input)
    {
      cout << num << " ";
    }
    cout << endl;

    return res;
  }

  double length1(double length)
  {
    count_++;
    return length / count_;
  }
};