#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;
class Rosenbrock
{

private:
  double count_ = 0;
  vector<double> data_;
  int dataLength_;
  double tau_ = 2;
  double c_ = 0.5;
  double gradient_limit_ = 10e-5;

public:
  Rosenbrock(vector<double> data) : data_(data), dataLength_(data.size())
  {
    if (dataLength_ % 2 != 0)
    {
      cout << "Wrong data size!";
    }
    else
    {
      cout << dataLength_ << " dimension imput loaded." << endl;
      cout << "gradient limit is " << gradient_limit_ << endl;
    }
  }

  double Fx(double x1, double x2)
  {
    return 100 * pow(x1 * x1 - x2, 2) + pow(x1 - 1, 2);
  }

  double DotFxX1(double x1, double x2) { return 100 * (4 * x1 * (x1 * x1 - x2)) + 2 * (x1 - 1); }

  double DotFxx2(double x1, double x2) { return -100 * (x1 * x1 - x2); }

  double Calculate(vector<double> input)
  {
    if (input.size() % 2 != 0)
    {
      cout << "wrong data size!";
      return -1;
    }
    double sum = 0;
    for (int i = 0; i < dataLength_; i = i + 2)
    {
      sum += Fx(input[i], input[i + 1]);
    }
    return sum;
  }

  vector<double> Gradient(vector<double> input)
  {
    vector<double> direct(input.size(), 0);
    for (int i = 0; i < dataLength_; i = i + 2)
    {
      direct[i] = DotFxX1(input[i], input[i + 1]);
      direct[i + 1] = DotFxx2(input[i], input[i + 1]);
    }
    return direct;
  }

  double Minimization()
  {

    vector<double> input = data_;
    double res = Calculate(input);
    vector<double> gradient = Gradient(input);
    int times = 0;
    while (any_of(gradient.begin(), gradient.end(), [this](double num)
                  { return num > gradient_limit_; }))
    {
      gradient = Gradient(input);
      input = Armijo(input, gradient);
      res = Calculate(input);
      times++;
    }

    cout << "After " << times << " iteration, result = " << res << endl;
    cout << "Last input:";
    for (auto num : input)
    {
      cout << num << " ";
    }
    cout << endl;

    return res;
  }

  vector<double> Armijo(vector<double> &input, vector<double> &gradient)
  {

    vector<double> direction(input.size(), 0);
    vector<double> new_input(input.size(), 0);
    double temp_res = 0;
    double cal_in = Calculate(input);
    direction = gradient;
    for (int i = 0; i < input.size(); i++)
    {
      temp_res += -direction[i] * direction[i];
    }
    double tau = tau_;
    do
    {
      tau = tau / 2;

      for (int i = 0; i < input.size(); i++)
      {
        new_input[i] = input[i] + (tau * -direction[i]);
      }
    } while (Calculate(new_input) > cal_in + (c_ * tau * temp_res));
    return new_input;
  }
};