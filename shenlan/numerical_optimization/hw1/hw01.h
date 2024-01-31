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
  // 0:gradiant, 1:newton
  int mode_ = 1;

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

  double DotFxx2(double x1, double x2) { return -200 * (x1 * x1 - x2); }

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

  // hessian
  double DotFxx1x1(double x1, double x2)
  {
    return 800 * x1 * x1 + 400 * x1 * x1 - 400 * x2 + 2;
  }
  double DotFxx1x2(double x1, double x2)
  {
    return -400 * x1;
  }
  double DotFxx2x1(double x1, double x2)
  {
    return -400 * x1;
  };
  double DotFxx2x2(double x1, double x2)
  {
    return 200;
  }
  vector<vector<double>> Hessian(vector<double> input)
  {
    vector<vector<double>> direct(input.size(), std::vector<double>(input.size(), 0.0));
    for (int i = 0; i < dataLength_; i++)
    {
      int j = (i % 2 == 0 ? i : i - 1);
      for (; j < 2; j++)
      {
        if (i % 2 == 0)
        {
          direct[i][j] = DotFxx1x1(input[i], input[i + 1]);
          direct[i][j + 1] = DotFxx1x2(input[i], input[i + 1]);
        }
        else if (i % 2 == 1)
        {
          direct[i][j] = DotFxx2x1(input[i], input[i + 1]);
          direct[i][j + 1] = DotFxx2x2(input[i], input[i + 1]);
        }
      }
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
      switch (mode_)
      {
      case 0:
        input = UpdateInput_Gradient(input, gradient);
        break;
      case 1:
        input = UpdateInput_Newton(input, gradient);
        break;
      default:
        input = UpdateInput_Gradient(input, gradient);
        break;
      }

      res = Calculate(input);
      times++;
    }

    cout << "Last input:";
    for (auto num : input)
    {
      cout << num << " ";
    }
    cout << endl;
    cout << "Last gradient:";
    for (auto num : gradient)
    {
      cout << num << " ";
    }
    cout << endl;
    cout << "==================================" << endl;
    cout << "After " << times << " iteration, result = " << res << endl;
    return res;
  }

  // todo: infinite loop
  vector<double> UpdateInput_Newton(vector<double> input, vector<double> &gradient)
  {
    gradient = Gradient(input);
    vector<double> new_input(input.size(), 0);
    double cal_in = Calculate(input);
    double temp_res = 0;
    vector<vector<double>> hessian = Hessian(input);
    vector<double> direction(dataLength_, 0);
    for (int i = 0; i < input.size(); i++)
    {
      temp_res += -gradient[i] * gradient[i];
    }

    for (int i = 0; i < dataLength_; i++)
    {
      double res = 0;
      for (int j = 0; j < dataLength_; j++)
      {
        res += hessian[i][j] * gradient[j];
      }
      direction[i] = res;
    }

    double tau = tau_;
    do
    {
      tau = tau / 2;
      for (int i = 0; i < input.size(); i++)
      {
        new_input[i] = input[i] + (tau * -direction[i]);
      }
      // test------------------
      cout << Calculate(new_input) << endl;
    } while (Calculate(new_input) > cal_in + (c_ * tau * temp_res));
    return new_input;
  }

  vector<double> UpdateInput_Gradient(vector<double> &input, vector<double> &gradient)
  {
    gradient = Gradient(input);
    vector<double> new_input(input.size(), 0);
    double temp_res = 0;
    double cal_in = Calculate(input);
    for (int i = 0; i < input.size(); i++)
    {
      temp_res += -gradient[i] * gradient[i];
    }

    double tau = tau_;
    do
    {
      tau = tau / 2;
      for (int i = 0; i < input.size(); i++)
      {
        new_input[i] = input[i] + (tau * -gradient[i]);
      }
    } while (Calculate(new_input) > cal_in + (c_ * tau * temp_res));
    cout << Calculate(new_input) << endl;

    return new_input;
  }
};