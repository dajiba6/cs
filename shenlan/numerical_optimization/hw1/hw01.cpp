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
  double tau_ = 2;
  double c_ = 0.5;

public:
  Rosenbrock(vector<double> data) : data_(data), dataLength_(data.size())
  {
    if (dataLength_ % 2 != 0)
    {
      cout << "Wrong data size!";
    }
    else
      cout << dataLength_ << " dimension imput loaded." << endl;
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

    int times = 0;
    while (res > 0.001)
    {
      input = Armijo(input);
      res = Calculate(input);
      // cout << res << endl;
      // for (size_t i = 0; i < input.size(); i++)
      // {
      //   cout << input[i] << " ";
      // }
      // cout << endl;
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

  vector<double> Armijo(vector<double> &input)
  {

    vector<double> direct(input.size(), 0);
    vector<double> new_input(input.size(), 0);
    double temp_res = 0;
    double cal_in = Calculate(input);
    direct = Gradient(input);
    for (int i = 0; i < input.size(); i++)
    {
      temp_res += -direct[i] * direct[i];
    }
    double tau = tau_;
    do
    {
      tau = tau / 2;

      for (int i = 0; i < input.size(); i++)
      {
        new_input[i] = input[i] + (tau * -direct[i]);
      }
    } while (Calculate(new_input) > cal_in + (c_ * tau * temp_res));

    // for (size_t i = 0; i < input.size(); i++)
    // {
    //   cout << new_input[i] << " ";
    // }
    return new_input;
  }
};

int main()
{
  double input;
  vector<double> numbers;
  char continueInput;
  do
  {
    std::cout << "push one number to input: ";
    std::cin >> input;
    numbers.push_back(input);
    std::cout << "continue pushing?(y/n): ";
    std::cin >> continueInput;

  } while (continueInput == 'y' || continueInput == 'Y');

  Rosenbrock r(numbers);

  double res = r.Calculate(numbers);
  cout << "Init result: " << res << endl;
  r.Minimization();
}