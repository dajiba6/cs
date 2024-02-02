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

private:
  double count_ = 0;
  VectorXd data_;
  int dataLength_;
  double tau_ = 10;
  double c_ = 0.5;
  double gradient_limit_ = 10e-5;
  // 0:gradiant, 1:newton
  int mode_ = 1;

public:
  vector<vector<double>> plot_gradient_;

  Rosenbrock(VectorXd data, int mode) : data_(data), dataLength_(data.size()), mode_(mode)
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

  double Calculate(VectorXd input)
  {
    if (input.size() % 2 != 0)
    {
      cout << "wrong data size!";
      return -1;
    }
    double sum = 0;
    for (int i = 0; i < dataLength_; i = i + 2)
    {
      sum += Fx(input(i), input(i + 1));
    }
    return sum;
  }

  VectorXd Gradient(VectorXd input)
  {
    VectorXd direct = VectorXd::Zero(input.size());
    for (int i = 0; i < dataLength_; i = i + 2)
    {
      direct(i) = DotFxX1(input[i], input[i + 1]);
      direct(i + 1) = DotFxx2(input[i], input[i + 1]);
    }
    return direct;
  }

  // hessian
  double DotFxx1x1(double x1, double x2)
  {
    return 1200 * x1 * x1 - 400 * x2 + 2;
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

  MatrixXd Hessian(VectorXd &input)
  {
    MatrixXd direct = Eigen::MatrixXd::Zero(input.size(), input.size());
    for (int i = 0; i < dataLength_; i += 2)
    {
      double a =
          DotFxx1x1(input[i], input[i + 1]);
      double b =
          DotFxx1x2(input[i], input[i + 1]);
      double c =
          DotFxx2x1(input[i], input[i + 1]);
      double d =
          DotFxx2x2(input[i], input[i + 1]);
      MatrixXd temp(2, 2);
      temp << a, b, c, d;
      direct.block(i, i, 2, 2) = temp;
    }
    // cout << "hessian:" << endl;
    // cout << direct << endl;
    return direct;
  }

  VectorXd NewtonMethod(VectorXd &input, VectorXd gradient)
  {
    MatrixXd hessian = Hessian(input);
    // print if hessian is PD
    if (gradient.transpose() * hessian.inverse() * gradient < 0)
      cout << "hessian is not PD:"
           << gradient.transpose() * hessian.inverse() * gradient
           << endl;

    return -hessian.inverse() * gradient;
  }

  VectorXd UpdateInput_armijo(VectorXd &input, VectorXd &gradient)
  {
    VectorXd direction = VectorXd::Zero(dataLength_);
    gradient = Gradient(input);
    VectorXd new_input = VectorXd::Zero(input.size());
    double temp_res = 0;
    double cal_in = Calculate(input);
    for (int i = 0; i < input.size(); i++)
    {
      temp_res += -gradient[i] * gradient[i];
    }

    switch (mode_)
    {
    case 0:
      direction = -gradient;
      break;
    case 1:
      direction = NewtonMethod(input, gradient);
      break;
    default:
      break;
    }

    double tau = tau_;
    int temp11 = 0;
    double con1 = 0;
    double con2 = 0;
    do
    {
      temp11++;
      tau = tau * 0.5;
      // cout << "gradient:\n"
      //      << gradient << endl;
      VectorXd d = tau * direction;

      new_input = input + d;

      // 打开一个输出文件流
      std::ofstream outputFile("output.txt", std::ios::app);

      // 将梯度信息写入文件
      if (outputFile.is_open())
      {
        for (int i = 0; i < dataLength_; i++)
        {
          outputFile << new_input(i) << " ";
        }
        outputFile << "\n";
      }
      else
      {
        std::cerr << "无法打开输出文件。\n";
      }

      // 关闭文件流
      outputFile.close();

      con1 = Calculate(new_input);
      con2 = cal_in + (c_ * tau * temp_res);
      // cout << "test:" << cal_in + c_ * tau * temp_res << endl;
      // if (con2 != con1)
      //   cout << temp11 << "| new:" << Calculate(new_input) << " ;old:" << cal_in << endl;
    } while (con1 > con2);
    cout << "gradient:\n"
         << gradient << endl;

    return new_input;
  }

  bool Minimization()
  {

    VectorXd input = data_;
    double res = Calculate(input);
    VectorXd gradient = Gradient(input);
    int times = 0;
    while ((gradient.array().abs() > gradient_limit_).any())
    {

      input = UpdateInput_armijo(input, gradient);

      res = Calculate(input);
      // cout << "No.:" << times << "  out: " << res << endl;
      times++;
    }

    cout << "Last input:";
    for (int i = 0; i < dataLength_; i++)
    {
      cout << input(i) << " ";
    }
    cout << endl;
    cout << "Last gradient:";
    for (int i = 0; i < dataLength_; i++)
    {
      cout << gradient(i) << " ";
    }
    cout << endl;
    cout << "==================================" << endl;
    cout << "After " << times << " iteration, result = " << res << endl;
    return true;
  }
};