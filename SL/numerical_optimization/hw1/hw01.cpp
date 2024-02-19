#include <iostream>
#include <sstream>
#include <fstream>
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
  // // read input from user
  // cout << "Enter number to select mode, 0:Steepest graident; 1:Newton's method:";
  // int mode;
  // cin >> mode;
  // cin.ignore();
  // double input;
  // string inputstring;
  // cout << "Enter numbers separated by space:";
  // getline(cin, inputstring);
  // istringstream iss(inputstring);
  // vector<double> numbersIn;
  // while (iss >> input)
  // {
  //   numbersIn.push_back(input);
  // }

  //==================test=========================
  vector<double> numbersIn = {1.3, 0};
  int mode = 0;
  //================================================
  Eigen::Map<VectorXd>
      numbers(numbersIn.data(), numbersIn.size());
  Rosenbrock r(numbers, mode);

  //==================================================
  // 打开一个输出文件流
  std::ofstream outputFile("output.txt", std::ios::app);
  outputFile << 0 << " " << numbersIn[0] << " " << numbersIn[1] << endl;
  outputFile << 1 << " " << numbersIn[0] << " " << numbersIn[1] << endl;
  // 关闭文件流
  outputFile.close();
  //==================================================

  double res = r.Calculate(numbers);
  cout << "Init result: " << res << endl;
  r.tau_ = 4;
  r.Minimization();

  //==================test=======================
  Rosenbrock t(numbers, 1);
  t.tau_ = 4;
  t.c_ = 0.5;
  t.Minimization();
  //===========================================
}