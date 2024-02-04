#include "stdafx.h"
#include <iostream>
#include <Dense>
using namespace std;
using namespace Eigen;

MatrixXd newton(MatrixXd, MatrixXd, int, float, float);
MatrixXd least_squre(MatrixXd, MatrixXd);
int get_min_m(MatrixXd, MatrixXd, float, float, MatrixXd, MatrixXd, MatrixXd);
MatrixXd get_error(Matrix<double, -1, -1>, Matrix<double, -1, -1>, MatrixXd);
MatrixXd first_derivative(MatrixXd, MatrixXd, MatrixXd);
MatrixXd second_derivative(MatrixXd);
int main()
{
  MatrixXd feature(5, 3);
  feature << 1, 1, 1,
      1, 2, 3,
      1, 4, 1,
      1, 5, 6,
      1, 3, 7;
  cout << "特征：" << endl
       << feature << endl;
  MatrixXd label(5, 1);
  label << 7.6, 10.1, 10.5, 16.1, 15.4;
  cout << "标签：" << endl
       << label << endl;
  MatrixXd result = newton(feature, label, 50, 0.1, 0.5);
  cout << "牛顿法系数：" << endl
       << result << endl;
  MatrixXd result2 = least_squre(feature, label);
  cout << "最小二乘法系数：" << endl
       << result2 << endl;
  getchar();
  return 0;
}

/// 最小二乘
MatrixXd least_squre(MatrixXd feature, MatrixXd label)
{
  return (feature.transpose() * feature).inverse() * (feature.transpose()) * label;
}

// 牛顿法
MatrixXd newton(MatrixXd feature, MatrixXd label, int iterMax, float sigma, float delta)
{
  double epsilon = 0.1;
  int n = feature.cols();
  MatrixXd w = MatrixXd::Zero(n, 1);
  MatrixXd g;
  MatrixXd G;
  MatrixXd d;
  double m;
  int it = 0;
  while (it <= iterMax)
  {
    g = first_derivative(feature, label, w);
    if (epsilon >= g.norm())
    {
      break;
    }
    G = second_derivative(feature);
    d = -G.inverse() * g;
    m = get_min_m(feature, label, sigma, delta, d, w, g);
    w = w + pow(sigma, m) * d;
    it++;
  }
  return w;
}

// 获取最小m
int get_min_m(MatrixXd feature, MatrixXd label, float sigma, float delta, MatrixXd w, MatrixXd d, MatrixXd g)
{
  int m = 0;
  MatrixXd w_new;
  MatrixXd left;
  MatrixXd right;
  while (true)
  {
    w_new = w + pow(sigma, m) * d;
    left = get_error(feature, label, w_new);
    right = get_error(feature, label, w) + delta * pow(sigma, m) * g.transpose() * d;
    if (left(0, 0) <= right(0, 0))
    {
      break;
    }
    else
    {
      m += 1;
    }
  }
  return m;
}

// 计算误差
MatrixXd get_error(MatrixXd feature, MatrixXd label, MatrixXd w)
{
  return (label - feature * w).transpose() * (label - feature * w) / 2;
}

// 一阶导
MatrixXd first_derivative(MatrixXd feature, MatrixXd label, MatrixXd w)
{
  int m = feature.rows();
  int n = feature.cols();
  MatrixXd g = MatrixXd::Zero(n, 1);
  MatrixXd err;
  for (int i = 0; i < m; i++)
  {
    err = label.block(i, 0, 1, 1) - feature.row(i) * w;
    for (int j = 0; j < n; j++)
    {
      g.row(j) -= err * feature(i, j);
    }
  }
  return g;
}

// 二阶导
MatrixXd second_derivative(MatrixXd feature)
{
  int m = feature.rows();
  int n = feature.cols();
  MatrixXd G = MatrixXd::Zero(n, n);
  MatrixXd x_left;
  MatrixXd x_right;
  for (int i = 0; i < m; i++)
  {
    x_left = feature.row(i).transpose();
    x_right = feature.row(i);
    G += x_left * x_right;
  }
  return G;
}