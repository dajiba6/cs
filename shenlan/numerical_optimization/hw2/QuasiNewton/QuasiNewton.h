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

typedef VectorXd (*gradient)(VectorXd input_data);
typedef double (*func)(VectorXd input_data);

class QuasiNewton
{
public:
  QuasiNewton(func cost_func, gradient gradient, VectorXd input_data) : target_cost_function_(cost_func),
                                                                        traget_gradient_function_(gradient),
                                                                        init_x_(input_data),
                                                                        lbfgs_limit_(30),
                                                                        lbfgs_alpha_vec(lbfgs_limit_, 0.0)
  {
    cout << "Initialization complete." << endl;
  }

  /**
   * @brief use backtracking line search to find an appropriate t
   * @return {*}
   */
  double LinsearchWeakWolfe(VectorXd direction, VectorXd x)
  {
    double cost_value = target_cost_function_(x);
    VectorXd gradient_value = traget_gradient_function_(x);
    double c1 = 0.5;
    double c2 = 0.5;
    double alpha = 1.0;
    while (cost_value - target_cost_function_(x + alpha * direction) >=
               c1 * alpha * direction.transpose() * gradient_value &&
           direction.transpose() * traget_gradient_function_(x + alpha * direction) >=
               c2 * direction.transpose() * gradient_value)
    {
      alpha *= 0.5;
    }
    return alpha;
  }

  /**
   * @brief use L-BFGS method to find B
   * @return {*}
   */
  MatrixXd LBFGS(VectorXd gradient)
  {
    VectorXd d = gradient;
    for (int i = 0; i < lbfgs_limit_; i++)
    {
      lbfgs_alpha_vec[i] = lbfgs_rho_vec[i] * lbfgs_y_vec[i].transpose() * lbfgs_s_vec[i];
      d = d - lbfgs_alpha_vec[i] * lbfgs_y_vec[i];
    }
    double gamma = lbfgs_rho_vec[lbfgs_limit_ - 2] *
                   lbfgs_y_vec[lbfgs_limit_ - 2].transpose() * lbfgs_y_vec[lbfgs_limit_ - 2];
    d = d / gamma;
    double beta = 0;
    for (int i = 0; i < lbfgs_limit_; i++)
    {
      beta = lbfgs_rho_vec[i] * lbfgs_y_vec[i].transpose() * d;
      d = d + lbfgs_s_vec[i] * (lbfgs_alpha_vec[i] - beta);
    }
    return d;
  }

  /**
   * @brief find the minimum value
   * @return {VectorXd} x
   */
  void Minimization()
  {
    VectorXd x = init_x_;
    VectorXd g;
    MatrixXd B;
    int count;
    VectorXd result;

    while (g.cwiseAbs().maxCoeff() > 1e-6)
    {
      VectorXd direction = -B * g;
      double t = LinsearchWeakWolfe(direction, x);
      VectorXd new_x = x + t * direction;
      VectorXd new_g = traget_gradient_function_(new_x);

      // 存入历史数据供LBFGS计算
      lbfgs_s_vec.push_back(new_x - x);
      lbfgs_y_vec.push_back(new_g - g);
      lbfgs_rho_vec
          .push_back(1.0 / ((new_x - x).transpose() * (new_g - g)));
      if (lbfgs_s_vec.size() > lbfgs_limit_)
      {
        lbfgs_s_vec.erase(lbfgs_s_vec.begin());
        lbfgs_y_vec.erase(lbfgs_y_vec.begin());
        lbfgs_rho_vec.erase(lbfgs_rho_vec.begin());
      }

      B = LBFGS(new_g);
      x = new_x;
      g = new_g;
      count++;
    }
    final_value_ = target_cost_function_(x);
    final_x = x;
  }

private:
  VectorXd init_x_;
  func target_cost_function_;
  gradient traget_gradient_function_;
  double final_value_;
  VectorXd final_x;
  int lbfgs_limit_ = 30;
  vector<VectorXd> lbfgs_s_vec;
  vector<VectorXd> lbfgs_y_vec;
  vector<double> lbfgs_rho_vec;
  vector<double> lbfgs_alpha_vec;
};
