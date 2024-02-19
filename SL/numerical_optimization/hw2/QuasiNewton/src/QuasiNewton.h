#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/LU>
#include <fstream>
#include <deque>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class QuasiNewton
{
public:
  QuasiNewton(std::function<double(Eigen::VectorXd)> cost_func,
              std::function<Eigen::VectorXd(Eigen::VectorXd)> gradient,
              VectorXd input_data) : target_cost_function_(cost_func),
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
  // todo: lbfgs当数组只有1个数据时的处理方法
  VectorXd LBFGS(VectorXd gradient)
  {
    VectorXd d = gradient;
    int current_length = std::min(lbfgs_limit_, static_cast<int>(lbfgs_s_vec.size()));
    for (int i = 0; i < current_length; i++)
    {
      lbfgs_alpha_vec[i] = lbfgs_rho_vec[i] * lbfgs_y_vec[i].transpose() * lbfgs_s_vec[i];
      d = d - lbfgs_alpha_vec[i] * lbfgs_y_vec[i];
    }
    double gamma = lbfgs_rho_vec[current_length - 2] *
                   lbfgs_y_vec[current_length - 2].transpose() * lbfgs_y_vec[current_length - 2];
    d = d / gamma;
    double beta = 0;
    for (int i = 0; i < current_length; i++)
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
    int count = 0;
    VectorXd result;
    g = traget_gradient_function_(x);
    VectorXd direction = g;

    while (g.cwiseAbs().maxCoeff() > 1e-6)
    {
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
        lbfgs_s_vec.pop_front();
        lbfgs_y_vec.pop_front();
        lbfgs_rho_vec.pop_front();
      }

      direction = LBFGS(new_g);
      x = new_x;
      g = new_g;
      count++;
    }
    iteration_times_ = count;
    final_value_ = target_cost_function_(x);
    final_x = x;
  }

  void ResultInfo()
  {
    cout << "======== result =========" << endl;
    cout << "iteration times: " << iteration_times_ << endl;
    cout << "final_value_: " << final_value_ << endl;
    cout << "final_x:" << final_x << endl;
  }

  int lbfgs_limit_;

private:
  VectorXd init_x_;
  std::function<double(Eigen::VectorXd)> target_cost_function_;
  std::function<Eigen::VectorXd(Eigen::VectorXd)> traget_gradient_function_;
  double final_value_;
  VectorXd final_x;
  deque<VectorXd> lbfgs_s_vec;
  deque<VectorXd> lbfgs_y_vec;
  deque<double> lbfgs_rho_vec;
  vector<double> lbfgs_alpha_vec;
  int iteration_times_ = 0;
};
