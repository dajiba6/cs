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
    double c1 = 1e-4;
    double c2 = 0.9;
    double alpha = 20;
    while (cost_value - target_cost_function_(x + alpha * direction) <
               -c1 * alpha * direction.transpose() * gradient_value ||
           direction.transpose() * traget_gradient_function_(x + alpha * direction) <
               c2 * direction.transpose() * gradient_value)
    {
      alpha *= 0.5;
      // cout << "cost: " << target_cost_function_(x + alpha * direction) << endl;
      bool con1 = (cost_value - target_cost_function_(x + alpha * direction) < -c1 * alpha * direction.transpose() * gradient_value);
      bool con2 = direction.transpose() * traget_gradient_function_(x + alpha * direction) <
                  c2 * direction.transpose() * gradient_value;
      // cout << "con1: " << con1 << " con2: " << con2 << endl;
    }

    //! 步长test
    // cout << "difference:" << cost_value - target_cost_function_(x + alpha * direction) << endl;
    // cout << "left       " << -c1 * alpha * direction.transpose() * gradient_value << endl;
    // cout << "gradient_value" << endl
    //      << gradient_value << endl;
    // cout << "direction" << endl
    //      << direction << endl;
    // cout << "alpha: " << alpha << endl;
    cout << "new cost: " << target_cost_function_(x + alpha * direction) << endl;
    // cout << "old cost: " << cost_value << endl;
    return alpha;
  }

  /**
   * @brief use L-BFGS method to find B
   * @return {*}
   */

  VectorXd LBFGS(VectorXd gradient)
  {
    VectorXd d = gradient;
    int current_length = std::min(lbfgs_limit_, static_cast<int>(lbfgs_s_vec.size()));
    for (int i = 0; i < current_length; i++)
    {
      lbfgs_alpha_vec[i] = lbfgs_rho_vec[i] * lbfgs_y_vec[i].transpose() * lbfgs_s_vec[i];
      d = d - lbfgs_alpha_vec[i] * lbfgs_y_vec[i];
    }
    double gamma = lbfgs_rho_vec[current_length - 1] *
                   lbfgs_y_vec[current_length - 1].transpose() * lbfgs_y_vec[current_length - 1];
    d = d / gamma;
    double beta = 0;
    for (int i = 0; i < current_length; i++)
    {
      beta = lbfgs_rho_vec[i] * lbfgs_y_vec[i].transpose() * d;
      d = d + lbfgs_s_vec[i] * (lbfgs_alpha_vec[i] - beta);
    }

    /* TODO LBFGS direction result wrong: lbfgs_s_vec goes 0 -nan -> lbfgs_rho_vec[current_length - 1] goes inf -nan -> gamma goes -nan -> d goes -nan
       x 错误移动导致 g 没改变 导致 rho求解爆炸
       第三次循环时x移动错误
       第三次移动由LBFGS决定方向，wolfe决定步长
       第三次步长移动了50，但x没改变多少，方向没有特别奇怪的数字
       步长中的 new cost 第二次和第四次数值无变化

    */

    // cout << "LBFGS_direction: " << endl
    //  << d << endl;
    // cout << "gamma: " << gamma << endl;
    // cout << "lbfgs_rho_vec[current_length - 1]: " << lbfgs_rho_vec[current_length - 1] << endl;
    // cout << "lbfgs_s_vec: " << lbfgs_s_vec[current_length - 1] << endl;
    // cout << "lbfgs_y_vec: " << lbfgs_y_vec[current_length - 1] << endl;
    // cout << "part of rho: " << lbfgs_s_vec[current_length - 1].transpose() * lbfgs_y_vec[current_length - 1] << endl;
    return d;
  }

  /**
   * @brief find the minimum value
   * @return {VectorXd} x
   */
  void
  Minimization()
  {
    VectorXd x = init_x_;
    VectorXd g;
    MatrixXd B;
    int count = 0;
    VectorXd result;
    g = traget_gradient_function_(x);
    VectorXd direction = -g;

    // 把数据输入到文件中画图
    std::ofstream outputFile("../src/output_file.txt");
    if (!outputFile.is_open())
    {
      std::cerr << "Error opening output file!" << std::endl;
      return;
    }

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

      // 把数据输入到文件中画图
      for (int i = 0; i < x.size(); ++i)
      {
        outputFile << x[i] << " ";
      }
      outputFile << "\n";

      direction = LBFGS(new_g);

      x = new_x;
      g = new_g;

      // cout << count << " iteration: " << target_cost_function_(x) << endl;
      count++;
    }
    // 把数据输入到文件中画图
    outputFile.close();

    iteration_times_ = count;
    final_value_ = target_cost_function_(x);
    final_x = x;
  }

  void ResultInfo()
  {
    cout << "======== result =========" << endl;
    cout << "iteration times: " << iteration_times_ << endl;
    cout << "final_value_: " << final_value_ << endl;
    cout << "final_x:" << endl
         << final_x << endl;
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
