#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * S.inverse();

  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

// Normalize an angle phi to the range -PI to PI
double normalizeAngle(double angle){
  while (angle < -M_PI){
    angle += 2*M_PI;
  }
  while (angle > M_PI){
    angle -= 2*M_PI;
  }

  return(angle);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
  float pos_x=x_(0);
  float pos_y=x_(1);
  float vel_x=x_(2);
  float vel_y=x_(3);

  // rho: polar distance
  // phi: polar direction (rads)
  // rho_dot: polar speed

  float rho = sqrt(pos_x*pos_x + pos_y*pos_y);
  //Avoid div by zero
  rho = ((fabs(rho) < 0.0000) ? 0.0001 : rho);

  float phi = atan2(pos_y, pos_x);
  float rho_dot = (pos_x*vel_x + pos_y*vel_y) / rho;
  VectorXd z_pred(3);
  z_pred << rho, phi, rho_dot;

  VectorXd y = z - z_pred;
  y(1) = normalizeAngle(y(1));
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ *  Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

}
