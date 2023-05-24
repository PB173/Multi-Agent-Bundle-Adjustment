#include <ceres/ceres.h>
#include <pybind11/pybind11.h>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

namespace py = pybind11;


// Example
struct ExampleFunctor {
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = T(10.0) - x[0];
    return true;
  }

  static ceres::CostFunction* Create() {
    return new ceres::AutoDiffCostFunction<ExampleFunctor, 1, 1>(
        new ExampleFunctor);
  }
};

// --------------------- Custom cost functions ------------------------------------------

// BA2D 
struct BA2DReprojectionError {
  BA2DReprojectionError(double observation)
      : observation(observation) {}
  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // initiate 2x1 vector 
    T p[2]; 
    // Transform point to camera system
    p[0] += point[0]-camera[0];
    p[1] += point[1]-camera[1];
    // Compute the center of distortion.
    T xp = p[0] / p[1];

    // Compute final projected point position.
    T predicted = xp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted - observation;
    return true;
  }

  static ceres::CostFunction* Create(const double observation) {
    return (new ceres::AutoDiffCostFunction<BA2DReprojectionError, 1, 2, 2>(
        new BA2DReprojectionError(observation)));
  }
  double observation;
};

// Simple reprojection error
struct ReprojectionError_FullBA {
  // Compute the reprojection error between two observations and a 3D point 
  ReprojectionError_FullBA(double observed_x, double observed_y, double fx, double fy, double cx, double cy)
      : observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy) {}
  template <typename T>
  bool operator()(const T* const q,
                  const T* const cam_translation,
                  const T* const point,
                  T* residuals) const {
           
    // Create inverse of transformation matrix  
    
    	// Initialize elements of quaternion q    	
    T qw = q[0];
    T qx = q[1];
    T qy = q[2];
    T qz = q[3];
    
        // Print input values, intermediate results and residuals for debugging
    
    /*
    std::cout << "q: " << qw << "  " << qx << "  " << qy << "  " << qz << std::endl;
    std::cout << "  " << std::endl; */
    
        // Normalize the quaternion
    T q_norm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    qw = qw / q_norm;
    qx = qx / q_norm;
    qy = qy / q_norm;
    qz = qz / q_norm;
    
    /*
    std::cout << "q_norm: " << qw << "  " << qx << "  " << qy << "  " << qz << std::endl;
    std::cout << "  " << std::endl; */
    
    	// Convert quaternion to rotation matrix R
    T R0 = T(1.0) - T(2.0)*(qy * qy + qz * qz);
    T R1 = T(2.0) * (qx * qy - qw * qz);
    T R2 = T(2.0) * (qx * qz + qw * qy);
    
    T R3 = T(2.0) * (qx * qy + qw * qz);
    T R4 = T(1.0) - T(2.0)*(qx * qx + qz * qz);
    T R5 = T(2.0) * (qy * qz - qw * qx);
    
    T R6 = T(2.0) * (qx * qz - qw * qy);
    T R7 = T(2.0) * (qy * qz + qw * qx);
    T R8 = T(1.0) - T(2.0)*(qx * qx + qy * qy);

        		
    	// Initialize elements of translation vector t
    T t0 = cam_translation[0];
    T t1 = cam_translation[1];
    T t2 = cam_translation[2];
    
    	// Calculate elements for transformation matrix T
    T xx = t0*R0 + t1*R3 + t2*R6;
    T yy = t0*R1 + t1*R4 + t2*R7;
    T zz = t0*R2 + t1*R5 + t2*R8;
    
    // Reproject point by multiplying with inverse transformation matrix
    T X_point = point[0];
    T Y_point = point[1];
    T Z_point = point[2];
    
    T X_reproj = R0*X_point + R3*Y_point + R6*Z_point - xx;
    T Y_reproj = R1*X_point + R4*Y_point + R7*Z_point - yy;
    T Z_reproj = R2*X_point + R5*Y_point + R8*Z_point - zz;

    // Project the 3D point onto the image plane (divide by Z component)
    T X_image = X_reproj / Z_reproj;
    T Y_image = Y_reproj / Z_reproj;
    
    // Multiply with the intrisic matrix to obtain Pixel Coordinates [Xp, Yp]
    T Xp = fx*X_image + cx;
    T Yp = fy*Y_image + cy;

    // The error is the difference between the point in Pixel Coordinates and the observation.
    residuals[0] = Xp - observed_x;
    residuals[1] = Yp - observed_y;
    
    
    // Print input values, intermediate results and residuals for debugging
    
    /*std::cout << "R:" << R0 << "  " << R1 << "  " << R2 << std::endl;
    std::cout << "R:" << R3 << "  " << R4 << "  " << R5 << std::endl;
    std::cout << "R:" << R6 << "  " << R7 << "  " << R8 << std::endl;
    std::cout << "q:" << qx << "  " << qy << "  " << qz << "  " << qw << std::endl;
    std::cout << "t:" << t0 << "  " << t1 << "  " << t2 << std::endl;
    std::cout << "P:" << X_point << "  " << Y_point << "  " << Z_point << std::endl;
    std::cout << "f:" << fx << "  " << fy << std::endl;
    std::cout << "c:" << cx << "  " << cy << std::endl;
    
    std::cout << "  " << std::endl;
    std::cout << "Intermediate results:" << std::endl;
    std::cout << "  " << std::endl;
    
    std::cout << "xx:" << xx << std::endl;
    std::cout << "yy:" << yy << std::endl;
    std::cout << "zz:" << zz << std::endl;
    std::cout << "  " << std::endl; 
    std::cout << "X_image:" << X_image << std::endl;
    std::cout << "Y_image:" << Y_image << std::endl;
    std::cout << "  " << std::endl; 
    std::cout << "Xp:" << Xp << std::endl;
    std::cout << "Yp:" << Yp << std::endl;
    
    std::cout << "  " << std::endl;    
    std::cout << "End results:" << std::endl;
    std::cout << "  " << std::endl;    
    
    std::cout << "Observations:" << observed_x<< " " << observed_y << std::endl;
    std::cout << "Residuals x:" << residuals[0] << std::endl;
    std::cout << "Residuals y:" << residuals[1] << std::endl;
    std::cout << "  " << std::endl; */
    
    return true;
  }
  
  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y,
                                     const double fx,
                                     const double fy,
                                     const double cx,
                                     const double cy) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError_FullBA, 2, 4, 3, 3>(
        new ReprojectionError_FullBA(observed_x, observed_y, fx, fy, cx, cy)));
  }
  double observed_x;
  double observed_y;
  double fx;
  double fy;
  double cx;
  double cy;
};


// Simple reprojection error motion only
struct ReprojectionError_MotionOnly {
  // Compute the reprojection error between two observations and a 3D point 
  ReprojectionError_MotionOnly(double observed_x, double observed_y, double X_point, double Y_point, double Z_point)
      : observed_x(observed_x), observed_y(observed_y), X_point(X_point), Y_point(Y_point), Z_point(Z_point) {}
  template <typename T>
  bool operator()(const T* const q,
                  const T* const cam_translation,
                  const T* const cam_parameters,
                  T* residuals) const {
        
    // cam_parameters[0,1] represent the focal distances of the camera
    T fx = cam_parameters[0];
    T fy = cam_parameters[1];
    
    // cam_parameters[2,3] represent the camera's principal point
    T cx = cam_parameters[2];
    T cy = cam_parameters[3];
    
    // Create inverse of transformation matrix  
    	// Initialize elements of quaternion q    	
    T qw = q[0];
    T qx = q[1];
    T qy = q[2];
    T qz = q[3];
  
        // Normalize the quaternion
    T q_norm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    qw = qw / q_norm;
    qx = qx / q_norm;
    qy = qy / q_norm;
    qz = qz / q_norm;
    
    	// Convert quaternion to rotation matrix R
    T R0 = T(1.0) - T(2.0)*(qy * qy + qz * qz);
    T R1 = T(2.0) * (qx * qy - qw * qz);
    T R2 = T(2.0) * (qx * qz + qw * qy);
    
    T R3 = T(2.0) * (qx * qy + qw * qz);
    T R4 = T(1.0) - T(2.0)*(qx * qx + qz * qz);
    T R5 = T(2.0) * (qy * qz - qw * qx);
    
    T R6 = T(2.0) * (qx * qz - qw * qy);
    T R7 = T(2.0) * (qy * qz + qw * qx);
    T R8 = T(1.0) - T(2.0)*(qx * qx + qy * qy);
        		
    	// Initialize elements of translation vector t
    T t0 = cam_translation[0];
    T t1 = cam_translation[1];
    T t2 = cam_translation[2];
    
    	// Calculate elements for transformation matrix T
    T xx = t0*R0 + t1*R3 + t2*R6;
    T yy = t0*R1 + t1*R4 + t2*R7;
    T zz = t0*R2 + t1*R5 + t2*R8;
    
    // Reproject point by multiplying with inverse transformation matrix 
    T X_reproj = R0*X_point + R3*Y_point + R6*Z_point - xx;
    T Y_reproj = R1*X_point + R4*Y_point + R7*Z_point - yy;
    T Z_reproj = R2*X_point + R5*Y_point + R8*Z_point - zz;

    // Project the 3D point onto the image plane (divide by Z component)
    T X_image = X_reproj / Z_reproj;
    T Y_image = Y_reproj / Z_reproj;
    
    // Multiply with the intrisic matrix to obtain Pixel Coordinates [Xp, Yp]
    T Xp = fx*X_image + cx;
    T Yp = fy*Y_image + cy;

    // The error is the difference between the point in Pixel Coordinates and the observation.
    residuals[0] = Xp - observed_x;
    residuals[1] = Yp - observed_y;
     
    return true;
  }
  
  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y,
                                     const double X_point,
                                     const double Y_point,
                                     const double Z_point) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError_MotionOnly, 2, 4, 3, 4>(
        new ReprojectionError_MotionOnly(observed_x, observed_y, X_point, Y_point, Z_point)));
  }
  double observed_x;
  double observed_y;
  double X_point;
  double Y_point;
  double Z_point;
};

// Reprojection error motion only
struct ReprojectionError_MotionOnly_fixed {
  // Compute the reprojection error between two observations and a 3D point 
  ReprojectionError_MotionOnly_fixed(double observed_x, double observed_y, double X_point, double Y_point, double Z_point, double fx, double fy, double cx, double cy)
      : observed_x(observed_x), observed_y(observed_y), X_point(X_point), Y_point(Y_point), Z_point(Z_point), fx(fx), fy(fy), cx(cx), cy(cy) {}
  template <typename T>
  bool operator()(const T* const q,
                  const T* const cam_translation,
                  T* residuals) const {
            
    // Create inverse of transformation matrix  
    	// Initialize elements of quaternion q    	
    T qw = q[0];
    T qx = q[1];
    T qy = q[2];
    T qz = q[3];
    
        // Print input values, intermediate results and residuals for debugging
    
    
        // Normalize the quaternion
    T q_norm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    qw = qw / q_norm;
    qx = qx / q_norm;
    qy = qy / q_norm;
    qz = qz / q_norm;
    
   
    	// Convert quaternion to rotation matrix R
    T R0 = T(1.0) - T(2.0)*(qy * qy + qz * qz);
    T R1 = T(2.0) * (qx * qy - qw * qz);
    T R2 = T(2.0) * (qx * qz + qw * qy);
    
    T R3 = T(2.0) * (qx * qy + qw * qz);
    T R4 = T(1.0) - T(2.0)*(qx * qx + qz * qz);
    T R5 = T(2.0) * (qy * qz - qw * qx);
    
    T R6 = T(2.0) * (qx * qz - qw * qy);
    T R7 = T(2.0) * (qy * qz + qw * qx);
    T R8 = T(1.0) - T(2.0)*(qx * qx + qy * qy);
        		
    	// Initialize elements of translation vector t
    T t0 = cam_translation[0];
    T t1 = cam_translation[1];
    T t2 = cam_translation[2];
    
    	// Calculate elements for transformation matrix T
    T xx = t0*R0 + t1*R3 + t2*R6;
    T yy = t0*R1 + t1*R4 + t2*R7;
    T zz = t0*R2 + t1*R5 + t2*R8;
    
    // Reproject point by multiplying with inverse transformation matrix 
    T X_reproj = R0*X_point + R3*Y_point + R6*Z_point - xx;
    T Y_reproj = R1*X_point + R4*Y_point + R7*Z_point - yy;
    T Z_reproj = R2*X_point + R5*Y_point + R8*Z_point - zz;

    // Project the 3D point onto the image plane (divide by Z component)
    T X_image = X_reproj / Z_reproj;
    T Y_image = Y_reproj / Z_reproj;
    
    // Multiply with the intrisic matrix to obtain Pixel Coordinates [Xp, Yp]
    T Xp = fx*X_image + cx;
    T Yp = fy*Y_image + cy;

    // The error is the difference between the point in Pixel Coordinates and the observation.
    residuals[0] = Xp - observed_x;
    residuals[1] = Yp - observed_y;
     
    return true;
  }
  
  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y,
                                     const double X_point,
                                     const double Y_point,
                                     const double Z_point,
                                     const double fx,
                                     const double fy,
                                     const double cx,
                                     const double cy) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError_MotionOnly_fixed, 2, 4, 3>(
        new ReprojectionError_MotionOnly_fixed(observed_x, observed_y, X_point, Y_point, Z_point, fx, fy, cx, cy)));
  }
  double observed_x;
  double observed_y;
  double X_point;
  double Y_point;
  double Z_point;
  double fx;
  double fy;
  double cx;
  double cy;
};

// Simple reprojection error points only
struct ReprojectionError_PointsOnly {
  // Compute the reprojection error between two observations and a 3D point 
  ReprojectionError_PointsOnly(double observed_x, double observed_y, double fx, double fy, double cx, double cy, double t0, double t1, double t2, double qw, double qx, double qy, double qz)
      : observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy), t0(t0), t1(t1), t2(t2), qw(qw), qx(qx), qy(qy), qz(qz) {}
  template <typename T>
  bool operator()(const T* const point,
                  T* residuals) const {
           
    // Create inverse of transformation matrix  
    
    	// Initialize elements of quaternion q    	

    
        // Print input values, intermediate results and residuals for debugging
    
    /*
    std::cout << "q: " << qw << "  " << qx << "  " << qy << "  " << qz << std::endl;
    std::cout << "  " << std::endl; */
    
        // Normalize the quaternion
    /*T q_norm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    qw = qw / q_norm;
    qx = qx / q_norm;
    qy = qy / q_norm;
    qz = qz / q_norm;*/
    
    /*
    std::cout << "q_norm: " << qw << "  " << qx << "  " << qy << "  " << qz << std::endl;
    std::cout << "  " << std::endl; */
    
    	// Convert quaternion to rotation matrix R
    T R0 = T(1.0) - T(2.0)*(qy * qy + qz * qz);
    T R1 = T(2.0) * (qx * qy - qw * qz);
    T R2 = T(2.0) * (qx * qz + qw * qy);
    
    T R3 = T(2.0) * (qx * qy + qw * qz);
    T R4 = T(1.0) - T(2.0)*(qx * qx + qz * qz);
    T R5 = T(2.0) * (qy * qz - qw * qx);
    
    T R6 = T(2.0) * (qx * qz - qw * qy);
    T R7 = T(2.0) * (qy * qz + qw * qx);
    T R8 = T(1.0) - T(2.0)*(qx * qx + qy * qy);

        		
  
    	// Calculate elements for transformation matrix T
    T xx = t0*R0 + t1*R3 + t2*R6;
    T yy = t0*R1 + t1*R4 + t2*R7;
    T zz = t0*R2 + t1*R5 + t2*R8;
    
    // Reproject point by multiplying with inverse transformation matrix
    T X_point = point[0];
    T Y_point = point[1];
    T Z_point = point[2];
    
    T X_reproj = R0*X_point + R3*Y_point + R6*Z_point - xx;
    T Y_reproj = R1*X_point + R4*Y_point + R7*Z_point - yy;
    T Z_reproj = R2*X_point + R5*Y_point + R8*Z_point - zz;

    // Project the 3D point onto the image plane (divide by Z component)
    T X_image = X_reproj / Z_reproj;
    T Y_image = Y_reproj / Z_reproj;
    
    // Multiply with the intrisic matrix to obtain Pixel Coordinates [Xp, Yp]
    T Xp = fx*X_image + cx;
    T Yp = fy*Y_image + cy;

    // The error is the difference between the point in Pixel Coordinates and the observation.
    residuals[0] = Xp - observed_x;
    residuals[1] = Yp - observed_y;
    
    
    // Print input values, intermediate results and residuals for debugging
    
    /*std::cout << "R:" << R0 << "  " << R1 << "  " << R2 << std::endl;
    std::cout << "R:" << R3 << "  " << R4 << "  " << R5 << std::endl;
    std::cout << "R:" << R6 << "  " << R7 << "  " << R8 << std::endl;
    std::cout << "q:" << qx << "  " << qy << "  " << qz << "  " << qw << std::endl;
    std::cout << "t:" << t0 << "  " << t1 << "  " << t2 << std::endl;
    std::cout << "P:" << X_point << "  " << Y_point << "  " << Z_point << std::endl;
    std::cout << "f:" << fx << "  " << fy << std::endl;
    std::cout << "c:" << cx << "  " << cy << std::endl;
    
    std::cout << "  " << std::endl;
    std::cout << "Intermediate results:" << std::endl;
    std::cout << "  " << std::endl;
    
    std::cout << "xx:" << xx << std::endl;
    std::cout << "yy:" << yy << std::endl;
    std::cout << "zz:" << zz << std::endl;
    std::cout << "  " << std::endl; 
    std::cout << "X_image:" << X_image << std::endl;
    std::cout << "Y_image:" << Y_image << std::endl;
    std::cout << "  " << std::endl; 
    std::cout << "Xp:" << Xp << std::endl;
    std::cout << "Yp:" << Yp << std::endl;
    
    std::cout << "  " << std::endl;    
    std::cout << "End results:" << std::endl;
    std::cout << "  " << std::endl;    
    
    std::cout << "Observations:" << observed_x<< " " << observed_y << std::endl;
    std::cout << "Residuals x:" << residuals[0] << std::endl;
    std::cout << "Residuals y:" << residuals[1] << std::endl;
    std::cout << "  " << std::endl; */
    
    return true;
  }
  
  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y,
                                     const double fx,
                                     const double fy,
                                     const double cx,
                                     const double cy,
                                     const double t0,
                                     const double t1,
                                     const double t2,
                                     const double qw,
                                     const double qx,
                                     const double qy,
                                     const double qz) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError_PointsOnly, 2, 3>(
        new ReprojectionError_PointsOnly(observed_x, observed_y, fx, fy, cx, cy, t0, t1, t2, qw, qx, qy, qz)));
  }
  double observed_x;
  double observed_y;
  double fx;
  double fy;
  double cx;
  double cy;
  double t0;
  double t1;
  double t2;
  double qw;
  double qx;
  double qy;
  double qz;
  
};


// Simple reprojection error_alternative
struct SimpleReprojectionError_alternative {
  // Compute the reprojection error between two observations and a 3D point 
  SimpleReprojectionError_alternative(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}
  template <typename T>
  bool operator()(const T* const T_inv,
                  const T* const point,
                  const T* const cam_parameters,
                  T* residuals) const {
        
    // cam_parameters[0,1] represent the focal distances of the camera
    T fx = cam_parameters[0];
    T fy = cam_parameters[1];
    
    // cam_parameters[2,3] represent the camera's principal point
    T cx = cam_parameters[2];
    T cy = cam_parameters[3];
    
    // Reproject point by multiplying with inverse transformation matrix
    T X_point = point[0];
    T Y_point = point[1];
    T Z_point = point[2];
    
    T X_reproj = T_inv[0]*X_point + T_inv[1]*Y_point + T_inv[2]*Z_point - T_inv[3];
    T Y_reproj = T_inv[4]*X_point + T_inv[5]*Y_point + T_inv[6]*Z_point - T_inv[7];
    T Z_reproj = T_inv[8]*X_point + T_inv[9]*Y_point + T_inv[10]*Z_point - T_inv[11];

    // Project the 3D point onto the image plane (divide by Z component)
    T X_image = X_reproj / Z_reproj;
    T Y_image = Y_reproj / Z_reproj;
    
    // Multiply with the intrisic matrix to obtain Pixel Coordinates [Xp, Yp]
    T Xp = fx*X_image + cx;
    T Yp = fy*Y_image + cy;

    // The error is the difference between the point in Pixel Coordinates and the observation.
    residuals[0] = Xp - observed_x;
    residuals[1] = Yp - observed_y;
    
    return true;
  }
  
  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SimpleReprojectionError_alternative, 2, 12, 3, 4>(
        new SimpleReprojectionError_alternative(observed_x, observed_y)));
  }
  double observed_x;
  double observed_y;
};


// Point Cloud Registration
struct PCR_Cost {
  // Compute the residual between the transformed point of the source point cloud and its counterpart of the target pointcloud
  PCR_Cost(double X_target, double Y_target, double Z_target, double X_source, double Y_source, double Z_source)
      : X_target(X_target), Y_target(Y_target), Z_target(Z_target), X_source(X_source), Y_source(Y_source), Z_source(Z_source) {}
  template <typename T>
  bool operator()(const T* const scale,
                  const T* const q,
                  const T* const t,
                  T* residuals) const {
        
    // Calculate new coordinates of source point cloud (src)
    	// Initialize elements of quaternion q    	
    T qw = q[0];
    T qx = q[1];
    T qy = q[2];
    T qz = q[3];
    
        // Print input values, intermediate results and residuals for debugging
    
    /*
    std::cout << "q: " << qw << "  " << qx << "  " << qy << "  " << qz << std::endl;
    std::cout << "  " << std::endl; */
    
        // Normalize the quaternion
    T q_norm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    qw = qw / q_norm;
    qx = qx / q_norm;
    qy = qy / q_norm;
    qz = qz / q_norm;
    
    /*
    std::cout << "q_norm: " << qw << "  " << qx << "  " << qy << "  " << qz << std::endl;
    std::cout << "  " << std::endl; */
    
    	// Convert quaternion to rotation matrix R
    T R0 = T(1.0) - T(2.0)*(qy * qy + qz * qz);
    T R1 = T(2.0) * (qx * qy - qw * qz);
    T R2 = T(2.0) * (qx * qz + qw * qy);
    
    T R3 = T(2.0) * (qx * qy + qw * qz);
    T R4 = T(1.0) - T(2.0)*(qx * qx + qz * qz);
    T R5 = T(2.0) * (qy * qz - qw * qx);
    
    T R6 = T(2.0) * (qx * qz - qw * qy);
    T R7 = T(2.0) * (qy * qz + qw * qx);
    T R8 = T(1.0) - T(2.0)*(qx * qx + qy * qy);
    
    /*  
    std::cout << "R: " << R0 << "  " << R1 << "  " << R2 << std::endl;
    std::cout << "R: " << R3 << "  " << R4 << "  " << R5 << std::endl;
    std::cout << "R: " << R6 << "  " << R7 << "  " << R8 << std::endl;
    std::cout << "  " << std::endl; */
      		
    	// Initialize elements of translation vector t
    T tx = t[0];
    T ty = t[1];
    T tz = t[2];
    
    /*
    std::cout << "t: " << tx << "  " << ty << "  " << tz << std::endl;
    std::cout << "  " << std::endl; */
    
    	// Initalize scale factor
    T sx = scale[0];
    T sy = scale[1];
    T sz = scale[2];
    
    /*
    std::cout << "scale:" << sx << "  " << sy << "  " << sz << std::endl;
    std::cout << "  " << std::endl; */
    
    	// Calculate new coordinates of source point cloud 
    T src_x = sx*(R0*X_source + R1*Y_source + R2*Z_source + tx);
    T src_y = sy*(R3*X_source + R4*Y_source + R5*Z_source + ty);
    T src_z = sz*(R6*X_source + R7*Y_source + R8*Z_source + tz);
    
    /*
    std::cout << "src:" << src_x << "  " << src_y << "  " << src_z << std::endl;
    std::cout << "  " << std::endl; */
    
    // The residual is the difference between the calculated coordinates [qx, qy, qz] and the target coordinates
    residuals[0] = src_x - X_target;
    residuals[1] = src_y - Y_target;
    residuals[2] = src_z - Z_target;
     
    return true;
  }
  
  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(const double X_target,
                                     const double Y_target,
                                     const double Z_target,
                                     const double X_source,
                                     const double Y_source,
                                     const double Z_source) {
    return (new ceres::AutoDiffCostFunction<PCR_Cost, 3, 3, 4, 3>(
        new PCR_Cost(X_target, Y_target, Z_target, X_source, Y_source, Z_source)));
  }
  double X_target;
  double Y_target;
  double Z_target;
  double X_source;
  double Y_source;
  double Z_source;
};

// ---------------------  Define cost functions for use with PyBinds --------------------- 
void add_custom_cost_functions(py::module& m) {
  // Use pybind11 code to wrap your own cost function which is defined in C++s

  // Here is an example
  m.def("CreateCustomExampleCostFunction", &ExampleFunctor::Create);
  m.def("BA2DCostFunction", &BA2DReprojectionError::Create);
  m.def("CostFunction_FullBA", &ReprojectionError_FullBA::Create);
  m.def("CostFunction_MotionOnly", &ReprojectionError_MotionOnly::Create);
  m.def("CostFunction_MotionOnly_fixed", &ReprojectionError_MotionOnly_fixed::Create);
  m.def("CostFunction_PointsOnly", &ReprojectionError_PointsOnly::Create);
  m.def("SimpleCostFunction_alternative", &SimpleReprojectionError_alternative::Create);
  m.def("PCR_CostFunction", &PCR_Cost::Create);
}
