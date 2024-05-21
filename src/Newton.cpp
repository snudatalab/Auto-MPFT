/*****************************************************************************
*    This is an implementation of Fast Multidimensional Partial              *
*    Fourier Transform with Automatic Hyperparameter Selection               *
*    (submitted to KDD 2024).                                                *
*                                                                            *
*    This code contains the optimzation algorithm for finding                *
*    the optimal hyperparameter of Auto-MPFT using Newton's method.          *
*****************************************************************************/

#include <iostream>
#include <cmath>


// Define constants
const double PI = 3.14159265358979323846;
const double ALPHA = 0.28513442812268959006;
const double EPSILON = 1e-7;

// Function p
double p_func(double r, double M) {
    double rf = std::tgamma(r + 1);
    return (M * PI / 2.0) * (std::pow(ALPHA * EPSILON * rf, -1.0 / r)) * (std::exp(-1.0 / (r * (r + 1)) * std::pow(ALPHA * EPSILON * rf, 2.0 / r)));
}

// Update step
double step(double r, double N, double M) {
    double h = 1e-6;
    double v_p = p_func(r, M);
    double forward = p_func(r + h, M);
    double backward = p_func(r - h, M);
    double v_pp = (forward - backward) / (2.0 * h);
    double v_pdp = (forward + backward - 2.0 * v_p) / (h * h);
    double lv_p = std::log(v_p);
    return ((N + 4.0 * v_p * lv_p + 4.0 * M) + 4.0 * r * v_pp * (1.0 + lv_p)) / (4.0 * (1.0 + lv_p) * (2.0 * v_pp + r * v_pdp) + 4.0 * r * v_pp * v_pp / v_p);
}

// Newton's method
double findMinimizer(double N, double M) {
    double r = 10.0; // Initial r
    double tolerance = 1e-1; 
    int max_iterations = 1000; 
    int iteration = 0;
    double delta_r;

    while (iteration < max_iterations) {
        delta_r = -step(r, N, M);
        r += delta_r;
        // Check convergence
        if (std::abs(delta_r) < tolerance) {
            return r;
        }
        iteration++;
    }
    std::cout << "Did not converge after " << max_iterations << " iterations." << std::endl;
    return std::numeric_limits<double>::quiet_NaN();
}

int main() {
    double N = 4096;
    double M = 128;
    double minimizer;
    double optimal_p;

    minimizer = findMinimizer(N, M);
    optimal_p = p_func(minimizer, M);
    std::cout << "Optimal parameter p = " << optimal_p << std::endl;
    std::cout << "Minimizer found at r = " << minimizer << std::endl;

    return 0;
}
