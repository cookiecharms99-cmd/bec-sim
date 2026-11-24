#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <selus/fft.hpp>
#include <selus/matrix.hpp>
#include <selus/vector.hpp>
using std::cout;

struct Hamiltonian {
  double g;
  selus::Vector<double> V;
  selus::Vector<double> T;
};

void timestep(selus::Vector<std::complex<double>> &psi, const Hamiltonian &H,
              double dt) {
  std::size_t n = psi.size();
  // Psi: evolve with exp(-it/2 * V) (IFFT) (exp -it * T) (FFT) exp(-it/2 *
  // V)
  // First the exp(-it/2 * V) step in x basis
  for (std::size_t i = 0; i < n; ++i) {
    psi[i] *= std::polar(1.0, -0.5 * dt * (H.V[i] + H.g * std::norm(psi[i])));
  }
  // convert to momentum and do the kinetic step
  selus::fft(psi.span());
  for (std::size_t i = 0; i < n; ++i) {
    psi[i] *= std::polar(1.0, -1 * dt * H.T[i]);
  }
  selus::ifft(psi.span());
  // back and do half potential step again :D
  for (std::size_t i = 0; i < n; ++i) {
    psi[i] *= std::polar(1.0, -0.5 * dt * (H.V[i] + H.g * std::norm(psi[i])));
  }
}

int main() {
  constexpr std::size_t N = 256;
  constexpr double L = 20.0; // Domain [-10, 10)
  constexpr double dx = L / static_cast<double>(N);
  constexpr double dk = 2.0 * std::numbers::pi / L;
  constexpr double omega = 1.0; // Harmonic trap frequency
  constexpr double g = 5.0;     // Repulsive interaction strength
  constexpr double dt = 0.005;
  constexpr std::size_t steps = 2000;

  selus::Vector<double> x(N);
  selus::Vector<double> V(N);
  selus::Vector<double> T(N);
  selus::Vector<std::complex<double>> psi(N);

  for (std::size_t j = 0; j < N; ++j) {
    // Spatial grid x_j in [-L/2, L/2)
    x[j] = -0.5 * L + static_cast<double>(j) * dx;
    V[j] = 0.5 * omega * omega * x[j] * x[j];

    // FFT momentum grid k_j (wrap-around ordering)
    const double mode = (j < N / 2)
                            ? static_cast<double>(j)
                            : static_cast<double>(j) - static_cast<double>(N);
    const double k = mode * dk;
    T[j] = 0.5 * k * k;

    // Initial displaced Gaussian wavepacket (so it oscillates/breathes in the
    // trap!)
    const double x0 = 1.5;
    psi[j] = std::exp(-0.5 * (x[j] - x0) * (x[j] - x0));
  }

  // Normalize psi so integral |psi(x)|^2 dx == 1.0
  const double initial_norm = std::sqrt(dx * psi.squared_norm());
  psi /= std::complex<double>(initial_norm, 0.0);

  const Hamiltonian H{g, std::move(V), std::move(T)};

  std::cout << std::fixed << std::setprecision(12);
  for (std::size_t step = 0; step <= steps; ++step) {
    if (step % 400 == 0) {
      const double total_prob = dx * psi.squared_norm();
      // Compute expectation value <x> = integral x * |psi(x)|^2 dx
      double mean_x = 0.0;
      for (std::size_t j = 0; j < N; ++j) {
        mean_x += x[j] * std::norm(psi[j]) * dx;
      }
      std::cout << "t = " << std::setw(6) << std::setprecision(2) << step * dt
                << " | norm = " << std::setprecision(12) << total_prob
                << " | <x> = " << std::setprecision(6) << mean_x << '\n';
    }
    timestep(psi, H, dt);
  }

  return 0;
}
