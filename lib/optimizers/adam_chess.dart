import 'dart:math' as math;
import '../core/tensor.dart';

class Adam {
  final List<Tensor> parameters;
  final double lr;
  final double beta1;
  final double beta2;
  final double epsilon;
  final double clipValue; // Prevent exploding gradients

  // State buffers for moments
  final List<List<double>> m;
  final List<List<double>> v;
  int t = 0;

  Adam(
    this.parameters, {
    this.lr = 0.0001, // Lower learning rate is usually safer for MuZero
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.epsilon = 1e-8,
    this.clipValue = 1.0, // Standard clipping threshold
  }) : m = List.generate(
         parameters.length,
         (i) => List.filled(parameters[i].data.length, 0.0),
       ),
       v = List.generate(
         parameters.length,
         (i) => List.filled(parameters[i].data.length, 0.0),
       );

  void step() {
    t++;
    final double bc1 = 1.0 - math.pow(beta1, t);
    final double bc2 = 1.0 - math.pow(beta2, t);

    for (int i = 0; i < parameters.length; i++) {
      final p = parameters[i];

      // Perform global gradient clipping for this parameter tensor
      _clipGradients(p.grad);

      for (int j = 0; j < p.data.length; j++) {
        final g = p.grad[j];

        // 1. Update biased first moment estimate
        m[i][j] = beta1 * m[i][j] + (1.0 - beta1) * g;

        // 2. Update biased second raw moment estimate
        v[i][j] = beta2 * v[i][j] + (1.0 - beta2) * (g * g);

        // 3. Compute bias-corrected estimates
        final mHat = m[i][j] / bc1;
        final vHat = v[i][j] / bc2;

        // 4. Update parameter with Adam rule
        // The lr is modulated by the signal-to-noise ratio (mHat/sqrt(vHat))
        p.data[j] -= lr * mHat / (math.sqrt(vHat) + epsilon);
      }
    }
  }

  /// Clips gradients to prevent numerical instability during RL
  void _clipGradients(List<double> grad) {
    double norm = 0.0;
    for (var g in grad) norm += g * g;
    norm = math.sqrt(norm);

    if (norm > clipValue) {
      final double scale = clipValue / (norm + 1e-6);
      for (int i = 0; i < grad.length; i++) {
        grad[i] *= scale;
      }
    }
  }

  void zeroGrad() {
    for (var p in parameters) {
      p.grad.fillRange(0, p.grad.length, 0.0);
    }
  }
}
