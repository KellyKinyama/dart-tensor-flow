// 1. Efficient Tensor-based SGD Optimizer
import '../core/tensor.dart';

class SGD {
  final List<Tensor> parameters;
  final double lr;

  SGD(this.parameters, this.lr);

  void step() {
    for (var p in parameters) {
      for (int i = 0; i < p.data.length; i++) {
        p.data[i] -= lr * p.grad[i];
      }
    }
  }

  void zeroGrad() {
    for (var p in parameters) {
      p.grad.fillRange(0, p.grad.length, 0.0);
    }
  }
}
