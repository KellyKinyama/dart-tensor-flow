import 'tensor.dart';

abstract class Module {
  List<Tensor> parameters();

  void zeroGrad() {
    for (var p in parameters()) {
      p.zeroGrad();
    }
  }
}
