import 'module.dart';
import 'tensor.dart';

class Layer extends Module {
  final Tensor weights;
  final Tensor bias;
  final bool useGelu;

  Layer(int nIn, int nOut, {this.useGelu = true})
    : weights = Tensor.xavier([nIn, nOut]),
      bias = Tensor.zeros([1, nOut]);

  Tensor forward(Tensor x) {
    // Standard Linear Transformation: xW + b
    Tensor out = x.matmul(weights) + bias;

    // Use GELU instead of ReLU to prevent "Dead Neurons"
    return useGelu ? out.gelu() : out;
  }

  @override
  List<Tensor> parameters() => [weights, bias];
}
