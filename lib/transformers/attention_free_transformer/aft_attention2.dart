import 'dart:math' as math;
import '../../core/layer.dart';
import '../../core/module.dart';
import '../../core/tensor.dart';

class AFTAttention extends Module {
  final Layer queryLayer, keyLayer, valueLayer;
  final int headSize;
  final Tensor posBias;
  final bool masked;
  final int maxSeqLen; // Store this for safety checks

  AFTAttention(
    int embedSize,
    this.headSize,
    this.maxSeqLen, {
    this.masked = false,
  }) : queryLayer = Layer(embedSize, headSize, useGelu: false),
       keyLayer = Layer(embedSize, headSize, useGelu: false),
       valueLayer = Layer(embedSize, headSize, useGelu: false),
       posBias = Tensor.random([maxSeqLen, maxSeqLen]);

  Tensor forward(Tensor x) {
    final int T = x.shape[0];
    final int d = headSize;

    // 1. Projections
    final Q = queryLayer.forward(x).sigmoid();
    final K = keyLayer.forward(x);
    final V = valueLayer.forward(x);

    final out = Tensor.zeros([T, d]);

    // 2. AFT-full Loop
    for (int t = 0; t < T; t++) {
      Tensor numerator = Tensor.zeros([1, d]);
      Tensor denominator = Tensor.zeros([1, d]);

      int limit = masked ? t + 1 : T;

      // SAFETY: Ensure we don't exceed the posBias matrix dimensions
      int biasRow = t < maxSeqLen ? t : maxSeqLen - 1;

      for (int tp = 0; tp < limit; tp++) {
        // SAFETY: Ensure column index is also within bounds
        int biasCol = tp < maxSeqLen ? tp : maxSeqLen - 1;

        // Correct index calculation using maxSeqLen (shape[1])
        double w_ttp = posBias.data[biasRow * maxSeqLen + biasCol];

        // exp(K_tp + w_ttp)
        // Optimization: Get the rows once
        final kRow = K.getRow(tp);
        final vRow = V.getRow(tp);

        final expWeight = (kRow + w_ttp).exp();

        numerator = numerator + (expWeight * vRow);
        denominator = denominator + expWeight;
      }

      // Y_t = Q_t * (num / (den + epsilon))
      final row_t = Q.getRow(t) * (numerator / (denominator + 1e-9));
      out.setRow(t, row_t);
    }

    return out;
  }

  @override
  List<Tensor> parameters() => [
    ...queryLayer.parameters(),
    ...keyLayer.parameters(),
    ...valueLayer.parameters(),
    posBias,
  ];
}
