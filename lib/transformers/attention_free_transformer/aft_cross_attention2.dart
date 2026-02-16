import 'dart:math' as math;
import '../../core/layer.dart';
import '../../core/module.dart';
import '../../core/tensor.dart';

class AFTCrossAttention extends Module {
  final Layer queryLayer, keyLayer, valueLayer;
  final int headSize;
  final Tensor posBias;
  final int maxSeqLen;

  AFTCrossAttention(
    int embedSize,
    int headSize,
    int encoderEmbedSize,
    this.maxSeqLen,
  ) : this.headSize = headSize,
      queryLayer = Layer(embedSize, headSize, useGelu: false),
      keyLayer = Layer(encoderEmbedSize, headSize, useGelu: false),
      valueLayer = Layer(encoderEmbedSize, headSize, useGelu: false),
      posBias = Tensor.random([maxSeqLen, maxSeqLen]);

  Tensor forward(Tensor x, Tensor encoderOutput) {
    final int T = x.shape[0];
    final int T_enc = encoderOutput.shape[0];
    final int d = headSize;

    final Q = queryLayer.forward(x).sigmoid();
    final K = keyLayer.forward(encoderOutput);
    final V = valueLayer.forward(encoderOutput);

    final out = Tensor.zeros([T, d]);

    for (int t = 0; t < T; t++) {
      Tensor numerator = Tensor.zeros([1, d]);
      Tensor denominator = Tensor.zeros([1, d]);

      // CLAMPING: Prevent index exceeding maxSeqLen
      int biasRow = (t < maxSeqLen) ? t : maxSeqLen - 1;

      for (int tp = 0; tp < T_enc; tp++) {
        int biasCol = (tp < maxSeqLen) ? tp : maxSeqLen - 1;

        // Safety check for the 256 RangeError
        double w_ttp = posBias.data[biasRow * maxSeqLen + biasCol];

        final kRow = K.getRow(tp);
        final vRow = V.getRow(tp);

        final expWeight = (kRow + w_ttp).exp();

        numerator = numerator + (expWeight * vRow);
        denominator = denominator + expWeight;
      }

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
