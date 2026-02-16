import '../../core/layer.dart';
import '../../core/module.dart';
import '../../core/tensor.dart';
import 'aft_cross_attention2.dart';

class MultiHeadAFTCross extends Module {
  final List<AFTCrossAttention> heads;
  final Layer proj;
  final int numHeads;
  final int headSize;

  MultiHeadAFTCross(
    int numHeads,
    int decoderEmbedSize,
    int encoderEmbedSize,
    int maxSeqLen,
  ) : this.numHeads = numHeads,
      this.headSize = decoderEmbedSize ~/ numHeads,
      heads = List.generate(
        numHeads,
        (i) => AFTCrossAttention(
          decoderEmbedSize,
          decoderEmbedSize ~/ numHeads,
          encoderEmbedSize,
          maxSeqLen,
        ),
      ),
      proj = Layer(decoderEmbedSize, decoderEmbedSize, useGelu: false);

  Tensor forward(Tensor xDec, Tensor xEnc) {
    final headOutputs = heads.map((h) => h.forward(xDec, xEnc)).toList();
    final concatenated = Tensor.concat(headOutputs, axis: 1);
    return proj.forward(concatenated);
  }

  @override
  List<Tensor> parameters() => [
    ...heads.expand((h) => h.parameters()),
    ...proj.parameters(),
  ];
}
