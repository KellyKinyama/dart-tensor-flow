import '../../core/layer_norm.dart';
import '../../core/linear.dart';
import '../../core/module.dart';
import '../../core/tensor.dart';

import 'multi_head_attention.dart';
import 'multi_head_cross_attention.dart';

class TransformerDecoderBlock extends Module {
  final MultiHeadAttention selfAttention;
  final MultiHeadCrossAttention crossAttention;
  final Linear ffn;
  final LayerNorm ln1;
  final LayerNorm ln2;
  final LayerNorm ln3;

  TransformerDecoderBlock(int embedSize, int numHeads, int encoderEmbedSize)
    : selfAttention = MultiHeadAttention(embedSize, numHeads),
      crossAttention = MultiHeadCrossAttention(
        numHeads,
        embedSize,
        encoderEmbedSize,
      ),
      ffn = Linear(embedSize),
      ln1 = LayerNorm(embedSize),
      ln2 = LayerNorm(embedSize),
      ln3 = LayerNorm(embedSize);

  Tensor forward(Tensor xDecoder, Tensor xEncoder) {
    // 1. Masked Self-Attention (decoder looks at itself)
    // x = x + SelfAttn(LN(x))
    Tensor xNorm1 = ln1.forward(xDecoder);
    Tensor selfAttnOut = selfAttention.forward(xNorm1, masked: true);
    Tensor xRes1 = xDecoder + selfAttnOut;

    // 2. Cross-Attention (decoder looks at encoder)
    // x = x + CrossAttn(LN(x), encoder_out)
    Tensor xNorm2 = ln2.forward(xRes1);
    Tensor crossAttnOut = crossAttention.forward(xNorm2, xEncoder);
    Tensor xRes2 = xRes1 + crossAttnOut;

    // 3. Feed-Forward Network
    // x = x + FFN(LN(x))
    Tensor xNorm3 = ln3.forward(xRes2);
    Tensor ffnOut = ffn.forward(xNorm3);
    Tensor out = xRes2 + ffnOut;

    return out;
  }

  @override
  List<Tensor> parameters() => [
    ...selfAttention.parameters(),
    ...crossAttention.parameters(),
    ...ffn.parameters(),
    ...ln1.parameters(),
    ...ln2.parameters(),
    ...ln3.parameters(),
  ];
}
