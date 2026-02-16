import '../../core/layer_norm.dart';
import '../../core/linear.dart';
import '../../core/module.dart';
import '../../core/tensor.dart';
import 'multi_head_attention.dart';

class TransformerEncoderBlock extends Module {
  final MultiHeadAttention attention;
  final Linear ffn;
  final LayerNorm ln1; // LayerNorm before attention
  final LayerNorm ln2; // LayerNorm before FFN

  TransformerEncoderBlock(int embedSize, int numHeads)
    : attention = MultiHeadAttention(
        embedSize,
        numHeads,
      ), // masked is false by default
      ffn = Linear(embedSize),
      ln1 = LayerNorm(embedSize),
      ln2 = LayerNorm(embedSize);

  Tensor forward(Tensor x) {
    // x shape: [T, embedSize]

    // 1. Multi-Head Self-Attention Sub-layer
    // x = x + Attention(LayerNorm(x))
    Tensor xNorm1 = ln1.forward(x);
    Tensor attnOut = attention.forward(xNorm1, masked: false);
    Tensor xRes1 = x + attnOut; // Vectorized residual addition

    // 2. Feed-Forward Sub-layer
    // x = x + FFN(LayerNorm(x))
    Tensor xNorm2 = ln2.forward(xRes1);
    Tensor ffnOut = ffn.forward(xNorm2);
    Tensor xRes2 = xRes1 + ffnOut;

    return xRes2;
  }

  @override
  List<Tensor> parameters() => [
    ...attention.parameters(),
    ...ffn.parameters(),
    ...ln1.parameters(),
    ...ln2.parameters(),
  ];
}
