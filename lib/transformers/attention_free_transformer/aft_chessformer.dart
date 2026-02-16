import '../../core/layer.dart';
import '../../core/layer_norm.dart';
import '../../core/module.dart';
import '../../core/tensor.dart';
import 'aft_transformer_decoder_block2.dart';
import 'dart:math' as math;

/// Standalone Tanh helper to replace missing math.tanh in Dart
double _mathTanh(double x) {
  if (x > 20) return 1.0;
  if (x < -20) return -1.0;
  double e2x = math.exp(2 * x);
  return (e2x - 1) / (e2x + 1);
}

class TransformerDecoder extends Module {
  final int vocabSize;
  final int embedSize;
  final int blockSize;
  final int numLayers;
  final int numHeads;
  final int encoderEmbedSize;

  // Embedding Tensors
  final Tensor wte; // Token Embeddings [vocabSize, embedSize]
  final Tensor wpe; // Position Embeddings [blockSize, embedSize]

  final List<TransformerDecoderBlock> blocks;
  final LayerNorm finalLayerNorm;

  // --- MuZero Dual Heads ---
  final Layer lmHead; // Policy Head (Logits)
  final Tensor vW1; // Value Head Hidden Layer [embedSize, 64]
  final Tensor vW2; // Value Head Output Layer [64, 1]

  TransformerDecoder({
    this.vocabSize = 4098,
    this.embedSize = 128,
    this.blockSize = 32,
    this.numLayers = 6,
    this.numHeads = 8,
    this.encoderEmbedSize = 128,
  }) : wte = Tensor.random([vocabSize, embedSize]),
       wpe = Tensor.random([blockSize, embedSize]),
       blocks = List.generate(
         numLayers,
         (i) => TransformerDecoderBlock(
           embedSize,
           numHeads,
           encoderEmbedSize,
           blockSize,
         ),
       ),
       finalLayerNorm = LayerNorm(embedSize),
       lmHead = Layer(embedSize, vocabSize, useGelu: false),
       vW1 = Tensor.random([embedSize, 64]),
       vW2 = Tensor.random([64, 1]) {
    // Scaling for numerical stability
    _initWeights(wte, 0.02);
    _initWeights(wpe, 0.02);
    _initWeights(vW1, math.sqrt(2.0 / embedSize));
    _initWeights(vW2, math.sqrt(2.0 / 64));
  }

  void _initWeights(Tensor t, double scale) {
    for (int i = 0; i < t.data.length; i++) t.data[i] *= scale;
  }

  /// Extracts the last hidden state to act as the "Latent State" s0
  Tensor getLatentState(List<int> idx, Tensor encoderOutput) {
    Tensor x = _embed(idx);
    for (final block in blocks) {
      x = block.forward(x, encoderOutput);
    }
    x = finalLayerNorm.forward(x);

    // Return only the last vector in the sequence [1, embedSize]
    return x.getRow(idx.length - 1);
  }

  /// Internal embedding logic with RangeError protection
  Tensor _embed(List<int> idx) {
    final int T = idx.length;
    Tensor x = Tensor.zeros([T, embedSize]);

    for (int t = 0; t < T; t++) {
      int tokenIdx = idx[t];

      // CRITICAL FIX: Boundary check to prevent RangeError
      if (tokenIdx < 0 || tokenIdx >= vocabSize) {
        // Fallback to index 4097 (your '.' or unknown token) if out of bounds
        tokenIdx = (vocabSize > 4097) ? 4097 : 0;
      }

      final tok_emb = wte.getRow(tokenIdx);

      // Position safety
      int posIdx = t < blockSize ? t : blockSize - 1;
      final pos_emb = wpe.getRow(posIdx);

      x.setRow(t, tok_emb + pos_emb);
    }
    return x;
  }

  /// MuZero Forward: Returns both Policy (Logits) and Value
  Map<String, Tensor> forwardMuZero(List<int> idx, Tensor encoderOutput) {
    Tensor hidden = getLatentState(idx, encoderOutput);

    // 1. Policy Head
    Tensor logits = lmHead.forward(hidden);

    // 2. Value Head (Hidden -> ReLU -> Linear -> Tanh)
    Tensor vHidden = hidden.matmul(vW1);
    for (int i = 0; i < vHidden.data.length; i++) {
      if (vHidden.data[i] < 0) vHidden.data[i] = 0; // ReLU
    }

    Tensor value = vHidden.matmul(vW2);
    for (int i = 0; i < value.data.length; i++) {
      value.data[i] = _mathTanh(value.data[i]);
    }

    return {"policy": logits, "value": value};
  }

  @override
  Tensor forward(List<int> idx, Tensor encoderOutput) {
    Tensor x = _embed(idx);
    for (final block in blocks) x = block.forward(x, encoderOutput);
    x = finalLayerNorm.forward(x);
    return lmHead.forward(x);
  }

  /// Projects a hidden state to policy logits
  Tensor policyHead(Tensor hiddenState) {
    return lmHead.forward(hiddenState);
  }

  @override
  List<Tensor> parameters() => [
    wte,
    wpe,
    vW1,
    vW2,
    ...blocks.expand((block) => block.parameters()),
    ...finalLayerNorm.parameters(),
    ...lmHead.parameters(),
  ];
}
