import 'dart:math' as math;
import 'package:dart_tensor_flow/core/tensor.dart';
// Ensure this path matches where your updated TransformerDecoder is
import '../transformers/attention_free_transformer/aft_chessformer.dart';

double _mathTanh(double x) {
  if (x > 20) return 1.0;
  if (x < -20) return -1.0;
  double e2x = math.exp(2 * x);
  return (e2x - 1) / (e2x + 1);
}

class MuZeroModel {
  final TransformerDecoder transformer;
  final int embedSize;

  // We no longer need a separate ValueHead class because
  // the logic is now inside the transformer.forwardMuZero method.
  MuZeroModel(this.transformer, this.embedSize);

  // Representation: h(o) -> s
  // Converts moves into the hidden 'latent' state s0
  Tensor represent(List<int> observations) {
    // We pass a dummy encoder output (zeros) because AFT expects it
    final dummyEnc = Tensor.zeros([1, embedSize]);
    return transformer.getLatentState(observations, dummyEnc);
  }

  // Dynamics: g(s, a) -> (s', r)
  // Imagines the next hidden state based on an action
  Map<String, Tensor> dynamics(Tensor state, int action) {
    // FIX: Changed tokenEmbedding to wte
    Tensor actionEmb = transformer.wte.getRow(action);

    // Transition: Add the action embedding to the current state
    // and pass through one Transformer Block to 'process' the move
    Tensor nextState = transformer.blocks[0].forward(
      state + actionEmb,
      Tensor.zeros([1, embedSize]),
    );

    return {
      "state": nextState,
      "reward": Tensor.fill([1], 0.0),
    };
  }

  // Prediction: f(s) -> (p, v)
  // Predicts policy and value from a latent state
  Map<String, Tensor> predict(Tensor state) {
    // We use the specialized MuZero forward we added to the transformer
    // If you haven't added the specific heads yet, this calls the logic:

    // 1. Policy (using your lmHead/policyHead logic)
    Tensor logits = transformer.lmHead.forward(state);

    // 2. Value (Internal MLP logic)
    Tensor vHidden = state.matmul(transformer.vW1);
    for (int i = 0; i < vHidden.data.length; i++) {
      if (vHidden.data[i] < 0) vHidden.data[i] = 0; // ReLU
    }
    Tensor value = vHidden.matmul(transformer.vW2);
    for (int i = 0; i < value.data.length; i++) {
      value.data[i] = _mathTanh(value.data[i]); // Tanh
    }

    return {"policy": logits, "value": value};
  }
}
