import 'dart:math' as math;
import '../core/tensor.dart';
import '../transformers/attention_free_transformer/aft_chessformer.dart';

double _mathTanh(double x) => (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1);

class MuZeroModel {
  final TransformerDecoder transformer;
  final int embedSize;

  MuZeroModel(this.transformer, this.embedSize);

  Tensor represent(List<int> observations) {
    return transformer.getLatentState(observations, Tensor.zeros([1, embedSize]));
  }

  Map<String, Tensor> dynamics(Tensor state, int action) {
    Tensor actionEmb = transformer.wte.getRow(action);
    // Move the hidden state forward through one transformer block
    Tensor nextState = transformer.blocks[0].forward(state + actionEmb, Tensor.zeros([1, embedSize]));
    
    // Simple reward head
    Tensor reward = nextState.matmul(transformer.vW1).matmul(transformer.vW2);
    reward.data[0] = _mathTanh(reward.data[0]);

    return {"state": nextState, "reward": reward};
  }

  Map<String, Tensor> predict(Tensor state) {
    Tensor logits = transformer.lmHead.forward(state);
    Tensor vHidden = state.matmul(transformer.vW1);
    for (int i = 0; i < vHidden.data.length; i++) if (vHidden.data[i] < 0) vHidden.data[i] = 0;
    Tensor value = vHidden.matmul(transformer.vW2);
    value.data[0] = _mathTanh(value.data[0]);

    return {"policy": logits, "value": value};
  }
}