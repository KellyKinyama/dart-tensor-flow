import 'dart:math' as math;
import '../mu_zero/model.dart';
import '../mu_zero/move_codec.dart';
import '../transformers/attention_free_transformer/aft_chessformer.dart';
import 'package:dart_tensor_flow/uitils/network_utils.dart';

void main() async {
  // 1. Setup Architecture
  final transformer = TransformerDecoder(
    vocabSize: 4098,
    embedSize: 128,
    numLayers: 6,
    blockSize: 16,
  );
  final model = MuZeroModel(transformer, 128);

  // 2. Load Weights
  const String checkpointPath = "muzero_chess_v1.json";
  try {
    await loadModuleParameters(transformer, checkpointPath);
    print("Weights successfully loaded from: $checkpointPath");
  } catch (e) {
    print("No valid checkpoint found. Testing with random weights.");
  }

  // 3. Play-Test Loop
  List<int> history = [0]; // Start token

  print("\n--- MuZero Policy Evaluation Test ---");
  print("Step | Rank | Move | Confidence | WinProb");
  print("-------------------------------------------");

  for (int i = 0; i < 5; i++) {
    // Inference
    final state = model.represent(history);
    final predictions = model.predict(state);

    final logits = predictions['policy']!.data;
    final value = predictions['value']!.data[0];

    // 1. Calculate Softmax Probabilities
    double maxLogit = logits.reduce(math.max);
    double sumExp = 0;
    for (var l in logits) {
      sumExp += math.exp(l - maxLogit);
    }

    List<Map<String, dynamic>> moveProbs = [];
    for (int j = 0; j < logits.length; j++) {
      double prob = math.exp(logits[j] - maxLogit) / sumExp;
      moveProbs.add({'idx': j, 'prob': prob});
    }

    // 2. Sort to find the Top 3 choices
    moveProbs.sort((a, b) => b['prob'].compareTo(a['prob']));

    // 3. Display Results
    double winProb = (value + 1) / 2 * 100;

    for (int rank = 0; rank < 3; rank++) {
      int idx = moveProbs[rank]['idx'];
      double conf = moveProbs[rank]['prob'] * 100;
      String uci = MoveCodec.indexToUci(idx);

      String row =
          "${" $i   |  #${rank + 1}  | ${uci.padRight(4)} | "
              "${conf.toStringAsFixed(3)}%".padRight(11)} | ${winProb.toStringAsFixed(1)}%";
      print(row);
    }
    print("-------------------------------------------");

    // Proceed with the #1 choice for the next turn
    int chosenIdx = moveProbs[0]['idx'];
    history.add(chosenIdx);

    // Safety: Sliding window for history
    if (history.length >= 16) history.removeAt(0);
  }
}
