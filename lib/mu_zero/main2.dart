import 'trainer3.dart';
import 'model2.dart';
import '../transformers/attention_free_transformer/aft_chessformer.dart';
import '../optimizers/adam_chess.dart';

void main() async {
  // 1. Initialize Architecture
  // vocabSize 4098 accounts for 64x64 moves + special tokens
  final transformer = TransformerDecoder(vocabSize: 4098, embedSize: 128);
  final model = MuZeroModel(transformer, 128);
  final optimizer = Adam(transformer.parameters());
  final trainer = MuZeroTrainer(model, optimizer);

  print("--- Starting True MuZero Training (MCTS-Based) ---");
  print("Parameters: EmbedSize=128, BufferLimit=${trainer.maxBufferSize}");

  double runningLoss = 0.0;

  for (int cycle = 1; cycle <= 100; cycle++) {
    // 2. Self-Play Phase
    // Note: MCTS makes this much slower but the data quality is 10x higher.
    // 'simulations' can be increased as your hardware allows.
    print("\n[Cycle $cycle] Generating games via MCTS Self-Play...");
    DateTime startSelfPlay = DateTime.now();

    await trainer.runSelfPlaySession(3, simulations: 40);

    Duration selfPlayDuration = DateTime.now().difference(startSelfPlay);

    // 3. Training Phase
    // We run multiple trainSteps per cycle to fully utilize the new data
    print(
      "[Cycle $cycle] Training on buffer (Size: ${trainer.replayBuffer.length})...",
    );
    double cycleLoss = 0;
    int stepsPerCycle = 5;

    for (int s = 0; s < stepsPerCycle; s++) {
      cycleLoss += trainer.trainStep(batchSize: 32);
    }

    double avgLoss = cycleLoss / stepsPerCycle;
    runningLoss = (cycle == 1) ? avgLoss : (runningLoss * 0.9 + avgLoss * 0.1);

    // 4. Logging
    print("------------------------------------------");
    print("Cycle $cycle Summary:");
    print("> Self-Play Time: ${selfPlayDuration.inSeconds}s");
    print("> Step Loss:      ${avgLoss.toStringAsFixed(5)}");
    print("> Smoothed Loss:  ${runningLoss.toStringAsFixed(5)}");
    print(
      "> Buffer Usage:   ${((trainer.replayBuffer.length / trainer.maxBufferSize) * 100).toStringAsFixed(1)}%",
    );
    print("------------------------------------------");

    // 5. Checkpointing
    if (cycle % 10 == 0) {
      _saveModelWeights(model, cycle);
    }
  }
}

/// Placeholder for weight persistence
void _saveModelWeights(MuZeroModel model, int cycle) {
  print("💾 [Checkpoint] Saving model weights for cycle $cycle...");
  // Implementation depends on your AFT weight format (usually JSON or Binary)
  // Example: File('weights_cycle_$cycle.json').writeAsStringSync(model.toJson());
}
