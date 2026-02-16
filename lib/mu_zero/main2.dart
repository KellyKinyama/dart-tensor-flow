import 'trainer2.dart';
import 'model2.dart';
import '../transformers/attention_free_transformer/aft_chessformer.dart';
import '../optimizers/adam_chess.dart';

void main() async {
  final transformer = TransformerDecoder(vocabSize: 4098, embedSize: 128);
  final model = MuZeroModel(transformer, 128);
  final optimizer = Adam(transformer.parameters());
  final trainer = MuZeroTrainer(model, optimizer);

  print("--- Starting MuZero Training with Bishop ---");

  for (int cycle = 1; cycle <= 100; cycle++) {
    print("\n[Cycle $cycle] Generating games...");
    await trainer.runSelfPlaySession(5); // Play 5 games per cycle

    print("[Cycle $cycle] Training on replay buffer...");
    double loss = trainer.trainStep(batchSize: 32);

    print(
      "Cycle $cycle - Loss: ${loss.toStringAsFixed(4)} - Buffer: ${trainer.replayBuffer.length}",
    );

    if (cycle % 10 == 0) {
      // Optional: Save weights
      print("Saving model...");
    }
  }
}
