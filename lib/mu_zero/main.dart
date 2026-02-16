import 'dart:async';
import 'dart:io';

import 'package:dart_tensor_flow/uitils/network_utils.dart';
import '../optimizers/adam_chess.dart';
import '../transformers/attention_free_transformer/aft_chessformer.dart';
import 'model.dart';
import 'trainer.dart';

void main() async {
  // CONFIGURATION
  const int vocabSize = 4098;
  const int embedSize = 128;
  const int blockSize = 16;
  const int numHeads = 8;
  const String checkpointPath = "muzero_chess_v1.json";

  print("--- Initializing MuZero Architecture ---");

  final transformer = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    encoderEmbedSize: embedSize,
    numLayers: 6,
    numHeads: numHeads,
    blockSize: blockSize,
  );

  if (embedSize % numHeads != 0) {
    throw Exception("EmbedSize must be divisible by numHeads");
  }

  final model = MuZeroModel(transformer, embedSize);
  final optimizer = Adam(transformer.parameters(), lr: 0.0001);
  final trainer = MuZeroTrainer(model, optimizer);

  // Load Checkpoint
  final file = File(checkpointPath);
  if (await file.exists()) {
    print("Loading weights from $checkpointPath...");
    try {
      await loadModuleParameters(transformer, checkpointPath);
      print("Weights loaded successfully.");
    } catch (e) {
      print("Warning: Architecture mismatch. Starting fresh.");
    }
  }

  print("--- Starting MuZero Training Loop ---");

  int cycle = 0;
  while (true) {
    cycle++;
    final stopwatch = Stopwatch()..start();

    // PHASE 1: SELF-PLAY
    print("\n[Cycle $cycle] Generating Self-Play Games...");
    await trainer.runSelfPlaySession(5);

    // PHASE 2: TRAINING
    print("[Cycle $cycle] Learning from Experience...");
    double cycleLoss = 0;
    const int numBatches = 16;

    for (int i = 0; i < numBatches; i++) {
      // Now capturing the returned loss
      double batchLoss = trainer.trainStep(batchSize: 32);
      cycleLoss += batchLoss;
    }

    double avgLoss = cycleLoss / numBatches;
    stopwatch.stop();

    // OUTPUT METRICS
    print("------------------------------------------------");
    print("Cycle: $cycle");
    print("Time:  ${stopwatch.elapsed.inSeconds}s");
    print("Loss:  ${avgLoss.toStringAsFixed(6)}");
    print("------------------------------------------------");

    // PHASE 3: PERSISTENCE
    await saveModuleParameters(transformer, checkpointPath);
  }
}
