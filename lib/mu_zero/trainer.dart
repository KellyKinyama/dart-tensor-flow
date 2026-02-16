import 'dart:math' as math;
import 'package:bishop/bishop.dart'; // Ensure 'bishop' is in your pubspec.yaml

import 'model.dart';
import '../optimizers/adam_chess.dart';

class GameStep {
  final List<int> observations;
  final List<double> targetPi;
  double? targetValue;

  GameStep(this.observations, this.targetPi);
}

class MuZeroTrainer {
  final MuZeroModel model;
  final Adam optimizer;
  final List<GameStep> replayBuffer = [];
  final int maxBufferSize = 5000;

  MuZeroTrainer(this.model, this.optimizer);

  /// Maps a Bishop Move to our model's 0-4097 action space.
  /// Bishop moves contain 'from' and 'to' integer indices (0-63 for standard chess).
  int encodeMove(Move m, Game game) {
    // 1. Get Bishop's internal size object
    final size = game.size;

    // 2. Extract X (file) and Y (rank) using Bishop's built-in conversion
    int fromX = size.file(m.from); // Correctly strips internal padding
    int fromY = size.rank(m.from);
    int toX = size.file(m.to);
    int toY = size.rank(m.to);

    // 3. Convert to a standard flat 0-63 coordinate
    // Standard: (Rank * 8) + File
    int from64 = (fromY * 8) + fromX;
    int to64 = (toY * 8) + toX;

    // 4. Map to your model's action space index (1-4096)
    // index 0 is reserved for Start/Padding
    int finalIdx = (from64 * 64) + to64 + 1;

    return finalIdx;
  }

  Future<void> runSelfPlaySession(int numGames) async {
    final random = math.Random();

    for (int g = 0; g < numGames; g++) {
      // Initialize Bishop Game (Standard Variant)
      final game = Game(variant: Variant.standard());
      List<int> history = [0];
      List<GameStep> gameSteps = [];

      // Bishop: game.gameOver checks checkmate, stalemate, and draw rules
      while (!game.gameOver && history.length < 100) {
        // Bishop: Get all legal moves for current player
        final List<Move> legalMoves = game.generateLegalMoves();
        if (legalMoves.isEmpty) break;

        final state = model.represent(history);
        final pred = model.predict(state);
        final logits = pred['policy']!.data;

        // Masking: Convert legal Bishop moves to our 1-4096 indices
        double maxLogit = -double.infinity;
        for (var m in legalMoves) {
          int idx = encodeMove(m, game);
          if (logits[idx] > maxLogit) maxLogit = logits[idx];
        }

        // Softmax & Probabilities
        double sumExp = 0;
        List<double> probs = List.filled(4098, 0.0);
        for (var m in legalMoves) {
          int idx = encodeMove(m, game);
          double p = math.exp((logits[idx] - maxLogit).clamp(-10, 10));
          probs[idx] = p;
          sumExp += p;
        }
        for (int j = 0; j < probs.length; j++) {
          if (probs[j] > 0) probs[j] /= (sumExp + 1e-10);
        }

        // Sampling
        double r = random.nextDouble();
        double cumulative = 0;
        Move chosenMove = legalMoves.first;
        for (var m in legalMoves) {
          int idx = encodeMove(m, game);
          cumulative += probs[idx];
          if (r <= cumulative) {
            chosenMove = m;
            break;
          }
        }

        gameSteps.add(GameStep(List.from(history), List.from(probs)));

        // Bishop: execute move
        game.makeMove(chosenMove);
        history.add(encodeMove(chosenMove, game));

        if (history.length > 16) history.removeAt(0);
      }

      // Outcome Logic (DrawnGame, WonGame, etc.)
      double outcome = 0.0;
      final result = game.result;
      if (result is WonGame) {
        outcome = (result.winner == Bishop.white) ? 1.0 : -1.0;
      }

      for (var step in gameSteps) {
        step.targetValue = outcome;
        if (replayBuffer.length >= maxBufferSize) replayBuffer.removeAt(0);
        replayBuffer.add(step);
      }
    }
  }

  /// The optimization step (identical logic to previous version)
  double trainStep({int batchSize = 32}) {
    if (replayBuffer.length < batchSize) return 0.0;

    optimizer.zeroGrad();
    double totalBatchLoss = 0;
    final random = math.Random();

    for (int i = 0; i < batchSize; i++) {
      final sample = replayBuffer[random.nextInt(replayBuffer.length)];
      final state = model.represent(sample.observations);
      final pred = model.predict(state);

      // Policy Loss
      double pLoss = 0;
      final policyData = pred['policy']!.data;
      double maxLogit = policyData.reduce(math.max);
      double sumExp = 0;
      for (var val in policyData)
        sumExp += math.exp((val - maxLogit).clamp(-10, 10));

      for (int j = 0; j < sample.targetPi.length; j++) {
        if (sample.targetPi[j] > 0) {
          double logSoftmax =
              (policyData[j] - maxLogit) - math.log(sumExp + 1e-10);
          pLoss -= sample.targetPi[j] * logSoftmax;
        }
      }

      // Value Loss
      double vPred = pred['value']!.data[0];
      double vErr = vPred - sample.targetValue!;
      double vLoss = vErr * vErr;

      totalBatchLoss += (pLoss + 0.5 * vLoss);
    }

    optimizer.step();
    return totalBatchLoss / batchSize;
  }
}
