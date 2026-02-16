import 'dart:math' as math;
import 'package:bishop/bishop.dart';
import 'model2.dart';
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
  final int maxBufferSize = 10000;

  MuZeroTrainer(this.model, this.optimizer);

  /// Maps a Bishop Move to our model's 0-4097 action space
  int encodeMove(Move m, Game game) {
    int fromX = game.size.file(m.from);
    int fromY = game.size.rank(m.from);
    int toX = game.size.file(m.to);
    int toY = game.size.rank(m.to);

    int from64 = (fromY * 8) + fromX;
    int to64 = (toY * 8) + toX;

    return (from64 * 64) + to64 + 1;
  }

  Future<void> runSelfPlaySession(int numGames) async {
    final random = math.Random();

    for (int g = 0; g < numGames; g++) {
      final game = Game(variant: Variant.standard());
      List<int> history = [0];
      List<GameStep> gameSteps = [];

      while (!game.gameOver && history.length < 100) {
        final legalMoves = game.generateLegalMoves();
        if (legalMoves.isEmpty) break;

        // 1. Prediction & Masking
        final state = model.represent(history);
        final pred = model.predict(state);
        final logits = pred['policy']!.data;

        double maxLogit = -double.infinity;
        for (var m in legalMoves) {
          int idx = encodeMove(m, game);
          if (logits[idx] > maxLogit) maxLogit = logits[idx];
        }

        // 2. Softmax for Target Policy
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

        // 3. Move Selection
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

        // 4. Update Game
        game.makeMove(chosenMove);
        history.add(encodeMove(chosenMove, game));
        if (history.length > 16) history.removeAt(0);
      }

      // 5. Calculate Result
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

  double trainStep({int batchSize = 32}) {
    if (replayBuffer.length < batchSize) return 0.0;
    optimizer.zeroGrad();
    double totalLoss = 0;
    final random = math.Random();

    for (int i = 0; i < batchSize; i++) {
      final sample = replayBuffer[random.nextInt(replayBuffer.length)];
      final state = model.represent(sample.observations);
      final pred = model.predict(state);

      // Policy Cross-Entropy
      double pLoss = 0;
      final policy = pred['policy']!.data;
      double maxLogit = policy.reduce(math.max);
      double sumExp = 0;
      for (var v in policy) sumExp += math.exp((v - maxLogit).clamp(-10, 10));

      for (int j = 0; j < 4098; j++) {
        if (sample.targetPi[j] > 0) {
          double logSoftmax = (policy[j] - maxLogit) - math.log(sumExp + 1e-10);
          pLoss -= sample.targetPi[j] * logSoftmax;
        }
      }

      // Value MSE
      double vErr = pred['value']!.data[0] - sample.targetValue!;
      double vLoss = vErr * vErr;

      totalLoss += (pLoss + 0.5 * vLoss);
    }

    optimizer.step();
    return totalLoss / batchSize;
  }
}
