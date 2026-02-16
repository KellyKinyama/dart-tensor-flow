import 'dart:math' as math;
import 'package:bishop/bishop.dart';
import 'model2.dart';
import 'mcts.dart';
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

  int encodeMove(Move m, Game game) {
    int fromX = game.size.file(m.from);
    int fromY = game.size.rank(m.from);
    int toX = game.size.file(m.to);
    int toY = game.size.rank(m.to);
    return ((fromY * 8 + fromX) * 64) + (toY * 8 + toX) + 1;
  }

  Future<void> runSelfPlaySession(int numGames, {int simulations = 30}) async {
    final search = MuZeroSearch(model);
    final random = math.Random();

    for (int g = 0; g < numGames; g++) {
      final game = Game(variant: Variant.standard());
      List<int> history = [0];
      List<GameStep> gameSteps = [];

      while (!game.gameOver && history.length < 100) {
        final legalMoves = game.generateLegalMoves();
        if (legalMoves.isEmpty) break;

        final List<int> legalActions = legalMoves
            .map((m) => encodeMove(m, game))
            .toList();

        // 1. MCTS SEARCH
        final rootState = model.represent(history);
        MCTSNode root = MCTSNode(rootState);

        final prediction = model.predict(rootState);
        root.priors = search.filterPriors(
          prediction['policy']!.data,
          legalActions,
        );

        for (int i = 0; i < simulations; i++) {
          search.runSimulation(root);
        }

        // 2. CREATE TARGET POLICY
        List<double> targetPi = List.filled(4098, 0.0);
        int totalVisits = root.visitCounts.values.fold(0, (a, b) => a + b);

        // Temperature handling
        double temp = (history.length < 30) ? 1.0 : 0.5;
        double sumVisitExp = 0;
        for (int action in root.priors.keys) {
          double v = (root.visitCounts[action] ?? 0).toDouble();
          double p = math.pow(v + 1e-8, 1 / temp).toDouble();
          targetPi[action] = p;
          sumVisitExp += p;
        }

        for (int i = 0; i < targetPi.length; i++) {
          if (targetPi[i] > 0) targetPi[i] /= (sumVisitExp + 1e-10);
        }

        // 3. SELECT ACTION
        int chosenAction = _sampleAction(targetPi, random);
        gameSteps.add(GameStep(List.from(history), List.from(targetPi)));

        Move chosenMove = legalMoves.firstWhere(
          (m) => encodeMove(m, game) == chosenAction,
        );
        game.makeMove(chosenMove);

        history.add(chosenAction);
        if (history.length > 16) history.removeAt(0);
      }

      // 4. BACKFILL OUTCOME (Corrected Result Logic)
      double outcome = 0.0;
      // final result = game.result;

      if (game.won) {
        // game.winner returns the player index (0 for White, 1 for Black)
        outcome = (game.winner == Bishop.white) ? 1.0 : -1.0;
      } else if (game.drawn) {
        outcome = 0.0; // Standard for draws
      }

      for (var step in gameSteps) {
        step.targetValue = outcome;
        if (replayBuffer.length >= maxBufferSize) replayBuffer.removeAt(0);
        replayBuffer.add(step);
      }
    }
  }

  int _sampleAction(List<double> probs, math.Random r) {
    double sample = r.nextDouble();
    double cumulative = 0;
    for (int i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (sample <= cumulative) return i;
    }
    return 0;
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

      // Policy Loss
      double pLoss = 0;
      final policyLogits = pred['policy']!.data;
      double maxLogit = policyLogits.reduce(math.max);
      double sumExp = 0;
      for (var v in policyLogits)
        sumExp += math.exp((v - maxLogit).clamp(-10, 10));

      for (int j = 0; j < 4098; j++) {
        if (sample.targetPi[j] > 0) {
          double logSoftmax =
              (policyLogits[j] - maxLogit) - math.log(sumExp + 1e-10);
          pLoss -= sample.targetPi[j] * logSoftmax;
        }
      }

      // Value Loss
      double vErr = pred['value']!.data[0] - sample.targetValue!;
      double vLoss = vErr * vErr;

      totalLoss += (pLoss + vLoss);
    }

    optimizer.step();
    return totalLoss / batchSize;
  }
}
