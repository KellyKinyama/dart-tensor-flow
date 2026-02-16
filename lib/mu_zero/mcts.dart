import 'dart:math' as math;
import '../core/tensor.dart';
import 'model2.dart';

class MCTSNode {
  final Tensor state;
  final Map<int, MCTSNode> children = {};
  final Map<int, int> visitCounts = {};
  final Map<int, double> valueSums = {};
  final Map<int, double> rewards = {};
  Map<int, double> priors = {};

  MCTSNode(this.state);

  double getQ(int action) {
    if (!visitCounts.containsKey(action) || visitCounts[action] == 0) return 0;
    return valueSums[action]! / visitCounts[action]!;
  }
}

class MuZeroSearch {
  final MuZeroModel model;
  final double pb_c_base = 19652;
  final double pb_c_init = 1.25;
  final double discount = 0.99;

  MuZeroSearch(this.model);

  /// Main entry point for playing: Runs simulations and returns the best move.
  int search(
    Tensor rootState,
    List<int> legalActions, {
    int numSimulations = 50,
  }) {
    MCTSNode root = MCTSNode(rootState);

    // Initial expansion of the root with legal move filtering
    final prediction = model.predict(rootState);
    root.priors = filterPriors(prediction['policy']!.data, legalActions);

    for (int i = 0; i < numSimulations; i++) {
      runSimulation(root);
    }

    if (root.visitCounts.isEmpty) {
      return legalActions.isNotEmpty ? legalActions.first : 0;
    }

    // Return the action with the highest visit count
    return root.visitCounts.entries
        .reduce((a, b) => a.value > b.value ? a : b)
        .key;
  }

  /// Public version of prior filtering used by the Trainer
  Map<int, double> filterPriors(List<double> logits, List<int> legalActions) {
    if (legalActions.isEmpty) return {};

    double maxLogit = -double.infinity;
    for (int action in legalActions) {
      if (logits[action] > maxLogit) maxLogit = logits[action];
    }

    Map<int, double> priors = {};
    double sumExp = 0;
    for (int action in legalActions) {
      double p = math.exp((logits[action] - maxLogit).clamp(-10, 10));
      priors[action] = p;
      sumExp += p;
    }

    for (int action in legalActions) {
      priors[action] = priors[action]! / (sumExp + 1e-10);
    }

    return priors;
  }

  /// Performs a single MCTS simulation (Select -> Expand -> Evaluate -> Backup)
  void runSimulation(MCTSNode root) {
    MCTSNode current = root;
    List<MCTSNode> searchPath = [current];
    List<int> actionPath = [];

    // 1. SELECT: Follow UCB until we find a leaf or an unexplored action
    while (current.children.isNotEmpty) {
      int action = _selectAction(current);

      if (!current.children.containsKey(action)) {
        actionPath.add(action);
        break;
      }

      actionPath.add(action);
      current = current.children[action]!;
      searchPath.add(current);
    }

    // 2. Handle first simulation or root-only expansion
    if (actionPath.isEmpty) {
      actionPath.add(_selectAction(current));
    }

    // 3. EXPAND & BACKUP: Internal call to the dynamics head and backprop
    _expandAndBackpropagate(searchPath, actionPath);
  }

  void _expandAndBackpropagate(
    List<MCTSNode> searchPath,
    List<int> actionPath,
  ) {
    int lastAction = actionPath.last;
    MCTSNode parent = searchPath.last;

    // Dynamics prediction (The "Imagination")
    final dynamicsResult = model.dynamics(parent.state, lastAction);
    Tensor nextState = dynamicsResult['state']!;
    double reward = dynamicsResult['reward']!.data[0];

    // Inference on the new state
    final prediction = model.predict(nextState);
    double value = prediction['value']!.data[0];

    // Create the leaf node
    MCTSNode newNode = MCTSNode(nextState);
    newNode.priors = _softmax(prediction['policy']!.data);

    // Expansion
    parent.children[lastAction] = newNode;
    parent.rewards[lastAction] = reward;

    // Backpropagate the value up the search path
    _backpropagate(searchPath, actionPath, value);
  }

  int _selectAction(MCTSNode node) {
    double bestScore = -double.infinity;
    int bestAction = -1;

    int totalVisits = node.visitCounts.values.fold(0, (a, b) => a + b);

    for (var action in node.priors.keys) {
      double score = _calculateUCB(node, action, totalVisits);
      if (score > bestScore) {
        bestScore = score;
        bestAction = action;
      }
    }

    return bestAction != -1 ? bestAction : node.priors.keys.first;
  }

  double _calculateUCB(MCTSNode node, int action, int totalVisits) {
    double prior = node.priors[action] ?? 0;
    double visitCount = (node.visitCounts[action] ?? 0).toDouble();
    double qValue = node.getQ(action);

    // Standard MuZero PUCT formula
    double pb_c =
        math.log((totalVisits + pb_c_base + 1) / pb_c_base) + pb_c_init;
    pb_c *= math.sqrt(totalVisits) / (visitCount + 1);

    return qValue + pb_c * prior;
  }

  void _backpropagate(List<MCTSNode> path, List<int> actions, double value) {
    double cumulativeValue = value;
    for (int i = path.length - 1; i >= 0; i--) {
      if (i < actions.length) {
        int action = actions[i];
        path[i].visitCounts[action] = (path[i].visitCounts[action] ?? 0) + 1;
        path[i].valueSums[action] =
            (path[i].valueSums[action] ?? 0) + cumulativeValue;

        // Cumulative reward: R + discount * V
        cumulativeValue =
            (path[i].rewards[action] ?? 0) + discount * cumulativeValue;
      }
    }
  }

  Map<int, double> _softmax(List<double> logits) {
    double maxLogit = logits.reduce(math.max);
    double sum = 0;
    Map<int, double> result = {};

    for (int i = 0; i < logits.length; i++) {
      double e = math.exp((logits[i] - maxLogit).clamp(-10, 10));
      result[i] = e;
      sum += e;
    }

    result.updateAll((k, v) => v / (sum + 1e-10));
    return result;
  }
}
