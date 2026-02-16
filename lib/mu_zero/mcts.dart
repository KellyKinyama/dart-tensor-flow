import 'dart:math' as math;
import '../core/tensor.dart';
import 'model.dart';

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

  /// Runs the search for a fixed number of simulations
  int search(
    Tensor rootState,
    List<int> legalActions, {
    int numSimulations = 50,
  }) {
    MCTSNode root = MCTSNode(rootState);

    // Initial expansion of the root
    final prediction = model.predict(rootState);
    root.priors = _filterPriors(prediction['policy']!.data, legalActions);

    for (int i = 0; i < numSimulations; i++) {
      _runSimulation(root);
    }

    // Return the action with the highest visit count
    return root.visitCounts.entries
        .reduce((a, b) => a.value > b.value ? a : b)
        .key;
  }

  void _runSimulation(MCTSNode root) {
    MCTSNode current = root;
    List<MCTSNode> searchPath = [current];
    List<int> actionPath = [];

    // 1. SELECT: Traverse the tree using UCB until we hit a node
    // that hasn't fully expanded its children.
    while (current.children.isNotEmpty) {
      int action = _selectAction(current);

      // If we haven't explored this action yet, stop selecting and start expanding
      if (!current.children.containsKey(action)) {
        actionPath.add(action);
        break;
      }

      actionPath.add(action);
      current = current.children[action]!;
      searchPath.add(current);
    }

    // 2. EXPAND: If the path is empty (first simulation), pick the best prior
    if (actionPath.isEmpty) {
      actionPath.add(_selectAction(current));
    }

    // 3. EVALUATE: Use the Dynamics Head
    int lastAction = actionPath.last;
    MCTSNode parent = searchPath.last;

    final dynamicsResult = model.dynamics(parent.state, lastAction);
    Tensor nextState = dynamicsResult['state']!;
    double reward = dynamicsResult['reward']!.data[0];

    final prediction = model.predict(nextState);
    double value = prediction['value']!.data[0];

    // 4. ADD NEW NODE
    MCTSNode newNode = MCTSNode(nextState);
    newNode.priors = _softmax(prediction['policy']!.data);
    parent.children[lastAction] = newNode;
    parent.rewards[lastAction] = reward;

    // 5. BACKUP
    _backpropagate(searchPath, actionPath, value);
  }

  int _selectAction(MCTSNode node) {
    double bestScore = -double.infinity;
    int bestAction = -1;

    // Sum total visits across all possible actions in priors
    int totalVisits = node.visitCounts.values.fold(0, (a, b) => a + b);

    for (var action in node.priors.keys) {
      double score = _calculateUCB(node, action, totalVisits);
      if (score > bestScore) {
        bestScore = score;
        bestAction = action;
      }
    }

    // Fallback if something goes wrong
    return bestAction != -1 ? bestAction : node.priors.keys.first;
  }

  double _calculateUCB(MCTSNode node, int action, int totalVisits) {
    double prior = node.priors[action] ?? 0;
    double visitCount = (node.visitCounts[action] ?? 0).toDouble();
    double qValue = node.getQ(action);

    // Upper Confidence Bound formula
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

        // Add immediate reward from dynamics
        cumulativeValue =
            (path[i].rewards[action] ?? 0) + discount * cumulativeValue;
      }
    }
  }

  Map<int, double> _filterPriors(List<double> logits, List<int> legalActions) {
    // Apply softmax only to legal moves
    double maxLogit = -double.infinity;
    for (int act in legalActions)
      if (logits[act] > maxLogit) maxLogit = logits[act];

    double sum = 0;
    Map<int, double> filtered = {};
    for (int act in legalActions) {
      double p = math.exp(logits[act] - maxLogit);
      filtered[act] = p;
      sum += p;
    }
    filtered.updateAll((k, v) => v / sum);
    return filtered;
  }

  Map<int, double> _softmax(List<double> logits) {
    double maxLogit = logits.reduce(math.max);
    double sum = 0;
    List<double> exps = logits.map((l) {
      double e = math.exp(l - maxLogit);
      sum += e;
      return e;
    }).toList();

    Map<int, double> result = {};
    for (int i = 0; i < exps.length; i++) {
      result[i] = exps[i] / sum;
    }
    return result;
  }
}
