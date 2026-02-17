import 'dart:math' as math;
import 'package:bishop/bishop.dart';
import 'package:dart_tensor_flow/core/tensor.dart';
import 'package:dart_tensor_flow/transformers/attention_free_transformer/aft_transformer_decoder.dart';
import 'package:dart_tensor_flow/uitils/network_utils.dart';

// --- ENCODING LOGIC ---

int encodeMoveBishop(Move m, Game game) {
  int fromX = game.size.file(m.from);
  int fromY = game.size.rank(m.from);
  int toX = game.size.file(m.to);
  int toY = game.size.rank(m.to);
  return (fromY * 8 + fromX) * 64 + (toY * 8 + toX) + 1;
}

Move? decodeMoveBishopObject(int actionIdx, Game game) {
  if (actionIdx <= 0 || actionIdx > 4096) return null;
  int flatIdx = actionIdx - 1;
  int fromSq = game.size.square(flatIdx ~/ 64 % 8, flatIdx ~/ 64 ~/ 8);
  int toSq = game.size.square(flatIdx % 64 % 8, flatIdx % 64 ~/ 8);
  for (var m in game.generateLegalMoves()) {
    if (m.from == fromSq && m.to == toSq) return m;
  }
  return null;
}

// --- MCTS LOGIC ---

class MCTSNode {
  final List<int> history;
  final Map<int, MCTSNode> children = {};
  final Map<int, int> visitCounts = {};
  final Map<int, double> valueSums = {};
  Map<int, double> priors = {};

  MCTSNode(this.history);

  double getQ(int action) => (visitCounts[action] ?? 0) == 0
      ? 0
      : valueSums[action]! / visitCounts[action]!;
}

class ChessSearch {
  final TransformerDecoder model;
  final double pb_c_base = 19652;
  final double pb_c_init = 1.25;

  ChessSearch(this.model);

  int search(List<int> currentHistory, Game game, {int simulations = 20}) {
    MCTSNode root = MCTSNode(List.from(currentHistory));
    _expand(root, game);

    for (int i = 0; i < simulations; i++) {
      _runSimulation(root, Game(fen: game.pgn()));
    }

    return root.visitCounts.entries
        .reduce((a, b) => a.value > b.value ? a : b)
        .key;
  }

  void _runSimulation(MCTSNode root, Game simGame) {
    MCTSNode current = root;
    List<MCTSNode> path = [current];
    List<int> actions = [];

    // 1. SELECT
    while (current.children.isNotEmpty) {
      int action = _selectAction(current);
      actions.add(action);
      Move? m = decodeMoveBishopObject(action, simGame);
      if (m != null) simGame.makeMove(m);
      if (!current.children.containsKey(action)) break;
      current = current.children[action]!;
      path.add(current);
    }

    // 2. EXPAND & EVALUATE
    double value = _expand(current, simGame);

    // 3. BACKPROPAGATE
    for (int i = 0; i < path.length; i++) {
      if (i < actions.length) {
        int act = actions[i];
        path[i].visitCounts[act] = (path[i].visitCounts[act] ?? 0) + 1;
        path[i].valueSums[act] = (path[i].valueSums[act] ?? 0) + value;
      }
    }
  }

  double _expand(MCTSNode node, Game game) {
    final logits = model.forward(node.history, Tensor.zeros([1, 128]));
    final legalMoves = game.generateLegalMoves();
    if (legalMoves.isEmpty) return 0.5; // Draw/Terminal

    // Filter priors
    int offset = (node.history.length - 1) * 4098;
    Map<int, double> filtered = {};
    double maxL = -double.infinity;

    for (var m in legalMoves) {
      int id = encodeMoveBishop(m, game);
      double l = logits.data[offset + id];
      if (l > maxL) maxL = l;
      filtered[id] = l;
    }

    double sumExp = 0;
    filtered.updateAll((id, l) {
      double e = math.exp(l - maxL);
      sumExp += e;
      return e;
    });
    node.priors = filtered.map((id, e) => MapEntry(id, e / sumExp));

    return 0.5; // Value head placeholder
  }

  int _selectAction(MCTSNode node) {
    int totalVisits = node.visitCounts.values.fold(0, (a, b) => a + b);
    double bestScore = -double.infinity;
    int bestAct = node.priors.keys.first;

    for (int action in node.priors.keys) {
      double visitCount = (node.visitCounts[action] ?? 0).toDouble();
      double pb_c =
          (math.log((totalVisits + pb_c_base + 1) / pb_c_base) + pb_c_init) *
          math.sqrt(totalVisits) /
          (visitCount + 1);
      double score = node.getQ(action) + pb_c * node.priors[action]!;
      if (score > bestScore) {
        bestScore = score;
        bestAct = action;
      }
    }
    return bestAct;
  }
}

// --- MAIN ---

Future<void> main() async {
  final gpt = TransformerDecoder(
    vocabSize: 4098,
    embedSize: 128,
    numLayers: 6,
    numHeads: 8,
    blockSize: 32,
  );
  await loadModuleParameters(
    gpt,
    'transformer_weights.json',
  ).catchError((_) => print("No weights found."));

  final searchEngine = ChessSearch(gpt);
  final game = Game(variant: Variant.standard());
  List<int> history = [4096]; // <start>

  for (int turn = 0; turn < 20; turn++) {
    print("\nTurn ${turn + 1}: Searching...");
    int bestMoveIdx = searchEngine.search(history, game, simulations: 30);

    Move? m = decodeMoveBishopObject(bestMoveIdx, game);
    if (m != null) {
      print("Model plays: ${game.toSan(m)}");
      game.makeMove(m);
      history.add(bestMoveIdx);
      if (history.length > 32) history.removeAt(0);
      print(game.ascii());
    } else {
      break;
    }
  }
}
