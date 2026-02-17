// import 'dart:io';
import 'package:bishop/bishop.dart';
import 'package:dart_tensor_flow/uitils/network_utils.dart';
import '../transformers/attention_free_transformer/aft_chessformer.dart';
import 'model2.dart';
import 'mcts.dart'; // Import the search logic

// ... [Keep encodeMove and decodeMove helpers from your previous snippet] ...
// Helper to translate model index back to Bishop Move
Move? decodeMove(int actionIdx, Game game) {
  if (actionIdx == 0) return null; // Padding/Start token

  int flatIdx = actionIdx - 1;
  int from64 = flatIdx ~/ 64;
  int to64 = flatIdx % 64;

  int fromX = from64 % 8;
  int fromY = from64 ~/ 8;
  int toX = to64 % 8;
  int toY = to64 ~/ 8;

  // Convert 8x8 coords back to Bishop's internal indices
  int fromSq = game.size.square(fromX, fromY);
  int toSq = game.size.square(toX, toY);

  // Find the matching legal move
  final legalMoves = game.generateLegalMoves();
  for (var m in legalMoves) {
    if (m.from == fromSq && m.to == toSq) return m;
  }
  return null;
}

// Helper to encode Bishop move to model index (same as trainer)
int encodeMove(Move m, Game game) {
  int fromX = game.size.file(m.from);
  int fromY = game.size.rank(m.from);
  int toX = game.size.file(m.to);
  int toY = game.size.rank(m.to);

  int from64 = (fromY * 8) + fromX;
  int to64 = (toY * 8) + toX;

  return (from64 * 64) + to64 + 1;
}

void main() async {
  // 1. Setup Architecture
  final transformer = TransformerDecoder(vocabSize: 4098, embedSize: 128);

  await loadModuleParameters(transformer, "muzero_chess_v1.json");
  final model = MuZeroModel(transformer, 128);

  // 2. Initialize the Search Engine
  final searchEngine = MuZeroSearch(model);

  // 3. Load Weights
  print("Loading weights...");

  // 4. Initialize Bishop Game
  final game = Game(variant: Variant.standard());
  List<int> history = [0];

  print("\n--- Model Play Test (Thinking Mode) ---\n");
  print(game.ascii());

  for (int turn = 0; turn < 50; turn++) {
    // 5. Get current latent state
    final rootState = model.represent(history);

    // 6. Get legal move indices for masking
    final legalMoves = game.generateLegalMoves();
    if (legalMoves.isEmpty) {
      print("Game Over: ${game.result?.readable}");
      break;
    }
    final List<int> legalActions = legalMoves
        .map((m) => encodeMove(m, game))
        .toList();

    // 7. THINK: Run MCTS simulations
    // This uses the Dynamics head to look ahead
    print("Model is thinking...");
    int bestActionIdx = searchEngine.search(
      rootState,
      legalActions,
      numSimulations: 30, // Adjust this for "deeper" thought
    );

    // 8. Execute the best move found by MCTS
    Move? chosenMove = decodeMove(bestActionIdx, game);

    if (chosenMove != null) {
      String san = game.toSan(chosenMove);
      game.makeMove(chosenMove);
      history.add(bestActionIdx);
      if (history.length > 16) history.removeAt(0);

      print("Turn ${turn + 1}: Model played $san");
      print(game.ascii());
      print("--------------------------------");
    } else {
      print("Error: Search returned an invalid move index.");
      break;
    }
  }

  print("\nFinal PGN: ${game.pgn()}");
}
