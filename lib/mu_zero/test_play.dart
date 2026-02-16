
import 'package:bishop/bishop.dart';
import 'package:dart_tensor_flow/uitils/network_utils.dart';
import '../transformers/attention_free_transformer/aft_chessformer.dart';
import 'model.dart';

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
  final model = MuZeroModel(transformer, 128);

  // 2. Load Weights
  print("Loading weights...");
  await loadModuleParameters(transformer, "muzero_chess_v1.json");

  // 3. Initialize Bishop Game
  final game = Game(variant: Variant.standard());
  List<int> history = [0]; // Model's sequence starting token

  print("\n--- Model Play Test ---\n");
  print(game.ascii()); // Print starting board

  for (int turn = 0; turn < 20; turn++) {
    // 4. Get Model Prediction
    final state = model.represent(history);
    final prediction = model.predict(state);
    final policyLogits = prediction['policy']!.data;

    // 5. Mask & Select Best Legal Move
    Move? bestMove;
    double maxLogit = -double.infinity;

    final legalMoves = game.generateLegalMoves();
    if (legalMoves.isEmpty) {
      print("Game Over: No legal moves left.");
      break;
    }

    for (var m in legalMoves) {
      int idx = encodeMove(m, game);
      if (policyLogits[idx] > maxLogit) {
        maxLogit = policyLogits[idx];
        bestMove = m;
      }
    }

    if (bestMove != null) {
      // 6. Execute Move
      String san = game.toSan(bestMove);
      int moveIdx = encodeMove(bestMove, game);

      game.makeMove(bestMove);
      history.add(moveIdx);
      if (history.length > 16) history.removeAt(0);

      print("Turn ${turn + 1}: Model played $san (Index: $moveIdx)");
      print(game.ascii());
      print("--------------------------------");
    } else {
      print("Error: Model couldn't pick a legal move.");
      break;
    }

    // Slight delay so you can watch the game
    await Future.delayed(Duration(milliseconds: 500));
  }

  print("\nFinal PGN: ${game.pgn()}");
}
