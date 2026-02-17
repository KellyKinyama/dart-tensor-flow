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

  int from64 = (fromY * 8) + fromX;
  int to64 = (toY * 8) + toX;

  // Your Bishop logic: (from*64) + to + 1
  return (from64 * 64) + to64 + 1;
}

String decodeMoveBishop(int index) {
  if (index == 4096) return "<start>";
  if (index == 4097) return ".";

  int flatIdx = index - 1;
  int from64 = flatIdx ~/ 64;
  int to64 = flatIdx % 64;

  String indexToSquare(int idx) {
    int rank = (idx ~/ 8) + 1;
    int file = idx % 8;
    return String.fromCharCode('a'.codeUnitAt(0) + file) + rank.toString();
  }

  return "${indexToSquare(from64)}${indexToSquare(to64)}";
}

// --- MAIN INFERENCE ---

Future<void> main() async {
  print("--- Chess AFT Policy Play (Bishop Validated) ---");

  final gpt = TransformerDecoder(
    vocabSize: 4098,
    embedSize: 128,
    encoderEmbedSize: 128,
    numLayers: 6,
    numHeads: 8,
    blockSize: 32,
  );

  try {
    await loadModuleParameters(gpt, 'transformer_weights.json');
    print("Weights loaded successfully.");
  } catch (e) {
    print("Warning: Weights not found. Using random initialization.");
  }

  // 1. Setup the Game and History
  final game = Game(variant: Variant.standard());

  // MANDATORY: Start the sequence with the <start> token (4096)
  List<int> history = [4096];

  print("Initial Board:\n${game.ascii()}");

  // 2. Play 20 turns using the model's policy
  for (int turn = 0; turn < 20; turn++) {
    final legalMoves = game.generateLegalMoves();
    if (legalMoves.isEmpty) {
      print("Game Over: ${game.result?.readable}");
      break;
    }

    // Get logits for the current sequence
    final dummyEnc = Tensor.zeros([1, 128]);
    final logits = gpt.forward(history, dummyEnc);

    // 3. Select best move strictly from legal options
    int bestActionIdx = selectBestLegalMove(
      logits,
      history.length - 1,
      game,
      legalMoves,
    );

    // 4. Update the game state and sequence history
    Move? chosenMove = decodeMoveBishopObject(bestActionIdx, game);
    if (chosenMove != null) {
      String san = game.toSan(chosenMove);
      game.makeMove(chosenMove);
      history.add(bestActionIdx);

      // Maintain context window
      if (history.length > 32) history.removeAt(0);

      print("Turn ${turn + 1}: Model plays $san (ID: $bestActionIdx)");
      print(game.ascii());
    } else {
      print("Error: Model returned an invalid index ($bestActionIdx)");
      break;
    }
  }

  print("\nFinal PGN: ${game.pgn()}");
}

// --- HELPERS ---

int selectBestLegalMove(
  Tensor logits,
  int row,
  Game game,
  List<Move> legalMoves,
) {
  const int vocabSize = 4098;
  int offset = row * vocabSize;

  int bestIdx = -1;
  double maxLogit = -double.infinity;

  for (var m in legalMoves) {
    int idx = encodeMoveBishop(m, game);
    double score = logits.data[offset + idx];

    if (score > maxLogit) {
      maxLogit = score;
      bestIdx = idx;
    }
  }
  return bestIdx;
}

/// Utility to find the Bishop Move object from a model index
Move? decodeMoveBishopObject(int actionIdx, Game game) {
  int flatIdx = actionIdx - 1;
  int fromSq = game.size.square(flatIdx ~/ 64 % 8, flatIdx ~/ 64 ~/ 8);
  int toSq = game.size.square(flatIdx % 64 % 8, flatIdx % 64 ~/ 8);

  for (var m in game.generateLegalMoves()) {
    if (m.from == fromSq && m.to == toSq) return m;
  }
  return null;
}
