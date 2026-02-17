import 'package:bishop/bishop.dart';
import 'package:dart_tensor_flow/transformers/attention_free_transformer/aft_transformer_decoder.dart';
import 'package:dart_tensor_flow/uitils/network_utils.dart';

import 'mcts_policy.dart';

Future<void> main() async {
  print("--- Chess AFT-GPT MCTS Inference ---");

  // 1. Setup Architecture
  // Note: Ensure vocabSize and blockSize match your trained model
  const int vocabSize = 4098;
  const int bigSize = 128;
  const int blockSize = 32;

  final gpt = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: bigSize,
    encoderEmbedSize: bigSize,
    numLayers: 6,
    numHeads: 8,
    blockSize: blockSize,
  );

  // 2. Load Weights
  try {
    await loadModuleParameters(gpt, 'transformer_weights.json');
    print("Weights loaded successfully.");
  } catch (e) {
    print("Warning: Weights not found. Running with random initialization.");
  }

  // 3. Initialize Search Engine and Game State
  final searchEngine = ChessSearch(gpt);
  final game = Game(variant: Variant.standard());

  // History starts with the <start> token (4096)
  List<int> history = [4096];

  print("\nInitial Board State:");
  print(game.ascii());
  print("--------------------------------");

  // 4. Play Loop
  for (int turn = 0; turn < 50; turn++) {
    // Check if the game is over before searching
    if (game.generateLegalMoves().isEmpty) {
      print("Game Over: ${game.result?.readable ?? 'Unknown result'}");
      break;
    }

    print("Turn ${turn + 1}: Model is thinking (MCTS Search)...");

    // 5. CALL MCTS SEARCH
    // simulations: 30-50 is usually a good balance between speed and quality
    int bestMoveIdx = searchEngine.search(history, game, simulations: 30);

    // 6. Execute the Move
    Move? chosenMove = decodeMoveBishopObject(bestMoveIdx, game);

    if (chosenMove != null) {
      String san = game.toSan(chosenMove);
      game.makeMove(chosenMove);

      // Update transformer history
      history.add(bestMoveIdx);

      // Slide context window if it exceeds transformer blockSize
      if (history.length > blockSize) {
        history.removeAt(0);
      }

      print("Model played: $san (Move ID: $bestMoveIdx)");
      print(game.ascii());
      print("--------------------------------");
    } else {
      print(
        "Error: Search engine returned an invalid move index ($bestMoveIdx).",
      );
      break;
    }
  }

  print("\nFinal PGN Output:");
  print(game.pgn());
}
