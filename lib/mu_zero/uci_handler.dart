import 'dart:io';
import 'dart:convert';
import 'package:bishop/bishop.dart';
import 'package:dart_tensor_flow/uitils/network_utils.dart';
import '../mu_zero/model2.dart';
import '../mu_zero/mcts.dart';
import '../transformers/attention_free_transformer/aft_chessformer.dart';

class UCIHandler {
  final MuZeroModel model;
  late MuZeroSearch search;
  Game game = Game(variant: Variant.standard());
  bool isDebug = false;

  UCIHandler(this.model) {
    search = MuZeroSearch(model);
  }

  /// The main loop that reads from stdin
  void start() {
    // UCI requires UTF-8 and handling of \n or \r\n
    stdin
        .transform(utf8.decoder)
        .transform(const LineSplitter())
        .listen(_handleCommand);
  }

  void _handleCommand(String line) {
    List<String> tokens = line.trim().split(RegExp(r'\s+'));
    if (tokens.isEmpty || tokens[0].isEmpty) return;

    String command = tokens[0].toLowerCase();

    switch (command) {
      case 'uci':
        _respondUCI();
        break;
      case 'isready':
        _send('readyok');
        break;
      case 'ucinewgame':
        game = Game(variant: Variant.standard());
        break;
      case 'position':
        _handlePosition(tokens);
        break;
      case 'go':
        _handleGo(tokens);
        break;
      case 'debug':
        isDebug = tokens.length > 1 && tokens[1] == 'on';
        break;
      case 'stop':
        // In a complex engine, this would trigger an early return from search
        break;
      case 'quit':
        exit(0);
      default:
        if (isDebug) _send('info string unknown command: $command');
    }
  }

  void _respondUCI() {
    _send('id name GeminiMuZero');
    _send('id author GeminiAI');
    // Add options here if you want to allow Hash/Threads configuration
    _send('uciok');
  }

  void _handlePosition(List<String> tokens) {
    // Format: position [startpos | fen <fen>] moves <move1> <move2> ...
    int movesIdx = tokens.indexOf('moves');

    // 1. Setup Board
    if (tokens.contains('startpos')) {
      game = Game(variant: Variant.standard());
    } else if (tokens.contains('fen')) {
      int fenStart = tokens.indexOf('fen') + 1;
      int fenEnd = (movesIdx != -1) ? movesIdx : tokens.length;
      String fen = tokens.sublist(fenStart, fenEnd).join(' ');
      game = Game(variant: Variant.standard(), fen: fen);
    }

    // 2. Play through moves
    if (movesIdx != -1) {
      for (int i = movesIdx + 1; i < tokens.length; i++) {
        Move? m = _parseUCIAction(tokens[i]);
        if (m != null) game.makeMove(m);
      }
    }
  }

  /// Parses UCI string (e.g., e2e4 or a7a8q) to a Bishop Move
  Move? _parseUCIAction(String uci) {
    // We generate legal moves for the current state to find a match
    final moves = game.generateLegalMoves();
    for (var m in moves) {
      if (_moveToUCI(m) == uci) {
        return m;
      }
    }
    // If no legal move matches the string, it's an invalid move
    if (isDebug) _send('info string error: invalid move $uci');
    return null;
  }

  void _handleGo(List<String> tokens) {
    // For MuZero, we need to convert the current game state to our history tensor
    // This assumes your model.represent takes a list of move indices
    List<int> history = _generateHistoryFromGame(game);

    final rootState = model.represent(history);
    final legalMoves = game.generateLegalMoves();
    final List<int> legalActions = legalMoves
        .map((m) => _encodeMove(m))
        .toList();

    // In a real UCI engine, you'd check 'wtime' / 'movetime' tokens here
    // For now, we use a fixed simulation count
    int chosenAction = search.search(
      rootState,
      legalActions,
      numSimulations: 50,
    );

    Move bestMove = legalMoves.firstWhere(
      (m) => _encodeMove(m) == chosenAction,
    );

    _send('bestmove ${_moveToUCI(bestMove)}');
  }

  /// Converts bishop Move to UCI string (e.g., e2e4)
  /// Converts bishop Move to UCI string (e.g., e2e4)
  String _moveToUCI(Move m) {
    // Use squareName from the board size
    final String from = game.size.squareName(m.from);
    final String to = game.size.squareName(m.to);
    String uci = from + to;

    // Handle promotion using the BuiltVariant's pieceSymbol lookup
    if (m.promotion && m.promoPiece != null) {
      // pieceSymbol returns the char (e.g., 'q') for the type index
      // We pass Bishop.white to ensure we get the lowercase version for UCI
      uci += game.variant
          .pieceSymbol(m.promoPiece!, Bishop.black)
          .toLowerCase();
    }

    // Handle Drops (UCI format: P@f3)
    if (m.handDrop && m.dropPiece != null) {
      uci = '${game.variant.pieceSymbol(m.dropPiece!, Bishop.white)}@$to';
    }

    return uci;
  }

  int _encodeMove(Move m) {
    int fromX = game.size.file(m.from);
    int fromY = game.size.rank(m.from);
    int toX = game.size.file(m.to);
    int toY = game.size.rank(m.to);

    int base = ((fromY * 8 + fromX) * 64) + (toY * 8 + toX);

    // If there is a promotion, offset the index to a new range
    // Queen: +4096, Rook: +8192, etc.
    if (m.promotion && m.promoPiece != null) {
      // Bishop uses piece type indices (usually Knight=3, Bishop=4, Rook=5, Queen=6)
      return 4096 * (m.promoPiece! - 2) + base;
    }

    return base;
  }

  List<int> _generateHistoryFromGame(Game g) {
    // MuZero usually needs the last N moves.
    // This is a simplified version; you should ideally store the
    // encoded moves as the game progresses.
    return [0]; // Placeholder: replace with actual encoded history
  }

  void _send(String msg) {
    stdout.write('$msg\n');
  }
}

void main(List<String> args) async {
  // Vocabulary: 4096 (moves) + 1 (<start>) + 1 (.) = 4098
  const int vocabSize = 4098;
  const int bigSize = 128; // Increased for policy complexity
  const int blockSize = 16;
  const int startToken = 4096;
  const int endToken = 4097;

  final gpt = TransformerDecoder(vocabSize: 4098, embedSize: 128);

  final model = MuZeroModel(gpt, bigSize); // Your model loading logic
  const String checkpointPath = "muzero_chess_v1.json";
  await loadModuleParameters(gpt, checkpointPath);

  if (args.contains('--uci')) {
    final handler = UCIHandler(model);
    handler.start();
  } else {
    // Start the trainer loop we wrote earlier
  }
}
