class MoveCodec {
  static const List<String> files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  static const List<String> ranks = ['1', '2', '3', '4', '5', '6', '7', '8'];

  /// Decodes a MuZero move index (0-4095) into a UCI string (e.g., "e2e4")
  static String indexToUci(int index) {
    if (index == 0) return "START";

    // The 4096 moves are usually structured as:
    // (from_square [0-63]) * 64 + (to_square [0-63])
    // Note: This is a simplified version of the AlphaZero 73-plane encoding
    int fromSq = (index - 1) ~/ 64;
    int toSq = (index - 1) % 64;

    if (fromSq > 63 || toSq > 63) return "invalid";

    String fromName = _squareName(fromSq);
    String toName = _squareName(toSq);

    return "$fromName$toName";
  }

  static String _squareName(int sq) {
    int file = sq % 8;
    int rank = sq ~/ 8;
    return "${files[file]}${ranks[rank]}";
  }
}
