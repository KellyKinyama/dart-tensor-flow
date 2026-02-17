import 'dart:math' as math;
import 'package:dart_tensor_flow/core/tensor.dart';
import 'package:dart_tensor_flow/dataset/chess.dart';
import 'package:dart_tensor_flow/loss/cross_entropy.dart';
import 'package:dart_tensor_flow/optimizers/adam.dart';
import 'package:dart_tensor_flow/transformers/attention_free_transformer/aft_transformer_decoder.dart';
import 'package:dart_tensor_flow/uitils/network_utils.dart';

// --- UCI ENCODING LOGIC ---

int encodeMove(String uci) {
  if (uci == "<start>") return 4096;
  if (uci == ".") return 4097;

  int squareToIndex(String sq) {
    int file = sq.codeUnitAt(0) - 'a'.codeUnitAt(0);
    int rank = int.parse(sq[1]) - 1;
    return rank * 8 + file;
  }

  try {
    int from = squareToIndex(uci.substring(0, 2));
    int to = squareToIndex(uci.substring(2, 4));
    return (from * 64) + to;
  } catch (e) {
    return 4097;
  }
}

// Added this to replace the need for the 'itos' map
String decodeMove(int index) {
  if (index == 4096) return "<start>";
  if (index == 4097) return ".";

  String indexToSquare(int idx) {
    int rank = (idx ~/ 8) + 1;
    int file = idx % 8;
    return String.fromCharCode('a'.codeUnitAt(0) + file) + rank.toString();
  }

  int from = index ~/ 64;
  int to = index % 64;
  return "${indexToSquare(from)}${indexToSquare(to)}";
}

Future<void> main() async {
  print("--- Running Chess AFT-GPT Inference ---");

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

  final String filePath = 'transformer_weights.json';

  // Load weights if they exist
  try {
    await loadModuleParameters(gpt, filePath);
    print("Weights loaded successfully.");
  } catch (e) {
    print(
      "Warning: Could not load weights. Running with random initialization.",
    );
  }

  final dummyEnc = Tensor.zeros([1, bigSize]);

  List<String> moves = ["e2e4", "d2d4", "g1f3", "c2c4", "a2a3"];

  for (int i = 0; i < moves.length; i++) {
    // Use the encodeMove function to create your seed
    List<int> seedIndices = [encodeMove("<start>"), encodeMove(moves[i])];

    generate(
      gpt,
      seedIndices,
      encodeMove("."), // endId
      vocabSize,
      blockSize,
      dummyEnc,
    );
  }
}

void generate(
  TransformerDecoder model,
  List<int> gen,
  int endId,
  int vocabSize,
  int blockSize,
  Tensor dummyEnc,
) {
  // Use decodeMove instead of itos map
  print("Seed: ${gen.map((id) => decodeMove(id)).join(' ')}");

  for (int i = 0; i < 15; i++) {
    List<int> context = gen.length > blockSize
        ? gen.sublist(gen.length - blockSize)
        : gen;

    final logits = model.forward(context, dummyEnc);

    // Sample next move
    int nextId = sample(logits, context.length - 1, vocabSize, 0.1);

    gen.add(nextId);
    print("  Next -> ${decodeMove(nextId)}");
    if (nextId == endId) break;
  }
  print("Result: ${gen.map((id) => decodeMove(id)).join(' ')}");
}

int sample(Tensor logits, int row, int vocabSize, double temperature) {
  int offset = row * vocabSize;
  double maxL = -double.infinity;

  for (int v = 0; v < vocabSize; v++) {
    double val = logits.data[offset + v] / temperature;
    if (val > maxL) maxL = val;
  }

  double sumExp = 0;
  List<double> probs = List.filled(vocabSize, 0);
  for (int v = 0; v < vocabSize; v++) {
    double p = math.exp((logits.data[offset + v] / temperature) - maxL);
    probs[v] = p;
    sumExp += p;
  }

  double r = math.Random().nextDouble() * sumExp;
  double cumulative = 0;
  for (int i = 0; i < vocabSize; i++) {
    cumulative += probs[i];
    if (r <= cumulative) return i;
  }
  return vocabSize - 1;
}
