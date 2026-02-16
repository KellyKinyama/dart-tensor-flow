import 'dart:math' as math;
// import 'dart:io';
// import 'dart:convert';
import 'package:dart_tensor_flow/core/tensor.dart';
import 'package:dart_tensor_flow/optimizers/adam.dart';
import 'package:dart_tensor_flow/transformers/attention_free_transformer/aft_transformer_decoder.dart';

// --- CHESS MOVE ENCODING ---

/// Converts a UCI move like "e2e4" to a unique index [0...4095]
int encodeMove(String uci) {
  if (uci == "<start>") return 4096; // Special IDs above the 64x64 range
  if (uci == ".") return 4097;

  int squareToIndex(String sq) {
    int file = sq.codeUnitAt(0) - 'a'.codeUnitAt(0);
    int rank = int.parse(sq[1]) - 1;
    return rank * 8 + file;
  }

  int from = squareToIndex(uci.substring(0, 2));
  int to = squareToIndex(uci.substring(2, 4));
  return (from * 64) + to;
}

/// Converts an index [0...4095] back to a UCI string
String decodeMove(int index) {
  if (index == 4096) return "<start>";
  if (index == 4097) return ".";

  String indexToSquare(int idx) {
    int rank = idx ~/ 8 + 1;
    int file = idx % 8;
    return String.fromCharCode('a'.codeUnitAt(0) + file) + rank.toString();
  }

  int from = index ~/ 64;
  int to = index % 64;
  return indexToSquare(from) + indexToSquare(to);
}

// --- CORE UTILS ---

int sample(Tensor logits, int row, int vocabSize, double temperature) {
  int offset = row * vocabSize;
  double maxL = -double.infinity;
  for (int v = 0; v < vocabSize; v++) {
    if (logits.data[offset + v] / temperature > maxL) {
      maxL = logits.data[offset + v] / temperature;
    }
  }

  List<double> probs = [];
  double sumExp = 0;
  for (int v = 0; v < vocabSize; v++) {
    double p = math.exp((logits.data[offset + v] / temperature) - maxL);
    probs.add(p);
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

Tensor crossEntropy(Tensor logits, List<int> targets, int vocabSize) {
  int numTokens = targets.length;
  double totalLoss = 0;
  for (int t = 0; t < numTokens; t++) {
    int target = targets[t];
    int offset = t * vocabSize;
    double maxL = -double.infinity;
    for (int v = 0; v < vocabSize; v++) {
      if (logits.data[offset + v] > maxL) maxL = logits.data[offset + v];
    }
    double sumExp = 0;
    for (int v = 0; v < vocabSize; v++) {
      sumExp += math.exp(logits.data[offset + v] - maxL);
    }
    totalLoss +=
        (maxL + math.log(sumExp + 1e-12) - logits.data[offset + target]);
  }

  final loss = Tensor([1], children: {logits});
  loss.data[0] = totalLoss / numTokens;
  loss.onBackward = () {
    double gradFactor = 1.0 / numTokens;
    for (int t = 0; t < numTokens; t++) {
      int target = targets[t];
      int offset = t * vocabSize;
      double maxL = -double.infinity;
      for (int v = 0; v < vocabSize; v++) {
        if (logits.data[offset + v] > maxL) maxL = logits.data[offset + v];
      }
      double sumExp = 0;
      for (int v = 0; v < vocabSize; v++) {
        sumExp += math.exp(logits.data[offset + v] - maxL);
      }
      for (int v = 0; v < vocabSize; v++) {
        double prob = math.exp(logits.data[offset + v] - maxL) / sumExp;
        logits.grad[offset + v] +=
            (prob - (v == target ? 1.0 : 0.0)) * gradFactor;
      }
    }
  };
  return loss;
}

// --- MAIN TRAINING ---

Future<void> main() async {
  print("--- Training Chess Policy AFT-GPT (4096 Outputs) ---");

  // Vocabulary: 4096 (moves) + 1 (<start>) + 1 (.) = 4098
  const int vocabSize = 4098;
  const int bigSize = 128; // Increased for policy complexity
  const int blockSize = 16;
  const int startToken = 4096;
  const int endToken = 4097;

  final gpt = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: bigSize,
    encoderEmbedSize: bigSize,
    numLayers: 6,
    numHeads: 8,
    blockSize: blockSize,
  );

  final optimizer = Adam(gpt.parameters(), lr: 0.001);
  final dummyEnc = Tensor.zeros([1, bigSize]);

  // Dataset: Encoded UCI sequences
  final rawData = [
    ["<start>", "e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6", "."],
    ["<start>", "d2d4", "d7d5", "c2c4", "e7e6", "g1f3", "."],
  ];

  final dataset = rawData.map((seq) => seq.map(encodeMove).toList()).toList();

  print("Total Parameter Tensors: ${gpt.parameters().length}");

  for (int epoch = 0; epoch <= 500; epoch++) {
    double totalLoss = 0;
    for (var seq in dataset) {
      optimizer.zeroGrad();
      final x = seq.sublist(0, seq.length - 1);
      final y = seq.sublist(1);

      final logits = gpt.forward(x, dummyEnc);
      final loss = crossEntropy(logits, y, vocabSize);
      loss.backward();
      optimizer.step();
      totalLoss += loss.data[0];
    }

    if (epoch % 100 == 0) {
      print(
        "Epoch $epoch | Loss: ${(totalLoss / dataset.length).toStringAsFixed(6)}",
      );
    }
  }

  print("\n--- Testing Chess Move Generation ---");
  List<int> gen = [startToken, encodeMove("e2e4")];

  for (int i = 0; i < 8; i++) {
    final logits = gpt.forward(gen, dummyEnc);
    int nextId = sample(logits, gen.length - 1, vocabSize, 0.1);
    gen.add(nextId);
    print("  Predicted Move: ${decodeMove(nextId)}");
    if (nextId == endToken) break;
  }

  print("\nFull Sequence: ${gen.map(decodeMove).join(' ')}");
}
