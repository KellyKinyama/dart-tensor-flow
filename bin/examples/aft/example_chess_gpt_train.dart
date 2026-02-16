// import 'dart:math' as math;
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

  // Standard square-to-index (a1=0, h8=63)
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
    return 4097; // Default to end token on error
  }
}

// --- TRAINING INTEGRATION ---

Future<void> main() async {
  print("--- Training Chess AFT-GPT with PGN Dataset ---");

  // 1. Load and Parse your dataset (mocking your dataset function)
  // Replace this with your actual dataset(100) call
  List<List<String>> pgnGames = dataset(100);

  // Add the <start> and <end> tokens to every game for the model
  final List<List<int>> encodedDataset = pgnGames.map((game) {
    return [encodeMove("<start>"), ...game.map(encodeMove), encodeMove(".")];
  }).toList();

  // 2. Model Parameters
  const int vocabSize = 4098; // 4096 moves + <start> + .
  const int bigSize = 128;
  const int blockSize = 32; // Increased for longer games

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

  final String filePath = 'transformer_weights.json';
  await loadModuleParameters(gpt, filePath);

  // 3. Training Loop
  for (int epoch = 0; epoch <= 1000; epoch++) {
    double totalLoss = 0;
    int sequencesProcessed = 0;

    for (var fullSeq in encodedDataset) {
      // Handle sequences longer than blockSize by sliding window or truncation
      if (fullSeq.length < 2) continue;

      final seq = fullSeq.length > blockSize + 1
          ? fullSeq.sublist(0, blockSize + 1)
          : fullSeq;

      optimizer.zeroGrad();

      // X is tokens 0 to N-1, Y is tokens 1 to N
      final xIndices = seq.sublist(0, seq.length - 1);
      final yIndices = seq.sublist(1);

      final logits = gpt.forward(xIndices, dummyEnc);
      final loss = crossEntropy(logits, yIndices, vocabSize);

      loss.backward();
      optimizer.step();

      totalLoss += loss.data[0];
      sequencesProcessed++;
    }

    if (epoch % 10 == 0) {
      print(
        "Epoch $epoch | Avg Loss: ${(totalLoss / sequencesProcessed).toStringAsFixed(6)}",
      );
    }

    // 2. Save the weights to a file
    await saveModuleParameters(gpt, filePath);
  }

  // 2. Save the weights to a file
  await saveModuleParameters(gpt, filePath);
}
