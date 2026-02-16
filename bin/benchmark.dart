// import 'dart:typed_data';
import 'package:dart_tensor_flow/core/tensor.dart';
// import 'package:dart_tensor_flow/dart_tensor_flow.dart'; // Adjust based on your package name

void main() {
  const int size = 512; // Test with a 512x512 matrix
  print('Benchmarking Matrix Multiplication ($size x $size)');
  print('--------------------------------------------------');

  final a = Tensor.random([size, size]);
  final b = Tensor.random([size, size]);

  // 1. Warm-up (Let the JIT compiler optimize the SIMD code)
  print('Warming up JIT compiler...');
  for (int i = 0; i < 5; i++) {
    a.matmul(b);
  }

  // 2. Execution Profile
  final sw = Stopwatch()..start();
  const int iterations = 10;

  for (int i = 0; i < iterations; i++) {
    a.matmul(b);
  }
  sw.stop();

  // 3. Results Calculation
  final double avgMs = sw.elapsedMilliseconds / iterations;
  final double totalOps = 2.0 * size * size * size; // Flops in matmul
  final double gflops = (totalOps / (avgMs / 1000.0)) / 1e9;

  print('Average execution time: ${avgMs.toStringAsFixed(2)} ms');
  print('Estimated Performance: ${gflops.toStringAsFixed(3)} GFLOPS');
  print('--------------------------------------------------');
}
