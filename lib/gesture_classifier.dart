import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class GestureClassifier {
  Interpreter? _interpreter;

  List<String> labels = [];

  bool get isLoaded => _interpreter != null;

  Future<void> load() async {
    // Load TensorFlow Lite model
    _interpreter = await Interpreter.fromAsset(
      'assets/models/gesture_model.tflite',
    );

    // Load gesture names
    final labelText = await rootBundle.loadString(
      'assets/models/gesture_labels.txt',
    );

    labels = labelText
        .split('\n')
        .map((label) => label.trim())
        .where((label) => label.isNotEmpty)
        .toList();

    final inputTensor =
        _interpreter!.getInputTensor(0);

    final outputTensor =
        _interpreter!.getOutputTensor(0);

    print('========== TFLITE MODEL ==========');

    print(
      'Input shape: ${inputTensor.shape}',
    );

    print(
      'Output shape: ${outputTensor.shape}',
    );

    print(
      'Labels (${labels.length}): $labels',
    );

    print('==================================');

    if (inputTensor.shape.length != 2 ||
        inputTensor.shape[0] != 1 ||
        inputTensor.shape[1] != 63) {
      throw Exception(
        'Unexpected model input shape: '
        '${inputTensor.shape}',
      );
    }

    if (outputTensor.shape.length != 2 ||
        outputTensor.shape[0] != 1 ||
        outputTensor.shape[1] != 10) {
      throw Exception(
        'Unexpected model output shape: '
        '${outputTensor.shape}',
      );
    }

    if (labels.length != 10) {
      throw Exception(
        'Expected 10 labels but found '
        '${labels.length}',
      );
    }
  }

  void dispose() {
    print('Disposing GestureClassifier...');
    _interpreter?.close();
    _interpreter = null;
  }
}