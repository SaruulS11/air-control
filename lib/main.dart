import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:hand_landmarker/hand_landmarker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const AirControlApp());
}

class AirControlApp extends StatelessWidget {
  const AirControlApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: HandTrackingScreen(),
    );
  }
}

class HandTrackingScreen extends StatefulWidget {
  const HandTrackingScreen({super.key});

  @override
  State<HandTrackingScreen> createState() =>
      _HandTrackingScreenState();
}

class _HandTrackingScreenState extends State<HandTrackingScreen> {
  // ==========================================================
  // CAMERA + MEDIAPIPE
  // ==========================================================

  CameraController? _cameraController;
  HandLandmarkerPlugin? _handLandmarker;

  StreamSubscription<List<Hand>>? _handSubscription;

  bool _isReady = false;

  GestureClassifier? _gestureClassifier;

  // ==========================================================
  // LIVE GESTURE PREDICTION
  // ==========================================================

  String _predictedGesture = 'Waiting...';
  double _predictionConfidence = 0.0;

  DateTime? _lastPredictionTime;

  static const Duration _predictionInterval =
      Duration(milliseconds: 100);

  static const double _confidenceThreshold = 0.80;

  List<Hand> _hands = [];

  // ==========================================================
  // CURSOR
  // ==========================================================

  double? _smoothX;
  double? _smoothY;

  final double _smoothing = 0.25;

  // ==========================================================
  // DATASET
  // ==========================================================

  final TextEditingController _labelController =
      TextEditingController();

  final Map<String, List<List<double>>> _dataset = {};

  List<double>? _latestFeatures;

  static const int samplesPerGesture = 10;

  bool _recording = false;

  int _recordedThisRun = 0;

  String? _currentLabel;

  String _status = 'Starting...';

  DateTime? _lastSampleTime;

  List<double>? _lastRecordedFeatures;

  String? _csvPath;

  // ==========================================================
  // STARTUP
  // ==========================================================

  @override
  void initState() {
    super.initState();

    _initialize();
  }

  Future<void> _initialize() async {
    try {
      // ------------------------------------------------------
      // Prepare persistent CSV location
      // ------------------------------------------------------

      await _prepareCsvFile();

      // Load previous data automatically if file exists.
      await _loadExistingDataset();

      // ------------------------------------------------------
      // TensorFlow Lite gesture classifier
      // ------------------------------------------------------

      _gestureClassifier = GestureClassifier();
      await _gestureClassifier!.load();

      // ------------------------------------------------------
      // Camera
      // ------------------------------------------------------

      final cameras = await availableCameras();

      final frontCamera = cameras.firstWhere(
        (camera) =>
            camera.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      _cameraController = CameraController(
        frontCamera,
        ResolutionPreset.medium,
        enableAudio: false,
      );

      await _cameraController!.initialize();

      // ------------------------------------------------------
      // Hand Landmarker
      // ------------------------------------------------------

      _handLandmarker = HandLandmarkerPlugin.create(
        numHands: 1,
        minHandDetectionConfidence: 0.6,
        delegate: HandLandmarkerDelegate.cpu,
      );

      // Listen directly to every new detection result.
      _handSubscription =
          _handLandmarker!.landmarkStream.listen(_onHandsReceived);

      await _cameraController!.startImageStream(
        _processCameraImage,
      );

      if (!mounted) return;

      setState(() {
        _isReady = true;

        if (_dataset.isEmpty) {
          _status = 'Model loaded ✅. Ready. Enter a gesture name.';
        } else {
          _status =
              'Model loaded ✅. Loaded ${_dataset.length} gesture classes from disk.';
        }
      });
    } catch (e) {
      debugPrint('INITIALIZATION ERROR: $e');

      if (!mounted) return;

      setState(() {
        _status = 'Initialization error: $e';
      });
    }
  }

  // ==========================================================
  // FILE LOCATION
  // ==========================================================

  Future<void> _prepareCsvFile() async {
    Directory? directory;

    // Android app-specific external storage.
    //
    // Usually:
    //
    // /storage/emulated/0/Android/data/
    // com.example.air_control/files/
    //
    directory = await getExternalStorageDirectory();

    // Fallback.
    directory ??= await getApplicationDocumentsDirectory();

    _csvPath =
        '${directory.path}/gesture_final_test.csv';

    debugPrint('');
    debugPrint('======================================');
    debugPrint('DATASET PATH:');
    debugPrint(_csvPath);
    debugPrint('======================================');
    debugPrint('');
  }

  // ==========================================================
  // CAMERA → MEDIAPIPE
  // ==========================================================

  void _processCameraImage(CameraImage image) {
    if (!_isReady && _handLandmarker == null) {
      return;
    }

    if (_handLandmarker == null ||
        _cameraController == null) {
      return;
    }

    try {
      _handLandmarker!.processFrame(
        image,
        _cameraController!
            .description
            .sensorOrientation,
      );
    } catch (e) {
      debugPrint('Hand detection error: $e');
    }
  }

  // ==========================================================
  // NEW HAND FRAME
  // ==========================================================

  void _onHandsReceived(List<Hand> hands) {
    if (!mounted) return;

    _hands = hands;

    if (hands.isEmpty) {
      _latestFeatures = null;
      _predictedGesture = 'No hand';
      _predictionConfidence = 0.0;

      setState(() {});

      return;
    }

    _latestFeatures =
        _extractFeatures(hands.first);

    // --------------------------------------------------------
    // LIVE TFLITE PREDICTION
    // --------------------------------------------------------

    if (_latestFeatures != null) {
      _tryPredictGesture(
        _latestFeatures!,
      );
    }

    // --------------------------------------------------------
    // DATA RECORDING
    // --------------------------------------------------------

    if (_recording &&
        _latestFeatures != null &&
        _currentLabel != null) {
      _tryRecordSample(
        _latestFeatures!,
      );
    }

    setState(() {});
  }

  // ==========================================================
  // LIVE TFLITE PREDICTION
  // ==========================================================

  void _tryPredictGesture(
    List<double> features,
  ) {
    final classifier = _gestureClassifier;

    if (classifier == null ||
        !classifier.isLoaded) {
      return;
    }

    final now = DateTime.now();

    if (_lastPredictionTime != null &&
        now.difference(_lastPredictionTime!) <
            _predictionInterval) {
      return;
    }

    _lastPredictionTime = now;

    try {
      final prediction =
          classifier.predict(features);

      _predictionConfidence =
          prediction.confidence;

      if (prediction.confidence >=
          _confidenceThreshold) {
        _predictedGesture =
            prediction.label;
      } else {
        _predictedGesture = 'unknown';
      }
    } catch (e) {
      debugPrint(
        'Gesture prediction error: $e',
      );

      _predictedGesture = 'prediction error';
      _predictionConfidence = 0.0;
    }
  }

  // ==========================================================
  // FEATURE EXTRACTION
  // ==========================================================

  List<double>? _extractFeatures(Hand hand) {
    final landmarks = hand.landmarks;

    if (landmarks.length != 21) {
      return null;
    }

    // Landmark 0 = wrist
    final wrist = landmarks[0];

    // Landmark 9 = base of middle finger
    final middleBase = landmarks[9];

    final dx =
        middleBase.x - wrist.x;

    final dy =
        middleBase.y - wrist.y;

    final scale =
        math.sqrt(dx * dx + dy * dy);

    if (scale < 0.000001) {
      return null;
    }

    final features = <double>[];

    for (final landmark in landmarks) {
      // ------------------------------------------------------
      // TRANSLATION NORMALIZATION
      //
      // Wrist becomes origin 0,0,0.
      // ------------------------------------------------------

      final x =
          (landmark.x - wrist.x) /
              scale;

      final y =
          (landmark.y - wrist.y) /
              scale;

      final z =
          (landmark.z - wrist.z) /
              scale;

      features.add(x);
      features.add(y);
      features.add(z);
    }

    // 21 × 3 = 63 features
    return features;
  }

  // ==========================================================
  // RECORD SAMPLE
  // ==========================================================

  void _tryRecordSample(
    List<double> features,
  ) {
    final now = DateTime.now();

    // Only save around 10 samples per second.
    //
    // 60 samples ≈ 6 seconds.
    if (_lastSampleTime != null) {
      final difference =
          now.difference(_lastSampleTime!);

      if (difference.inMilliseconds < 100) {
        return;
      }
    }

    // Skip almost-identical consecutive rows.
    if (_lastRecordedFeatures != null &&
        _almostIdentical(
          features,
          _lastRecordedFeatures!,
        )) {
      return;
    }

    _lastSampleTime = now;

    _lastRecordedFeatures =
        List<double>.from(features);

    final samples =
        _dataset.putIfAbsent(
      _currentLabel!,
      () => [],
    );

    samples.add(
      List<double>.from(features),
    );

    _recordedThisRun++;

    // --------------------------------------------------------
    // Finished
    // --------------------------------------------------------

    if (_recordedThisRun >=
        samplesPerGesture) {
      final finishedLabel =
          _currentLabel!;

      _recording = false;

      _currentLabel = null;

      _finishRecording(
        finishedLabel,
      );
    }
  }

  // ==========================================================
  // DUPLICATE DETECTION
  // ==========================================================

  bool _almostIdentical(
    List<double> a,
    List<double> b,
  ) {
    if (a.length != b.length) {
      return false;
    }

    double totalDifference = 0;

    for (int i = 0; i < a.length; i++) {
      totalDifference +=
          (a[i] - b[i]).abs();
    }

    final averageDifference =
        totalDifference / a.length;

    // Only skip extremely similar frames.
    return averageDifference < 0.0005;
  }

  // ==========================================================
  // START RECORDING
  // ==========================================================

  Future<void> _startRecording() async {
    if (_recording) return;

    final label =
        _labelController.text
            .trim()
            .toLowerCase()
            .replaceAll(' ', '_')
            .replaceAll(',', '_');

    if (label.isEmpty) {
      setState(() {
        _status =
            'Enter a gesture name first.';
      });

      return;
    }

    if (_hands.isEmpty ||
        _latestFeatures == null) {
      setState(() {
        _status =
            'Show your hand to the camera first.';
      });

      return;
    }

    // --------------------------------------------------------
    // Countdown
    // --------------------------------------------------------

    for (int i = 3; i >= 1; i--) {
      if (!mounted) return;

      setState(() {
        _status =
            'Recording "$label" starts in $i...';
      });

      await Future.delayed(
        const Duration(seconds: 1),
      );
    }

    _recordedThisRun = 0;

    _lastSampleTime = null;

    _lastRecordedFeatures = null;

    _currentLabel = label;

    setState(() {
      _recording = true;

      _status =
          'Recording $label... Move your hand slightly.';
    });
  }

  // ==========================================================
  // FINISH RECORDING
  // ==========================================================

  Future<void> _finishRecording(
    String label,
  ) async {
    // AUTOSAVE IMMEDIATELY.
    await _saveDatasetToCsv();

    if (!mounted) return;

    setState(() {
      _status =
          '✅ "$label" saved. '
          '$_recordedThisRun samples recorded.';
    });
  }

  // ==========================================================
  // SAVE CSV
  // ==========================================================

  Future<void> _saveDatasetToCsv() async {
    if (_csvPath == null) {
      await _prepareCsvFile();
    }

    final buffer = StringBuffer();

    // --------------------------------------------------------
    // CSV HEADER
    // --------------------------------------------------------

    buffer.write('label');

    for (int i = 0; i < 63; i++) {
      buffer.write(',f$i');
    }

    buffer.writeln();

    // --------------------------------------------------------
    // DATA
    // --------------------------------------------------------

    int rows = 0;

    for (final entry in _dataset.entries) {
      final label = entry.key;

      for (final sample in entry.value) {
        buffer.write(label);

        for (final value in sample) {
          buffer.write(
            ',${value.toStringAsFixed(8)}',
          );
        }

        buffer.writeln();

        rows++;
      }
    }

    final file = File(_csvPath!);

    await file.writeAsString(
      buffer.toString(),
      flush: true,
    );

    debugPrint('');
    debugPrint('======================================');
    debugPrint('DATASET SAVED');
    debugPrint('Rows: $rows');
    debugPrint('Classes: ${_dataset.length}');
    debugPrint('Path: $_csvPath');
    debugPrint('======================================');
    debugPrint('');
  }

  // ==========================================================
  // MANUAL SAVE BUTTON
  // ==========================================================

  Future<void> _manualSave() async {
    if (_dataset.isEmpty) {
      setState(() {
        _status = 'Dataset is empty.';
      });

      return;
    }

    await _saveDatasetToCsv();

    if (!mounted) return;

    setState(() {
      _status =
          '✅ Dataset saved permanently.';
    });
  }

  // ==========================================================
  // LOAD EXISTING CSV
  // ==========================================================

  Future<void> _loadExistingDataset() async {
    if (_csvPath == null) return;

    final file = File(_csvPath!);

    if (!await file.exists()) {
      return;
    }

    try {
      final lines =
          await file.readAsLines();

      if (lines.length <= 1) {
        return;
      }

      for (int i = 1;
          i < lines.length;
          i++) {
        final line =
            lines[i].trim();

        if (line.isEmpty) {
          continue;
        }

        final parts =
            line.split(',');

        // label + 63 features = 64 columns
        if (parts.length != 64) {
          continue;
        }

        final label =
            parts.first;

        final values =
            <double>[];

        bool valid = true;

        for (int j = 1;
            j < parts.length;
            j++) {
          final value =
              double.tryParse(
            parts[j],
          );

          if (value == null) {
            valid = false;
            break;
          }

          values.add(value);
        }

        if (!valid ||
            values.length != 63) {
          continue;
        }

        _dataset
            .putIfAbsent(
          label,
          () => [],
        )
            .add(values);
      }

      debugPrint(
        'Loaded ${_dataset.length} classes from CSV.',
      );
    } catch (e) {
      debugPrint(
        'Could not load existing dataset: $e',
      );
    }
  }

  // ==========================================================
  // CLEAR DATASET
  // ==========================================================

  Future<void> _clearDataset() async {
    _recording = false;

    _currentLabel = null;

    _recordedThisRun = 0;

    _dataset.clear();

    if (_csvPath != null) {
      final file =
          File(_csvPath!);

      if (await file.exists()) {
        await file.delete();
      }
    }

    if (!mounted) return;

    setState(() {
      _status =
          'Dataset cleared. Fresh start.';
    });
  }

  // ==========================================================
  // CURSOR
  // ==========================================================

  Offset? _getSmoothedCursor() {
    if (_hands.isEmpty) {
      _smoothX = null;
      _smoothY = null;

      return null;
    }

    final landmarks =
        _hands.first.landmarks;

    if (landmarks.length != 21) {
      return null;
    }

    // Index fingertip.
    final indexTip =
        landmarks[8];

    final rawX =
        1 - indexTip.x;

    final rawY =
        indexTip.y;

    if (_smoothX == null ||
        _smoothY == null) {
      _smoothX = rawX;
      _smoothY = rawY;
    } else {
      _smoothX =
          _smoothX! +
          _smoothing *
              (rawX -
                  _smoothX!);

      _smoothY =
          _smoothY! +
          _smoothing *
              (rawY -
                  _smoothY!);
    }

    return Offset(
      _smoothX!,
      _smoothY!,
    );
  }

  // ==========================================================
  // DATASET INFO
  // ==========================================================

  int get _totalSamples {
    int total = 0;

    for (final samples
        in _dataset.values) {
      total += samples.length;
    }

    return total;
  }

  // ==========================================================
  // DISPOSE
  // ==========================================================

  @override
  void dispose() {
    _handSubscription?.cancel();

    if (_cameraController
            ?.value
            .isStreamingImages ??
        false) {
      _cameraController
          ?.stopImageStream();
    }

    _cameraController?.dispose();

    _handLandmarker?.dispose();

    _gestureClassifier?.dispose();

    _labelController.dispose();

    super.dispose();
  }

  // ==========================================================
  // UI
  // ==========================================================

  @override
  Widget build(
    BuildContext context,
  ) {
    if (!_isReady ||
        _cameraController == null ||
        !_cameraController!
            .value
            .isInitialized) {
      return Scaffold(
        backgroundColor: Colors.black,
        body: Center(
          child: Column(
            mainAxisSize:
                MainAxisSize.min,
            children: [
              const CircularProgressIndicator(),

              const SizedBox(
                height: 20,
              ),

              Padding(
                padding:
                    const EdgeInsets.all(
                  20,
                ),
                child: Text(
                  _status,
                  textAlign:
                      TextAlign.center,
                  style:
                      const TextStyle(
                    color:
                        Colors.white,
                  ),
                ),
              ),
            ],
          ),
        ),
      );
    }

    final cursor =
        _getSmoothedCursor();

    return Scaffold(
      resizeToAvoidBottomInset: true,
      backgroundColor: Colors.black,

      body: Stack(
        fit: StackFit.expand,
        children: [
          // ==================================================
          // CAMERA
          // ==================================================

          CameraPreview(
            _cameraController!,
          ),

          // ==================================================
          // HAND + CURSOR
          // ==================================================

          CustomPaint(
            painter: HandPainter(
              hands: _hands,
              cursor: cursor,
            ),
          ),

          // ==================================================
          // STATUS TOP
          // ==================================================

          SafeArea(
            child: Align(
              alignment:
                  Alignment.topLeft,
              child: Container(
                margin:
                    const EdgeInsets.all(
                  12,
                ),
                padding:
                    const EdgeInsets.symmetric(
                  horizontal: 15,
                  vertical: 10,
                ),
                decoration:
                    BoxDecoration(
                  color:
                      Colors.black87,
                  borderRadius:
                      BorderRadius.circular(
                    14,
                  ),
                ),
                child: Text(
                  _hands.isEmpty
                      ? 'No hand detected'
                      : 'Hand detected ✅',
                  style:
                      const TextStyle(
                    color:
                        Colors.white,
                    fontSize: 17,
                  ),
                ),
              ),
            ),
          ),

          // ==================================================
          // LIVE GESTURE PREDICTION
          // ==================================================

          SafeArea(
            child: Align(
              alignment:
                  Alignment.topRight,
              child: Container(
                width: 190,
                margin:
                    const EdgeInsets.all(
                  12,
                ),
                padding:
                    const EdgeInsets.symmetric(
                  horizontal: 15,
                  vertical: 12,
                ),
                decoration:
                    BoxDecoration(
                  color:
                      Colors.black87,
                  borderRadius:
                      BorderRadius.circular(
                    14,
                  ),
                  border: Border.all(
                    color: Colors.white24,
                  ),
                ),
                child: Column(
                  mainAxisSize:
                      MainAxisSize.min,
                  crossAxisAlignment:
                      CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'LIVE GESTURE',
                      style: TextStyle(
                        color: Colors.white54,
                        fontSize: 11,
                        fontWeight:
                            FontWeight.bold,
                        letterSpacing: 1.2,
                      ),
                    ),
                    const SizedBox(
                      height: 5,
                    ),
                    Text(
                      _predictedGesture
                          .replaceAll('_', ' ')
                          .toUpperCase(),
                      maxLines: 1,
                      overflow:
                          TextOverflow.ellipsis,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight:
                            FontWeight.bold,
                      ),
                    ),
                    const SizedBox(
                      height: 4,
                    ),
                    Text(
                      _hands.isEmpty
                          ? 'Confidence: --'
                          : 'Confidence: ${(_predictionConfidence * 100).toStringAsFixed(1)}%',
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 13,
                      ),
                    ),
                    const SizedBox(
                      height: 3,
                    ),
                    const Text(
                      'Threshold: 80%',
                      style: TextStyle(
                        color: Colors.white38,
                        fontSize: 11,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // ==================================================
          // BOTTOM PANEL
          // ==================================================

          Align(
            alignment:
                Alignment.bottomCenter,

            child: SafeArea(
              child: Container(
                width:
                    double.infinity,

                constraints:
                    const BoxConstraints(
                  maxHeight: 380,
                ),

                margin:
                    const EdgeInsets.all(
                  12,
                ),

                padding:
                    const EdgeInsets.all(
                  14,
                ),

                decoration:
                    BoxDecoration(
                  color:
                      Colors.black87,

                  borderRadius:
                      BorderRadius.circular(
                    20,
                  ),
                ),

                child:
                    SingleChildScrollView(
                  child: Column(
                    mainAxisSize:
                        MainAxisSize.min,
                    children: [
                      const Text(
                        'Custom Gesture Dataset',
                        style:
                            TextStyle(
                          color:
                              Colors.white,
                          fontSize:
                              20,
                          fontWeight:
                              FontWeight.bold,
                        ),
                      ),

                      const SizedBox(
                        height: 6,
                      ),

                      Text(
                        '${_dataset.length} gestures • '
                        '$_totalSamples samples',
                        style:
                            const TextStyle(
                          color:
                              Colors.white70,
                        ),
                      ),

                      const SizedBox(
                        height: 12,
                      ),

                      // --------------------------------------
                      // LABEL
                      // --------------------------------------

                      TextField(
                        controller:
                            _labelController,

                        enabled:
                            !_recording,

                        style:
                            const TextStyle(
                          color:
                              Colors.white,
                        ),

                        decoration:
                            InputDecoration(
                          hintText:
                              'Gesture name e.g. fist',
                          hintStyle:
                              const TextStyle(
                            color:
                                Colors.white54,
                          ),

                          filled: true,

                          fillColor:
                              Colors.white12,

                          border:
                              OutlineInputBorder(
                            borderRadius:
                                BorderRadius.circular(
                              12,
                            ),
                          ),
                        ),
                      ),

                      const SizedBox(
                        height: 10,
                      ),

                      // --------------------------------------
                      // RECORD
                      // --------------------------------------

                      SizedBox(
                        width:
                            double.infinity,

                        child:
                            ElevatedButton.icon(
                          onPressed:
                              _recording
                                  ? null
                                  : _startRecording,

                          icon:
                              const Icon(
                            Icons.fiber_manual_record,
                          ),

                          label:
                              Text(
                            _recording
                                ? 'RECORDING $_recordedThisRun / $samplesPerGesture'
                                : 'RECORD $samplesPerGesture SAMPLES',
                          ),
                        ),
                      ),

                      const SizedBox(
                        height: 8,
                      ),

                      // --------------------------------------
                      // SAVE
                      // --------------------------------------

                      SizedBox(
                        width:
                            double.infinity,

                        child:
                            ElevatedButton.icon(
                          onPressed:
                              _dataset.isEmpty
                                  ? null
                                  : _manualSave,

                          icon:
                              const Icon(
                            Icons.save,
                          ),

                          label:
                              const Text(
                            'SAVE DATASET NOW',
                          ),
                        ),
                      ),

                      const SizedBox(
                        height: 8,
                      ),

                      // --------------------------------------
                      // CLEAR
                      // --------------------------------------

                      SizedBox(
                        width:
                            double.infinity,

                        child:
                            OutlinedButton.icon(
                          onPressed:
                              _clearDataset,

                          icon:
                              const Icon(
                            Icons.delete_forever,
                          ),

                          label:
                              const Text(
                            'CLEAR DATASET',
                          ),
                        ),
                      ),

                      const SizedBox(
                        height: 10,
                      ),

                      Text(
                        _status,
                        textAlign:
                            TextAlign.center,
                        style:
                            const TextStyle(
                          color:
                              Colors.white,
                        ),
                      ),

                      if (_recording) ...[
                        const SizedBox(
                          height: 10,
                        ),

                        LinearProgressIndicator(
                          value:
                              _recordedThisRun /
                                  samplesPerGesture,
                        ),
                      ],

                      if (_dataset
                          .isNotEmpty) ...[
                        const SizedBox(
                          height: 12,
                        ),

                        const Divider(
                          color:
                              Colors.white24,
                        ),

                        ..._dataset.entries.map(
                          (entry) =>
                              Padding(
                            padding:
                                const EdgeInsets.symmetric(
                              vertical: 2,
                            ),
                            child: Text(
                              '${entry.key}: '
                              '${entry.value.length} samples',
                              style:
                                  const TextStyle(
                                color:
                                    Colors.white70,
                              ),
                            ),
                          ),
                        ),
                      ],

                      if (_csvPath !=
                          null) ...[
                        const SizedBox(
                          height: 12,
                        ),

                        const Divider(
                          color:
                              Colors.white24,
                        ),

                        const Text(
                          'Saved file:',
                          style:
                              TextStyle(
                            color:
                                Colors.white54,
                            fontSize:
                                12,
                          ),
                        ),

                        const SizedBox(
                          height: 4,
                        ),

                        Text(
                          _csvPath!,
                          textAlign:
                              TextAlign.center,
                          style:
                              const TextStyle(
                            color:
                                Colors.white54,
                            fontSize:
                                10,
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================
// HAND PAINTER
// ============================================================

class HandPainter extends CustomPainter {
  final List<Hand> hands;

  final Offset? cursor;

  HandPainter({
    required this.hands,
    required this.cursor,
  });

  @override
  void paint(
    Canvas canvas,
    Size size,
  ) {
    final pointPaint = Paint()
      ..color = Colors.green
      ..style =
          PaintingStyle.fill;

    final linePaint = Paint()
      ..color = Colors.green
      ..strokeWidth = 3;

    const connections = [
      // Thumb
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],

      // Index
      [0, 5],
      [5, 6],
      [6, 7],
      [7, 8],

      // Middle
      [5, 9],
      [9, 10],
      [10, 11],
      [11, 12],

      // Ring
      [9, 13],
      [13, 14],
      [14, 15],
      [15, 16],

      // Pinky
      [13, 17],
      [17, 18],
      [18, 19],
      [19, 20],

      // Palm
      [0, 17],
    ];

    // --------------------------------------------------------
    // DRAW HAND
    // --------------------------------------------------------

    for (final hand in hands) {
      final landmarks =
          hand.landmarks;

      if (landmarks.length != 21) {
        continue;
      }

      for (final connection
          in connections) {
        final start =
            landmarks[
                connection[0]];

        final end =
            landmarks[
                connection[1]];

        final startPoint =
            Offset(
          (1 - start.x) *
              size.width,
          start.y *
              size.height,
        );

        final endPoint =
            Offset(
          (1 - end.x) *
              size.width,
          end.y *
              size.height,
        );

        canvas.drawLine(
          startPoint,
          endPoint,
          linePaint,
        );
      }

      for (final landmark
          in landmarks) {
        final point =
            Offset(
          (1 - landmark.x) *
              size.width,
          landmark.y *
              size.height,
        );

        canvas.drawCircle(
          point,
          6,
          pointPaint,
        );
      }
    }

    // --------------------------------------------------------
    // CURSOR
    // --------------------------------------------------------

    if (cursor != null) {
      final point =
          Offset(
        cursor!.dx *
            size.width,
        cursor!.dy *
            size.height,
      );

      final cursorPaint =
          Paint()
            ..color =
                Colors.red
            ..style =
                PaintingStyle.fill;

      final borderPaint =
          Paint()
            ..color =
                Colors.white
            ..style =
                PaintingStyle.stroke
            ..strokeWidth = 3;

      canvas.drawCircle(
        point,
        18,
        cursorPaint,
      );

      canvas.drawCircle(
        point,
        18,
        borderPaint,
      );
    }
  }

  @override
  bool shouldRepaint(
    covariant HandPainter oldDelegate,
  ) {
    return true;
  }
}

// ============================================================
// TFLITE GESTURE CLASSIFIER
// ============================================================

class GestureClassifier {
  Interpreter? _interpreter;

  List<String> labels = [];

  bool get isLoaded => _interpreter != null;

  Future<void> load() async {
    // Load the TensorFlow Lite model from Flutter assets.
    _interpreter = await Interpreter.fromAsset(
      'assets/models/gesture_model.tflite',
    );

    // Load the gesture labels.
    final labelText = await rootBundle.loadString(
      'assets/models/gesture_labels.txt',
    );

    labels = labelText
        .split('\n')
        .map((label) => label.trim())
        .where((label) => label.isNotEmpty)
        .toList();

    final inputTensor = _interpreter!.getInputTensor(0);
    final outputTensor = _interpreter!.getOutputTensor(0);

    debugPrint('');
    debugPrint('========== TFLITE MODEL ==========');
    debugPrint('Input shape: ${inputTensor.shape}');
    debugPrint('Output shape: ${outputTensor.shape}');
    debugPrint('Labels (${labels.length}): $labels');
    debugPrint('==================================');
    debugPrint('');

    // Your trained model expects 63 landmark features.
    if (inputTensor.shape.length != 2 ||
        inputTensor.shape[0] != 1 ||
        inputTensor.shape[1] != 63) {
      throw Exception(
        'Unexpected model input shape: ${inputTensor.shape}',
      );
    }

    // Your trained model outputs probabilities for 10 gestures.
    if (outputTensor.shape.length != 2 ||
        outputTensor.shape[0] != 1 ||
        outputTensor.shape[1] != 10) {
      throw Exception(
        'Unexpected model output shape: ${outputTensor.shape}',
      );
    }

    if (labels.length != 10) {
      throw Exception(
        'Expected 10 labels but found ${labels.length}',
      );
    }
  }

  GesturePrediction predict(
    List<double> features,
  ) {
    final interpreter = _interpreter;

    if (interpreter == null) {
      throw StateError(
        'Gesture model is not loaded.',
      );
    }

    if (features.length != 63) {
      throw ArgumentError(
        'Expected 63 features but got ${features.length}.',
      );
    }

    if (labels.length != 10) {
      throw StateError(
        'Expected 10 labels but found ${labels.length}.',
      );
    }

    final input = <List<double>>[
      List<double>.from(features),
    ];

    final output = <List<double>>[
      List<double>.filled(
        labels.length,
        0.0,
      ),
    ];

    interpreter.run(
      input,
      output,
    );

    final probabilities = output.first;

    int bestIndex = 0;
    double bestConfidence =
        probabilities[0];

    for (int i = 1;
        i < probabilities.length;
        i++) {
      if (probabilities[i] >
          bestConfidence) {
        bestConfidence =
            probabilities[i];
        bestIndex = i;
      }
    }

    return GesturePrediction(
      label: labels[bestIndex],
      confidence: bestConfidence,
    );
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}

class GesturePrediction {
  final String label;
  final double confidence;

  const GesturePrediction({
    required this.label,
    required this.confidence,
  });
}
