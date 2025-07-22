import 'dart:async';
import 'dart:typed_data' show Uint8List;

import "package:logging/logging.dart";
import "package:photos/models/ml/face/box.dart";
import "package:photos/services/isolate_functions.dart";
import "package:photos/services/isolate_service.dart";
import "package:photos/utils/image_ml_util.dart";

@pragma('vm:entry-point')
class FaceThumbnailGenerator extends SuperIsolate {
  @override
  Logger get logger => _logger;
  final _logger = Logger('FaceThumbnailGenerator');

  @override
  bool get isDartUiIsolate => true;

  @override
  String get isolateName => "FaceThumbnailGenerator";

  @override
  bool get shouldAutomaticDispose => true;

  // Singleton pattern
  FaceThumbnailGenerator._privateConstructor();
  static final FaceThumbnailGenerator instance =
      FaceThumbnailGenerator._privateConstructor();
  factory FaceThumbnailGenerator() => instance;

  /// Generates face thumbnails for all [faceBoxes] in [imageData].
  ///
  /// Uses [generateFaceThumbnailsUsingCanvas] inside the isolate.
  Future<List<Uint8List>> generateFaceThumbnails(
    String imagePath,
    List<FaceBox> faceBoxes, {
    int? fileID,
  }) async {
    try {
      _logger.info(
        "[$fileID] FaceThumbnailGenerator START: Generating face thumbnails for ${faceBoxes.length} face boxes in $imagePath",
      );

      _logger
          .info("[$fileID] Converting ${faceBoxes.length} face boxes to JSON");
      final List<Map<String, dynamic>> faceBoxesJson =
          faceBoxes.map((box) => box.toJson()).toList();
      _logger.info(
        "[$fileID] Face boxes converted to JSON, preparing isolate operation",
      );

      _logger.info(
        "[$fileID] Running generateFaceThumbnails in isolate with imagePath: $imagePath",
      );
      final stopwatch = Stopwatch()..start();

      final List<Uint8List> faces = await runInIsolate(
        IsolateOperation.generateFaceThumbnails,
        {
          'imagePath': imagePath,
          'faceBoxesList': faceBoxesJson,
        },
      ).then((value) => value.cast<Uint8List>());

      stopwatch.stop();
      _logger.info(
        "[$fileID] Isolate operation completed in ${stopwatch.elapsedMilliseconds}ms",
      );
      _logger.info(
        "[$fileID] Generated ${faces.length} face thumbnails with sizes: ${faces.map((e) => '${(e.length / 1024).toStringAsFixed(1)}KB').toList()}",
      );

      if (faces.length != faceBoxes.length) {
        _logger.severe(
          "[$fileID] Mismatch: Expected ${faceBoxes.length} face thumbnails but got ${faces.length}",
        );
      }

      _logger.info("[$fileID] Starting face thumbnail compression");
      final compressionStopwatch = Stopwatch()..start();

      final compressedFaces =
          await compressFaceThumbnails({'listPngBytes': faces}, fileID: fileID);

      compressionStopwatch.stop();
      _logger.info(
        "[$fileID] Compression completed in ${compressionStopwatch.elapsedMilliseconds}ms",
      );
      _logger.info(
        "[$fileID] Compressed face thumbnails from sizes ${faces.map((e) => '${(e.length / 1024).toStringAsFixed(1)}KB').toList()} to ${compressedFaces.map((e) => '${(e.length / 1024).toStringAsFixed(1)}KB').toList()}",
      );

      final totalOriginalSize =
          faces.fold<int>(0, (sum, face) => sum + face.length);
      final totalCompressedSize =
          compressedFaces.fold<int>(0, (sum, face) => sum + face.length);
      final compressionRatio = totalOriginalSize > 0
          ? (totalCompressedSize / totalOriginalSize * 100).toStringAsFixed(1)
          : "0.0";

      _logger.info(
        "[$fileID] FaceThumbnailGenerator END: Total size reduced from ${(totalOriginalSize / 1024).toStringAsFixed(1)}KB to ${(totalCompressedSize / 1024).toStringAsFixed(1)}KB (${compressionRatio}% of original)",
      );

      return compressedFaces;
    } catch (e, s) {
      _logger.severe(
        "[$fileID] Failed to generate face thumbnails for $imagePath",
        e,
        s,
      );
      rethrow;
    }
  }
}
