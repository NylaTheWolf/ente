import "dart:async";
import "dart:io" show File;
import "dart:math" show pow;

import "package:flutter/foundation.dart";
import "package:logging/logging.dart";
import "package:photos/core/cache/lru_map.dart";
import "package:photos/db/files_db.dart";
import "package:photos/db/ml/db.dart";
import "package:photos/extensions/stop_watch.dart";
import "package:photos/models/file/file.dart";
import "package:photos/models/file/file_type.dart";
import "package:photos/models/ml/face/box.dart";
import "package:photos/models/ml/face/face.dart";
import "package:photos/services/machine_learning/face_thumbnail_generator.dart";
import "package:photos/utils/file_util.dart";
import "package:photos/utils/standalone/task_queue.dart";
import "package:photos/utils/thumbnail_util.dart";

final _logger = Logger("FaceCropUtils");

const int _retryLimit = 3;
final LRUMap<String, Uint8List?> _faceCropCache = LRUMap(100);
final LRUMap<String, Uint8List?> _faceCropThumbnailCache = LRUMap(100);

final LRUMap<String, String> _personOrClusterIdToCachedFaceID = LRUMap(2000);

TaskQueue _queueFullFileFaceGenerations = TaskQueue<String>(
  maxConcurrentTasks: 5,
  taskTimeout: const Duration(minutes: 1),
  maxQueueSize: 100,
);
TaskQueue _queueThumbnailFaceGenerations = TaskQueue<String>(
  maxConcurrentTasks: 5,
  taskTimeout: const Duration(minutes: 1),
  maxQueueSize: 100,
);

Uint8List? checkInMemoryCachedCropForPersonOrClusterID(
  String personOrClusterID,
) {
  final String? faceID =
      _personOrClusterIdToCachedFaceID.get(personOrClusterID);
  if (faceID == null) return null;
  final Uint8List? cachedCover = _faceCropCache.get(faceID);
  return cachedCover;
}

Uint8List? _checkInMemoryCachedCropForFaceID(String faceID, int? fileID) {
  _logger.info("[$fileID] Checking in-memory cache for faceID: $faceID");
  final Uint8List? cachedCover = _faceCropCache.get(faceID);
  _logger.info(
    "[$fileID] In-memory cache result for faceID $faceID: ${cachedCover != null ? 'found (${cachedCover.length} bytes)' : 'not found'}",
  );
  return cachedCover;
}

Future<String?> checkUsedFaceIDForPersonOrClusterId(
  String personOrClusterID,
) async {
  final String? cachedFaceID =
      _personOrClusterIdToCachedFaceID.get(personOrClusterID);
  if (cachedFaceID != null) return cachedFaceID;
  final String? faceIDFromDB = await MLDataDB.instance
      .getFaceIdUsedForPersonOrCluster(personOrClusterID);
  if (faceIDFromDB != null) {
    _personOrClusterIdToCachedFaceID.put(personOrClusterID, faceIDFromDB);
  }
  return faceIDFromDB;
}

Future<void> putFaceIdCachedForPersonOrCluster(
  String personOrClusterID,
  String faceID,
) async {
  await MLDataDB.instance.putFaceIdCachedForPersonOrCluster(
    personOrClusterID,
    faceID,
  );
  _personOrClusterIdToCachedFaceID.put(personOrClusterID, faceID);
}

Future<void> _putCachedCropForFaceID(
  String faceID,
  Uint8List data, [
  String? personOrClusterID,
  int? fileID,
]) async {
  _logger.info(
    "[$fileID] _putCachedCropForFaceID: faceID=$faceID, dataSize=${data.length} bytes, personOrClusterID=$personOrClusterID",
  );
  _faceCropCache.put(faceID, data);
  _logger.info("[$fileID] Put faceID $faceID in memory cache");
  if (personOrClusterID != null) {
    _logger.info(
      "[$fileID] Putting faceID $faceID for personOrClusterID $personOrClusterID in DB",
    );
    await putFaceIdCachedForPersonOrCluster(personOrClusterID, faceID);
    _logger.info(
      "[$fileID] Successfully put faceID $faceID for personOrClusterID $personOrClusterID in DB",
    );
  }
}

Future<void> checkRemoveCachedFaceIDForPersonOrClusterId(
  String personOrClusterID,
) async {
  final String? cachedFaceID = await MLDataDB.instance
      .getFaceIdUsedForPersonOrCluster(personOrClusterID);
  if (cachedFaceID != null) {
    _personOrClusterIdToCachedFaceID.remove(personOrClusterID);
    await MLDataDB.instance
        .removeFaceIdCachedForPersonOrCluster(personOrClusterID);
  }
}

/// Careful to only use [personOrClusterID] if all [faces] are from the same person or cluster.
Future<Map<String, Uint8List>?> getCachedFaceCrops(
  EnteFile enteFile,
  Iterable<Face> faces, {
  int fetchAttempt = 1,
  bool useFullFile = true,
  String? personOrClusterID,
  required bool useTempCache,
}) async {
  _logger.info(
    "[${enteFile.uploadedFileID}] getCachedFaceCrops START: faces=${faces.length}, fetchAttempt=$fetchAttempt, useFullFile=$useFullFile, personOrClusterID=$personOrClusterID, useTempCache=$useTempCache",
  );
  try {
    final faceIdToCrop = <String, Uint8List>{};
    final facesWithoutCrops = <String, FaceBox>{};
    _logger.info(
      "[${enteFile.uploadedFileID}] Initialized maps: faceIdToCrop and facesWithoutCrops",
    );
    _logger.info(
      "[${enteFile.uploadedFileID}] Processing ${faces.length} faces with IDs: ${faces.map((f) => f.faceID).toList()}",
    );

    // Validate faces have valid detection boxes
    for (final face in faces) {
      if (face.detection.box.width <= 0 || face.detection.box.height <= 0) {
        _logger.severe(
          "[${enteFile.uploadedFileID}] Face ${face.faceID} has invalid detection box: ${face.detection.box}",
        );
      }
    }

    for (final face in faces) {
      _logger.info(
        "[${enteFile.uploadedFileID}] face ${face.faceID} has score ${face.score}",
      );
      _logger
          .info("[${enteFile.uploadedFileID}] Processing face ${face.faceID}");
      final Uint8List? cachedFace = _checkInMemoryCachedCropForFaceID(
        face.faceID,
        enteFile.uploadedFileID,
      );
      if (cachedFace != null) {
        _logger.info(
          "[${enteFile.uploadedFileID}] Found face ${face.faceID} in memory cache, size: ${cachedFace.length} bytes",
        );
        faceIdToCrop[face.faceID] = cachedFace;
      } else {
        _logger.info(
          "[${enteFile.uploadedFileID}] Face ${face.faceID} not in memory cache, checking file cache",
        );
        final faceCropCacheFile = cachedFaceCropPath(
          face.faceID,
          useTempCache,
          enteFile.uploadedFileID,
        );
        _logger.info(
          "[${enteFile.uploadedFileID}] Cache file path for face ${face.faceID}: ${faceCropCacheFile.path}",
        );

        // Check if cache directory exists
        final cacheDir = faceCropCacheFile.parent;
        if (!(await cacheDir.exists())) {
          _logger.severe(
            "[${enteFile.uploadedFileID}] Cache directory does not exist: ${cacheDir.path}",
          );
          try {
            await cacheDir.create(recursive: true);
            _logger.info(
              "[${enteFile.uploadedFileID}] Created cache directory: ${cacheDir.path}",
            );
          } catch (e, s) {
            _logger.severe(
              "[${enteFile.uploadedFileID}] Failed to create cache directory: ${cacheDir.path}",
              e,
              s,
            );
            facesWithoutCrops[face.faceID] = face.detection.box;
            continue;
          }
        }

        if ((await faceCropCacheFile.exists())) {
          _logger.info(
            "[${enteFile.uploadedFileID}] Cache file exists for face ${face.faceID}, attempting to read",
          );
          try {
            final data = await faceCropCacheFile.readAsBytes();
            _logger.info(
              "[${enteFile.uploadedFileID}] Read ${data.length} bytes from cache file for face ${face.faceID}",
            );
            if (data.isNotEmpty) {
              _logger.info(
                "[${enteFile.uploadedFileID}] Cache file data is valid for face ${face.faceID}, putting in memory cache",
              );
              await _putCachedCropForFaceID(
                face.faceID,
                data,
                personOrClusterID,
                enteFile.uploadedFileID,
              );
              faceIdToCrop[face.faceID] = data;
              _logger.info(
                "[${enteFile.uploadedFileID}] Successfully cached face ${face.faceID} from file",
              );
            } else {
              _logger.severe(
                "[${enteFile.uploadedFileID}] Cached face crop for faceID ${face.faceID} is empty, deleting file ${faceCropCacheFile.path}",
              );
              await faceCropCacheFile.delete();
              facesWithoutCrops[face.faceID] = face.detection.box;
              _logger.info(
                "[${enteFile.uploadedFileID}] Added face ${face.faceID} to facesWithoutCrops due to empty cache file",
              );
            }
          } catch (e, s) {
            _logger.severe(
              "[${enteFile.uploadedFileID}] Error reading cached face crop for faceID ${face.faceID} from file ${faceCropCacheFile.path}",
              e,
              s,
            );
            facesWithoutCrops[face.faceID] = face.detection.box;
            _logger.info(
              "[${enteFile.uploadedFileID}] Added face ${face.faceID} to facesWithoutCrops due to read error",
            );
          }
        } else {
          _logger.info(
            "[${enteFile.uploadedFileID}] Cache file does not exist for face ${face.faceID}",
          );
          facesWithoutCrops[face.faceID] = face.detection.box;
          _logger.info(
            "[${enteFile.uploadedFileID}] Added face ${face.faceID} to facesWithoutCrops",
          );
        }
      }
    }
    _logger.info(
      "[${enteFile.uploadedFileID}] Cache check complete: ${faceIdToCrop.length} faces found in cache, ${facesWithoutCrops.length} faces need generation",
    );
    if (facesWithoutCrops.isEmpty) {
      _logger.info(
        "[${enteFile.uploadedFileID}] All face crops gotten from cache, returning ${faceIdToCrop.length} crops",
      );
      return faceIdToCrop;
    }
    _logger.info(
      "[${enteFile.uploadedFileID}] Faces without crops: ${facesWithoutCrops.keys.toList()}",
    );

    if (!useFullFile) {
      _logger.info(
        "[${enteFile.uploadedFileID}] Using thumbnail mode, checking thumbnail cache for ${facesWithoutCrops.length} faces",
      );
      for (final face in faces) {
        if (facesWithoutCrops.containsKey(face.faceID)) {
          final Uint8List? cachedFaceThumbnail =
              _faceCropThumbnailCache.get(face.faceID);
          if (cachedFaceThumbnail != null) {
            _logger.info(
              "[${enteFile.uploadedFileID}] Found face ${face.faceID} in thumbnail cache, size: ${cachedFaceThumbnail.length} bytes",
            );
            faceIdToCrop[face.faceID] = cachedFaceThumbnail;
            facesWithoutCrops.remove(face.faceID);
          } else {
            _logger.info(
              "[${enteFile.uploadedFileID}] Face ${face.faceID} not found in thumbnail cache",
            );
          }
        }
      }
      _logger.info(
        "[${enteFile.uploadedFileID}] After thumbnail cache check: ${facesWithoutCrops.length} faces still need generation",
      );
      if (facesWithoutCrops.isEmpty) {
        _logger.info(
          "[${enteFile.uploadedFileID}] All faces found in thumbnail cache, returning ${faceIdToCrop.length} crops",
        );
        return faceIdToCrop;
      }
    }

    _logger.info(
      "[${enteFile.uploadedFileID}] Starting face crop generation for ${facesWithoutCrops.length} faces using ${useFullFile ? 'full file' : 'thumbnail'}",
    );
    final result = await _getFaceCropsUsingHeapPriorityQueue(
      enteFile,
      facesWithoutCrops,
      useFullFile: useFullFile,
    );
    _logger.info(
      "[${enteFile.uploadedFileID}] Face crop generation completed, result: ${result != null ? 'success with ${result.length} crops' : 'null'}",
    );
    if (result == null) {
      _logger.severe(
        "[${enteFile.uploadedFileID}] Face crop generation returned null, returning ${faceIdToCrop.isEmpty ? 'null' : '${faceIdToCrop.length} cached crops'}",
      );
      return (faceIdToCrop.isEmpty) ? null : faceIdToCrop;
    }
    _logger.info(
      "[${enteFile.uploadedFileID}] Processing ${result.length} generated face crops",
    );
    for (final entry in result.entries) {
      final Uint8List? computedCrop = result[entry.key];
      if (computedCrop != null) {
        _logger.info(
          "[${enteFile.uploadedFileID}] Processing generated crop for face ${entry.key}, size: ${computedCrop.length} bytes",
        );
        faceIdToCrop[entry.key] = computedCrop;
        if (useFullFile) {
          _logger.info(
            "[${enteFile.uploadedFileID}] Caching generated crop for face ${entry.key} in memory and file",
          );
          await _putCachedCropForFaceID(
            entry.key,
            computedCrop,
            personOrClusterID,
            enteFile.uploadedFileID,
          );
          final faceCropCacheFile = cachedFaceCropPath(
            entry.key,
            useTempCache,
            enteFile.uploadedFileID,
          );
          try {
            _logger.info(
              "[${enteFile.uploadedFileID}] Writing crop to file: ${faceCropCacheFile.path}",
            );

            // Ensure directory exists before writing
            final cacheDir = faceCropCacheFile.parent;
            if (!(await cacheDir.exists())) {
              _logger.severe(
                "[${enteFile.uploadedFileID}] Creating cache directory before write: ${cacheDir.path}",
              );
              await cacheDir.create(recursive: true);
            }

            await faceCropCacheFile.writeAsBytes(computedCrop);
            _logger.info(
              "[${enteFile.uploadedFileID}] Successfully wrote crop file for face ${entry.key}",
            );
          } catch (e, s) {
            _logger.severe(
              "[${enteFile.uploadedFileID}] Error writing cached face crop for faceID ${entry.key} to file ${faceCropCacheFile.path}",
              e,
              s,
            );
          }
        } else {
          _logger.info(
            "[${enteFile.uploadedFileID}] Caching generated crop for face ${entry.key} in thumbnail cache",
          );
          _faceCropThumbnailCache.put(entry.key, computedCrop);
        }
      } else {
        _logger.severe(
          "[${enteFile.uploadedFileID}] Generated crop for face ${entry.key} is null",
        );
      }
    }
    _logger.info(
      "[${enteFile.uploadedFileID}] Final result: ${faceIdToCrop.length} total face crops, returning ${faceIdToCrop.isEmpty ? 'null' : 'map with ${faceIdToCrop.length} entries'}",
    );
    return faceIdToCrop.isEmpty ? null : faceIdToCrop;
  } catch (e, s) {
    _logger.severe(
      "[${enteFile.uploadedFileID}] getCachedFaceCrops EXCEPTION: faces=${faces.map((face) => face.faceID).toList()}, error=$e",
    );
    if (e is! TaskQueueTimeoutException &&
        e is! TaskQueueOverflowException &&
        e is! TaskQueueCancelledException) {
      if (fetchAttempt <= _retryLimit) {
        final backoff = Duration(
          milliseconds: 100 * pow(2, fetchAttempt + 1).toInt(),
        );
        _logger.info(
          "[${enteFile.uploadedFileID}] Will retry after ${backoff.inMilliseconds}ms delay",
        );
        await Future.delayed(backoff);
        _logger.severe(
          "[${enteFile.uploadedFileID}] Error getting face crops for faceIDs: ${faces.map((face) => face.faceID).toList()}, retrying (attempt ${fetchAttempt + 1}) in ${backoff.inMilliseconds} ms",
          e,
          s,
        );
        return getCachedFaceCrops(
          enteFile,
          faces,
          fetchAttempt: fetchAttempt + 1,
          useFullFile: useFullFile,
          useTempCache: useTempCache,
        );
      }
      _logger.severe(
        "[${enteFile.uploadedFileID}] Error getting face crops for faceIDs: ${faces.map((face) => face.faceID).toList()}",
        e,
        s,
      );
    } else {
      _logger.severe(
        "[${enteFile.uploadedFileID}] Stopped getting face crops for faceIDs: ${faces.map((face) => face.faceID).toList()} due to $e",
      );
    }
    _logger.info(
      "[${enteFile.uploadedFileID}] getCachedFaceCrops returning null due to exception",
    );
    return null;
  }
}

Future<Uint8List?> precomputeClusterFaceCrop(
  file,
  clusterID, {
  required bool useFullFile,
}) async {
  try {
    final w = (kDebugMode ? EnteWatch('precomputeClusterFaceCrop') : null)
      ?..start();
    final Face? face = await MLDataDB.instance.getCoverFaceForPerson(
      recentFileID: file.uploadedFileID!,
      clusterID: clusterID,
    );
    w?.log('getCoverFaceForPerson');
    if (face == null) {
      debugPrint(
        "No cover face for cluster $clusterID and recentFile ${file.uploadedFileID}",
      );
      return null;
    }
    EnteFile? fileForFaceCrop = file;
    if (face.fileID != file.uploadedFileID!) {
      fileForFaceCrop = await FilesDB.instance.getAnyUploadedFile(face.fileID);
      w?.log('getAnyUploadedFile');
    }
    if (fileForFaceCrop == null) {
      return null;
    }
    final cropMap = await getCachedFaceCrops(
      fileForFaceCrop,
      [face],
      useFullFile: useFullFile,
      useTempCache: true,
    );
    w?.logAndReset('getCachedFaceCrops');
    return cropMap?[face.faceID];
  } catch (e, s) {
    _logger.severe(
      "Error getting cover face for cluster $clusterID",
      e,
      s,
    );
    return null;
  }
}

void checkStopTryingToGenerateFaceThumbnails(
  int fileID, {
  bool useFullFile = true,
}) {
  final taskId = [fileID, useFullFile ? "-full" : "-thumbnail"].join();
  if (useFullFile) {
    _queueFullFileFaceGenerations.removeTask(taskId);
  } else {
    _queueThumbnailFaceGenerations.removeTask(taskId);
  }
}

Future<Map<String, Uint8List>?> _getFaceCropsUsingHeapPriorityQueue(
  EnteFile file,
  Map<String, FaceBox> faceBoxeMap, {
  bool useFullFile = true,
}) async {
  _logger.info(
    "[${file.uploadedFileID}] _getFaceCropsUsingHeapPriorityQueue START: faces=${faceBoxeMap.length}, useFullFile=$useFullFile",
  );
  final completer = Completer<Map<String, Uint8List>?>();

  late final TaskQueue relevantTaskQueue;
  late final String taskId;
  if (useFullFile) {
    relevantTaskQueue = _queueFullFileFaceGenerations;
    taskId = [file.uploadedFileID!, "-full"].join();
    _logger.info(
      "[${file.uploadedFileID}] Using full file task queue with taskId: $taskId",
    );
  } else {
    relevantTaskQueue = _queueThumbnailFaceGenerations;
    taskId = [file.uploadedFileID!, "-thumbnail"].join();
    _logger.info(
      "[${file.uploadedFileID}] Using thumbnail task queue with taskId: $taskId",
    );
  }

  _logger.info("[${file.uploadedFileID}] Adding task to queue: $taskId");
  try {
    await relevantTaskQueue.addTask(taskId, () async {
      _logger.info("[${file.uploadedFileID}] Task $taskId started execution");
      final faceCrops = await _getFaceCrops(
        file,
        faceBoxeMap,
        useFullFile: useFullFile,
      );
      _logger.info(
        "[${file.uploadedFileID}] Task $taskId completed, result: ${faceCrops != null ? '${faceCrops.length} crops' : 'null'}",
      );
      completer.complete(faceCrops);
    });
  } catch (e, s) {
    _logger.severe(
      "[${file.uploadedFileID}] Error adding task $taskId to queue",
      e,
      s,
    );
    completer.complete(null);
  }

  _logger.info("[${file.uploadedFileID}] Waiting for task $taskId to complete");
  final result = await completer.future;
  _logger.info(
    "[${file.uploadedFileID}] _getFaceCropsUsingHeapPriorityQueue END: taskId=$taskId, result=${result != null ? '${result.length} crops' : 'null'}",
  );
  return result;
}

Future<Map<String, Uint8List>?> _getFaceCrops(
  EnteFile file,
  Map<String, FaceBox> faceBoxeMap, {
  bool useFullFile = true,
}) async {
  _logger.info(
    "[${file.uploadedFileID}] _getFaceCrops START: faces=${faceBoxeMap.length}, useFullFile=$useFullFile, fileType=${file.fileType}",
  );
  late String? imagePath;
  if (useFullFile && file.fileType != FileType.video) {
    _logger.info("[${file.uploadedFileID}] Getting full file");
    final File? ioFile = await getFile(file);
    if (ioFile == null) {
      _logger.severe(
        "[${file.uploadedFileID}] Failed to get file for face crop generation",
      );
      return null;
    }
    imagePath = ioFile.path;
    _logger.info(
      "[${file.uploadedFileID}] Got full file path: $imagePath, exists: ${await ioFile.exists()}",
    );
  } else {
    _logger.info("[${file.uploadedFileID}] Getting thumbnail");
    final thumbnail = await getThumbnailForUploadedFile(file);
    if (thumbnail == null) {
      _logger.severe(
        "[${file.uploadedFileID}] Failed to get thumbnail for face crop generation",
      );
      return null;
    }
    imagePath = thumbnail.path;
    _logger.info(
      "[${file.uploadedFileID}] Got thumbnail path: $imagePath, exists: ${await thumbnail.exists()}",
    );
  }

  _logger.info("[${file.uploadedFileID}] Preparing face data for generation");
  final List<String> faceIds = [];
  final List<FaceBox> faceBoxes = [];
  for (final e in faceBoxeMap.entries) {
    faceIds.add(e.key);
    faceBoxes.add(e.value);
    _logger.info(
      "[${file.uploadedFileID}] Face ${e.key}: box=${e.value.x}, ${e.value.y}, ${e.value.width}, ${e.value.height}",
    );
  }

  _logger.info(
    "[${file.uploadedFileID}] Calling FaceThumbnailGenerator with ${faceBoxes.length} face boxes",
  );
  try {
    final List<Uint8List> faceCrop =
        await FaceThumbnailGenerator.instance.generateFaceThumbnails(
      imagePath,
      faceBoxes,
      fileID: file.uploadedFileID,
    );
    _logger.info(
      "[${file.uploadedFileID}] FaceThumbnailGenerator returned ${faceCrop.length} crops",
    );

    final Map<String, Uint8List> result = {};
    for (int i = 0; i < faceCrop.length; i++) {
      result[faceIds[i]] = faceCrop[i];
      _logger.info(
        "[${file.uploadedFileID}] Mapped face ${faceIds[i]} to crop of size ${faceCrop[i].length} bytes",
      );
    }
    _logger.info(
      "[${file.uploadedFileID}] _getFaceCrops END: returning ${result.length} face crops",
    );
    return result;
  } catch (e, s) {
    _logger.severe(
      "[${file.uploadedFileID}] Error in FaceThumbnailGenerator.generateFaceThumbnails",
      e,
      s,
    );
    return null;
  }
}
