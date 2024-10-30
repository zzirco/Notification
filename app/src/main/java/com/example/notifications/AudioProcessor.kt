package com.example.audioprocessing

import ai.onnxruntime.*
import android.content.Context
import android.media.MediaExtractor
import android.media.MediaFormat
import android.util.Log
import com.google.gson.JsonParser
import org.apache.commons.math3.complex.Complex
import org.apache.commons.math3.transform.FastFourierTransformer
import org.apache.commons.math3.transform.DftNormalization
import org.apache.commons.math3.transform.TransformType
import java.io.*
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import kotlin.math.*

class AudioProcessor(private val context: Context) : AutoCloseable {
    companion object {
        private const val TAG = "AudioProcessor"
        private const val MODEL_FILENAME = "ast_model_with_metadata.onnx"
        private const val BUFFER_SIZE = 8192 // 8KB buffer
        const val SAMPLE_RATE = 16000
        const val EXPECTED_LENGTH = 1024
        const val MEAN = -4.2677393f
        const val STD = 4.5689974f
        const val NUM_MEL_FILTERS = 128
        const val FRAME_LENGTH = 400
        const val HOP_LENGTH = 160
        const val FFT_LENGTH = 512
        const val MIN_FREQ = 20f
        const val MEL_FLOOR = 1.192092955078125e-7f
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var onnxSession: OrtSession? = null
    private val fft = FastFourierTransformer(DftNormalization.STANDARD)
    private lateinit var labelMap: Map<Int, String>

    init {
        try {
            initializeModel()
            loadModelMetadata()
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing model", e)
            throw RuntimeException("Failed to initialize audio model", e)
        }
    }

    override fun close() {
        try {
            onnxSession?.close()
            onnxSession = null
        } catch (e: Exception) {
            Log.e(TAG, "Error closing ONNX session", e)
        }
    }

    private fun initializeModel() {
        try {
            val modelFile = File(context.filesDir, MODEL_FILENAME)

            if (!modelFile.exists()) {
                context.assets.open(MODEL_FILENAME).use { input ->
                    BufferedInputStream(input, BUFFER_SIZE).use { bufferedInput ->
                        FileOutputStream(modelFile).use { output ->
                            val buffer = ByteArray(BUFFER_SIZE)
                            var len: Int
                            while (bufferedInput.read(buffer).also { len = it } != -1) {
                                output.write(buffer, 0, len)
                            }
                            output.flush()
                        }
                    }
                }
            }

            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(1)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setMemoryPatternOptimization(true)

                // 메모리 최적화를 위한 추가 설정
                addConfigEntry("session.use_arena_allocation", "0")  // 메모리 아레나 할당 비활성화
                addConfigEntry("session.use_arena_memory_planner", "0")  // 메모리 플래너 비활성화
            }

            onnxSession = env.createSession(modelFile.absolutePath, sessionOptions)
            Log.d(TAG, "Model initialized successfully")


            onnxSession = env.createSession(modelFile.absolutePath, sessionOptions)

            // 모델 정보 로깅
            Log.d(TAG, "Model Input Info:")
            for (input in onnxSession?.inputInfo ?: emptyMap()) {
                Log.d(TAG, "Input name: ${input.key}")
                Log.d(TAG, "Input info: ${input.value}")
            }

            Log.d(TAG, "Model Output Info:")
            for (output in onnxSession?.outputInfo ?: emptyMap()) {
                Log.d(TAG, "Output name: ${output.key}")
                Log.d(TAG, "Output info: ${output.value}")
            }

            Log.d(TAG, "Model initialized successfully")
        } catch (e: IOException) {
            Log.e(TAG, "Error loading model file from assets", e)
            throw RuntimeException("Failed to load audio model file", e)
        } catch (e: Exception) {
            Log.e(TAG, "Error creating ONNX session", e)
            throw RuntimeException("Failed to create ONNX session", e)
        }
    }

    private fun loadAndResampleAudio(filePath: String): FloatArray {
        val extractor = MediaExtractor()
        extractor.setDataSource(filePath)

        val format = extractor.getTrackFormat(0)
        extractor.selectTrack(0)  // 트랙 선택 추가

        val originalSampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channelCount = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

        val maxBufferSize = format.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE)
        val buffer = ByteBuffer.allocate(maxBufferSize)
        val samples = mutableListOf<Float>()

        try {
            while (true) {
                // 버퍼 초기화
                buffer.clear()

                // 샘플 데이터 읽기
                val sampleSize = extractor.readSampleData(buffer, 0)
                if (sampleSize < 0) break

                // position을 0으로 설정
                buffer.position(0)
                buffer.limit(sampleSize)

                // short 데이터 읽기
                while (buffer.remaining() >= 2) {  // short는 2바이트
                    samples.add(buffer.short / 32768f)
                }

                if (!extractor.advance()) break
            }
        } finally {
            extractor.release()
        }

        // 로그 추가
        Log.d(TAG, "Original samples size: ${samples.size}")

        val monoSamples = if (channelCount == 2) {
            samples.chunked(2) { it[0] * 0.5f + it[1] * 0.5f }.toFloatArray()
        } else {
            samples.toFloatArray()
        }

        Log.d(TAG, "Mono samples size: ${monoSamples.size}")

        val resampledAudio = resample(monoSamples, originalSampleRate, SAMPLE_RATE)
        Log.d(TAG, "Resampled audio size: ${resampledAudio.size}")

        return resampledAudio
    }

    private fun resample(input: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        val ratio = toRate.toDouble() / fromRate
        val outputLength = (input.size * ratio).toInt()
        val output = FloatArray(outputLength)

        for (i in output.indices) {
            val inputIdx = (i / ratio).toInt()
            val nextInputIdx = minOf(inputIdx + 1, input.size - 1)
            val fraction = ((i / ratio) - inputIdx).toFloat()

            output[i] = input[inputIdx] * (1f - fraction) + input[nextInputIdx] * fraction
        }

        return output
    }

    private fun createWindow(size: Int): FloatArray {
        return FloatArray(size) { i ->
            0.5f * (1 - cos(2.0 * PI * i / (size - 1))).toFloat()
        }
    }

    private fun createMelFilterbank(): Array<FloatArray> {
        val fftFreqs = FloatArray(FFT_LENGTH / 2 + 1) { i ->
            i * SAMPLE_RATE.toFloat() / FFT_LENGTH
        }

        val melMin = hzToMel(MIN_FREQ)
        val melMax = hzToMel(SAMPLE_RATE / 2f)
        val melPoints = FloatArray(NUM_MEL_FILTERS + 2) { i ->
            melMin + i * (melMax - melMin) / (NUM_MEL_FILTERS + 1)
        }
        val freqPoints = melPoints.map { melToHz(it) }

        return Array(NUM_MEL_FILTERS) { i ->
            FloatArray(FFT_LENGTH / 2 + 1) { f ->
                val freq = fftFreqs[f]
                when {
                    freq < freqPoints[i] -> 0f
                    freq > freqPoints[i + 2] -> 0f
                    freq <= freqPoints[i + 1] -> {
                        (freq - freqPoints[i]) / (freqPoints[i + 1] - freqPoints[i])
                    }
                    else -> {
                        (freqPoints[i + 2] - freq) / (freqPoints[i + 2] - freqPoints[i + 1])
                    }
                }
            }
        }
    }

    private fun hzToMel(freq: Float): Float {
        return 1127f * ln(1f + freq / 700f)
    }

    private fun melToHz(mel: Float): Float {
        return 700f * (exp(mel / 1127f) - 1f)
    }

    private fun computeSpectrogram(audio: FloatArray): Array<FloatArray> {
        if (audio.isEmpty()) {
            throw IllegalArgumentException("Audio data is empty")
        }

        val window = createWindow(FRAME_LENGTH)
        // 프레임 수 계산 시 반올림 처리
        val numFrames = max(1, (audio.size - FRAME_LENGTH + HOP_LENGTH - 1) / HOP_LENGTH)
        Log.d(TAG, "Audio length: ${audio.size}, Num frames: $numFrames")

        val spectrogram = Array(numFrames) { FloatArray(FFT_LENGTH / 2 + 1) }
        val fftBuffer = DoubleArray(FFT_LENGTH)

        for (i in 0 until numFrames) {
            val start = i * HOP_LENGTH
            val end = minOf(start + FRAME_LENGTH, audio.size)

            // Clear FFT buffer
            fftBuffer.fill(0.0)

            // Apply window function and prepare FFT input
            for (j in 0 until (end - start)) {
                fftBuffer[j] = (audio[start + j] * window[j]).toDouble()
            }

            // Compute FFT
            val fftResult = fft.transform(fftBuffer, TransformType.FORWARD)

            // Compute power spectrum
            for (j in 0..FFT_LENGTH / 2) {
                val real = fftResult[j].real
                val imag = fftResult[j].imaginary
                spectrogram[i][j] = sqrt(real * real + imag * imag).toFloat()
            }
        }

        return spectrogram
    }

    private fun applyMelFilters(spectrogram: Array<FloatArray>): Array<FloatArray> {
        val melFilters = createMelFilterbank()
        val melSpectrogram = Array(spectrogram.size) { FloatArray(NUM_MEL_FILTERS) }

        for (i in spectrogram.indices) {
            for (j in 0 until NUM_MEL_FILTERS) {
                var sum = 0f
                for (k in spectrogram[i].indices) {
                    sum += spectrogram[i][k] * melFilters[j][k]
                }
                melSpectrogram[i][j] = maxOf(MEL_FLOOR, sum)
            }
        }

        return melSpectrogram
    }

    private fun normalize(input: Array<FloatArray>): Array<FloatArray> {
        return Array(input.size) { i ->
            FloatArray(input[i].size) { j ->
                (input[i][j] - MEAN) / (STD * 2f)
            }
        }
    }

    private fun flattenFeatures(features: Array<FloatArray>): FloatArray {
        val result = FloatArray(features.size * features[0].size)
        var idx = 0
        for (i in features.indices) {
            for (j in features[i].indices) {
                result[idx++] = features[i][j]
            }
        }
        return result
    }

    private fun padOrTrimFeatures(features: Array<FloatArray>): Array<FloatArray> {
        val paddedFeatures = Array(EXPECTED_LENGTH) { FloatArray(NUM_MEL_FILTERS) }

        // Copy existing features
        for (i in features.indices.take(EXPECTED_LENGTH)) {
            features[i].copyInto(paddedFeatures[i])
        }

        // If features.size < EXPECTED_LENGTH, remaining rows will be zero-padded
        return paddedFeatures
    }

    fun processAudioFile(filePath: String): String {
        try {
            val audio = loadAndResampleAudio(filePath)
            val spectrogram = computeSpectrogram(audio)
            val melSpectrogram = applyMelFilters(spectrogram)
            val normalizedFeatures = normalize(melSpectrogram)

            // 패딩 또는 트리밍 적용
            val paddedFeatures = padOrTrimFeatures(normalizedFeatures)

            val inputShape = longArrayOf(1, EXPECTED_LENGTH.toLong(), NUM_MEL_FILTERS.toLong())
            val flattenedInput = flattenFeatures(paddedFeatures)  // paddedFeatures 사용

            Log.d(TAG, "Features before padding: ${normalizedFeatures.size} x ${normalizedFeatures[0].size}")
            Log.d(TAG, "Features after padding: ${paddedFeatures.size} x ${paddedFeatures[0].size}")
            Log.d(TAG, "Flattened input size: ${flattenedInput.size}")

            return OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(flattenedInput),
                inputShape
            ).use { inputTensor ->
                onnxSession?.run(mapOf("input_values" to inputTensor))?.use { output ->
                    val outputTensor = output[0].value as Array<FloatArray>
                    val predictionIdx = outputTensor[0].indices.maxByOrNull { outputTensor[0][it] } ?: 0
                    mapIndexToLabel(predictionIdx)
                } ?: throw RuntimeException("Failed to run inference")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing audio file", e)
            throw RuntimeException("Failed to process audio file", e)
        }
    }

    private fun loadModelMetadata() {
        try {
            // assets에서 라벨 정보 JSON 파일 읽기
            val labelsJson = context.assets.open("labels.json").bufferedReader().use { it.readText() }

            labelMap = JsonParser.parseString(labelsJson)
                .asJsonObject
                .entrySet()
                .associate { (key, value) ->
                    key.toInt() to value.asString
                }

            Log.d(TAG, "Loaded label map from JSON: $labelMap")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading labels.json", e)
            // 기본값 사용
            labelMap = mapOf(
                0 to "Vehicle",
                1 to "Speech",
                2 to "Music"
            )
            Log.w(TAG, "Using default label map due to error: ${e.message}")
        }
    }

    private fun mapIndexToLabel(index: Int): String {
        return labelMap[index] ?: "Unknown"
    }
}