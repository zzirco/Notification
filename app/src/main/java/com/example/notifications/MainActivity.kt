package com.example.notifications

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.audioprocessing.AudioProcessor
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var audioProcessor: AudioProcessor
    private lateinit var recordButton: Button
    private lateinit var fileListButton: Button
    private lateinit var resultTextView: TextView
    private var mediaRecorder: MediaRecorder? = null
    private var isRecording = false
    private var audioFile: File? = null
    private val coroutineScope = CoroutineScope(Dispatchers.Main + Job())

    companion object {
        private const val TAG = "MainActivity"
        private const val PERMISSION_REQUEST_CODE = 123

        private fun getRequiredPermissions(): Array<String> {
            return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                arrayOf(
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.READ_MEDIA_AUDIO
                )
            } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                arrayOf(
                    Manifest.permission.RECORD_AUDIO
                )
            } else {
                arrayOf(
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE,
                    Manifest.permission.READ_EXTERNAL_STORAGE
                )
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        recordButton = findViewById(R.id.recordButton)
        fileListButton = findViewById(R.id.fileListButton)
        resultTextView = findViewById(R.id.resultTextView)

        audioProcessor = AudioProcessor(this)

        recordButton.setOnClickListener {
            if (isRecording) {
                stopRecording()
            } else {
                checkAndRequestPermissions()
            }
        }

        fileListButton.setOnClickListener {
            showRecordedFiles()
        }
    }

    private fun checkAndRequestPermissions() {
        val permissions = getRequiredPermissions()
        Log.d(TAG, "Required permissions: ${permissions.joinToString()}")

        val missingPermissions = permissions.filter { permission ->
            ContextCompat.checkSelfPermission(this, permission) !=
                    PackageManager.PERMISSION_GRANTED
        }

        Log.d(TAG, "Missing permissions: ${missingPermissions.joinToString()}")

        when {
            missingPermissions.isEmpty() -> {
                Log.d(TAG, "All permissions already granted")
                startRecording()
            }
            missingPermissions.any { permission ->
                shouldShowRequestPermissionRationale(permission)
            } -> {
                Log.d(TAG, "Should show permission rationale")
                showPermissionRationaleDialog(missingPermissions.toTypedArray())
            }
            else -> {
                Log.d(TAG, "Requesting permissions")
                ActivityCompat.requestPermissions(
                    this,
                    missingPermissions.toTypedArray(),
                    PERMISSION_REQUEST_CODE
                )
            }
        }
    }

    private fun showPermissionRationaleDialog(permissions: Array<String>) {
        AlertDialog.Builder(this)
            .setTitle("권한 필요")
            .setMessage("앱이 정상적으로 동작하기 위해서는 다음 권한이 필요합니다:\n" +
                    permissions.joinToString("\n") { permission ->
                        when (permission) {
                            Manifest.permission.RECORD_AUDIO -> "• 마이크 (음성 녹음)"
                            Manifest.permission.READ_MEDIA_AUDIO -> "• 오디오 미디어 접근"
                            Manifest.permission.WRITE_EXTERNAL_STORAGE -> "• 저장소 쓰기"
                            Manifest.permission.READ_EXTERNAL_STORAGE -> "• 저장소 읽기"
                            else -> "• $permission"
                        }
                    })
            .setPositiveButton("권한 설정") { _, _ ->
                ActivityCompat.requestPermissions(
                    this,
                    permissions,
                    PERMISSION_REQUEST_CODE
                )
            }
            .setNegativeButton("취소", null)
            .show()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            val deniedPermissions = permissions.filterIndexed { index, _ ->
                grantResults[index] != PackageManager.PERMISSION_GRANTED
            }

            when {
                deniedPermissions.isEmpty() -> {
                    Log.d(TAG, "All permissions granted")
                    startRecording()
                }
                deniedPermissions.any { permission ->
                    !shouldShowRequestPermissionRationale(permission)
                } -> {
                    Log.d(TAG, "Some permissions permanently denied")
                    showPermissionSettingsDialog()
                }
                else -> {
                    Log.d(TAG, "Some permissions denied")
                    showPermissionRationaleDialog(deniedPermissions.toTypedArray())
                }
            }
        }
    }

    private fun showPermissionSettingsDialog() {
        AlertDialog.Builder(this)
            .setTitle("권한 설정 필요")
            .setMessage("앱 설정 화면에서 필요한 권한을 허용해주세요.")
            .setPositiveButton("설정으로 이동") { _, _ ->
                startActivity(Intent().apply {
                    action = Settings.ACTION_APPLICATION_DETAILS_SETTINGS
                    data = Uri.fromParts("package", packageName, null)
                })
            }
            .setNegativeButton("취소", null)
            .show()
    }

    private fun startRecording() {
        try {
            val outputDir = getExternalFilesDir(Environment.DIRECTORY_MUSIC)
                ?: throw IllegalStateException("Cannot access external storage")

            Log.d(TAG, "Storage Directory: ${outputDir.absolutePath}")

            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            audioFile = File(outputDir, "audio_record_$timestamp.m4a")

            Log.d(TAG, "Recording will be saved to: ${audioFile?.absolutePath}")

            mediaRecorder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                MediaRecorder(this)
            } else {
                @Suppress("DEPRECATION")
                MediaRecorder()
            }.apply {
                setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)  // MIC 대신 VOICE_RECOGNITION 사용
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                setAudioChannels(1)
                setAudioSamplingRate(16000)  // 16kHz로 고정
                setAudioEncodingBitRate(192000)  // 비트레이트는 높게 설정
                setOutputFile(audioFile?.absolutePath)
                prepare()
                start()
            }

            isRecording = true
            recordButton.text = "녹음 중지"
            resultTextView.text = "녹음 중...\n저장 경로: ${audioFile?.absolutePath}"

        } catch (e: Exception) {
            Log.e(TAG, "Recording failed", e)
            Toast.makeText(this, "녹음 시작 실패: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun stopRecording() {
        try {
            mediaRecorder?.apply {
                stop()
                release()
            }
            mediaRecorder = null
            isRecording = false
            recordButton.text = "녹음 시작"

            audioFile?.let { file ->
                // 파일 기본 정보 로깅
                Log.d(TAG, "Audio file debug:")
                Log.d(TAG, "File exists: ${file.exists()}")
                Log.d(TAG, "File path: ${file.absolutePath}")
                Log.d(TAG, "File size: ${file.length()} bytes")
                Log.d(TAG, "File can read: ${file.canRead()}")

                if (file.exists() && file.length() > 0) {
                    resultTextView.text = "녹음 완료. 분석 중..."

                    coroutineScope.launch(Dispatchers.IO) {
                        try {
                            // MediaExtractor를 사용한 오디오 파일 정보 로깅
                            try {
                                val extractor = MediaExtractor()
                                extractor.setDataSource(file.absolutePath)
                                Log.d(TAG, "MediaExtractor debug:")
                                Log.d(TAG, "Track count: ${extractor.trackCount}")

                                if (extractor.trackCount > 0) {
                                    val format = extractor.getTrackFormat(0)
                                    Log.d(TAG, "Track format: $format")
                                    Log.d(TAG, "Mime type: ${format.getString(MediaFormat.KEY_MIME)}")
                                    Log.d(TAG, "Duration: ${format.getLong(MediaFormat.KEY_DURATION)}")
                                    Log.d(TAG, "Sample rate: ${format.getInteger(MediaFormat.KEY_SAMPLE_RATE)}")
                                    Log.d(TAG, "Channel count: ${format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)}")
                                }
                                extractor.release()
                            } catch (e: Exception) {
                                Log.e(TAG, "Failed to extract audio information", e)
                            }

                            val result = audioProcessor.processAudioFile(file.absolutePath)
                            withContext(Dispatchers.Main) {
                                resultTextView.text = "감지된 소리: $result"
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Audio processing failed", e)
                            withContext(Dispatchers.Main) {
                                Toast.makeText(
                                    this@MainActivity,
                                    "오디오 처리 실패: ${e.message}",
                                    Toast.LENGTH_LONG
                                ).show()
                                resultTextView.text = "오디오 처리 실패"
                            }
                        }
                    }
                } else {
                    Log.e(TAG, "Recording file is empty or does not exist")
                    resultTextView.text = "녹음 파일이 생성되지 않았습니다"
                    Toast.makeText(this, "녹음 파일이 생성되지 않았습니다", Toast.LENGTH_SHORT).show()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Stop recording failed", e)
            Toast.makeText(this, "녹음 중지 실패: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun showRecordedFiles() {
        val outputDir = getExternalFilesDir(Environment.DIRECTORY_MUSIC)
        outputDir?.listFiles()?.let { files ->
            val fileList = files.filter { it.extension in listOf("wav", "m4a", "mp4") }
                .sortedByDescending { it.lastModified() }
                .joinToString("\n\n") { file ->
                    val sizeInKB = file.length() / 1024
                    val lastModified = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
                        .format(Date(file.lastModified()))
                    "${file.name} ($sizeInKB KB)\n$lastModified"
                }

            AlertDialog.Builder(this)
                .setTitle("녹음된 파일 목록")
                .setMessage(fileList.ifEmpty { "녹음 파일이 없습니다" })
                .setPositiveButton("확인", null)
                .setNeutralButton("저장 경로 복사") { _, _ ->
                    val clipboardManager = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                    val clip = ClipData.newPlainText("저장 경로", outputDir.absolutePath)
                    clipboardManager.setPrimaryClip(clip)
                    Toast.makeText(this, "저장 경로가 복사되었습니다", Toast.LENGTH_SHORT).show()
                }
                .show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        coroutineScope.cancel()
        mediaRecorder?.release()
        mediaRecorder = null
    }
}