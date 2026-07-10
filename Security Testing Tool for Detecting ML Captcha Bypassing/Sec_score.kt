import android.app.Service
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.IBinder
import android.view.MotionEvent
import androidx.room.*
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt
import kotlin.math.pow
 
// ==============================================================================
// 1. DATA MODELS & ROOM DATABASE
// ==============================================================================
 
data class TelemetryInput(
    val touchPressures: FloatArray,
    val swipeVelocities: FloatArray,
    val gyroVariance: FloatArray,
    val interactionIntervalsMs: LongArray
)
 
@Entity(tableName = "anomaly_reports")
data class AnomalyPostureReport(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val timestamp: Long,
    val anomalyScore: Float,          // 0.0 (looks human) to 1.0 (looks scripted/bot-like)
    val confidenceInterval: Float,
    val flaggedTelemetryNodes: String, // Stored as comma-separated string
    val anomalyDetected: Boolean
)
 
@Dao
interface ScoringReportDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertReport(report: AnomalyPostureReport)
 
    @Query("SELECT * FROM anomaly_reports ORDER BY timestamp DESC LIMIT 50")
    suspend fun getLatestReports(): List<AnomalyPostureReport>
}
 
// ==============================================================================
// 2. SCORING ENGINE (ML Tensor Processing)
// ==============================================================================
 
class AutomatedScoringEngine(
    private val tfLiteInterpreter: Interpreter,
    private val reportDao: ScoringReportDao
) {
    companion object {
        private const val ANOMALY_THRESHOLD = 0.85f
        private const val SAMPLES_PER_CHANNEL = 25
        private const val NUM_CHANNELS = 4
    }
 
    suspend fun analyzeTelemetryAndStore(telemetry: TelemetryInput): AnomalyPostureReport =
        withContext(Dispatchers.Default) {
 
            val inputBuffer = preprocessTelemetry(telemetry)
 
            // Output shape must match the actual model. Declaring it explicitly
            // instead of assuming a bare 4-byte buffer avoids a runtime crash
            // if the interpreter expects a different tensor shape.
            val outputShape = tfLiteInterpreter.getOutputTensor(0).shape()
            val outputBuffer = ByteBuffer
                .allocateDirect(outputShape.fold(1) { acc, d -> acc * d } * 4)
                .order(ByteOrder.nativeOrder())
 
            tfLiteInterpreter.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()
 
            val anomalyScore = outputBuffer.float.coerceIn(0f, 1f)
 
            val report = generateReport(anomalyScore, telemetry)
            reportDao.insertReport(report)
 
            report
        }
 
    private fun preprocessTelemetry(telemetry: TelemetryInput): ByteBuffer {
        val buffer = ByteBuffer
            .allocateDirect(SAMPLES_PER_CHANNEL * NUM_CHANNELS * 4)
            .order(ByteOrder.nativeOrder())
 
        writeChannel(buffer, telemetry.touchPressures)
        writeChannel(buffer, telemetry.swipeVelocities)
        writeChannel(buffer, telemetry.gyroVariance)
        writeChannel(buffer, telemetry.interactionIntervalsMs.map { it.toFloat() }.toFloatArray())
 
        buffer.rewind()
        return buffer
    }
 
    // Always writes exactly SAMPLES_PER_CHANNEL floats: truncates long arrays,
    // zero-pads short ones. The original .take(25) left the buffer's remaining
    // bytes unwritten (stale/garbage) whenever fewer than 25 samples existed.
    private fun writeChannel(buffer: ByteBuffer, data: FloatArray) {
        for (i in 0 until SAMPLES_PER_CHANNEL) {
            val value = if (i < data.size) normalize(data[i]) else 0f
            buffer.putFloat(value)
        }
    }
 
    private fun generateReport(score: Float, telemetry: TelemetryInput): AnomalyPostureReport {
        val flaggedNodes = mutableListOf<String>()
 
        if (score > ANOMALY_THRESHOLD) {
            if (telemetry.touchPressures.size >= 2 &&
                calculateVariance(telemetry.touchPressures) < 0.01f
            ) {
                flaggedNodes.add("TOUCH_PRESSURE_TOO_UNIFORM")
            }
            val intervalFloats = telemetry.interactionIntervalsMs.map { it.toFloat() }.toFloatArray()
            if (intervalFloats.size >= 2 && calculateVariance(intervalFloats) < 0.05f) {
                flaggedNodes.add("INTERACTION_TIMING_BOT_LIKE")
            }
        }
 
        return AnomalyPostureReport(
            timestamp = System.currentTimeMillis(),
            anomalyScore = score,
            confidenceInterval = 0.95f,
            flaggedTelemetryNodes = flaggedNodes.joinToString(","),
            anomalyDetected = score > ANOMALY_THRESHOLD
        )
    }
 
    private fun normalize(value: Float): Float = value / 100f
 
    // Actual sample variance (was previously just returning the mean).
    private fun calculateVariance(data: FloatArray): Float {
        if (data.size < 2) return 0f
        val mean = data.average().toFloat()
        val sumSquaredDiffs = data.sumOf { (it - mean).toDouble().pow(2) }
        return (sumSquaredDiffs / data.size).toFloat()
    }
}
 
// ==============================================================================
// 3. TELEMETRY COLLECTION SERVICE (Android Native Hook)
// ==============================================================================
 
class TelemetryCollectionService : Service(), SensorEventListener {
 
    private lateinit var sensorManager: SensorManager
    private var gyroSensor: Sensor? = null
 
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
 
    private val gyroBuffer = mutableListOf<Float>()
    private val touchPressureBuffer = mutableListOf<Float>()
    private val interactionTimingBuffer = mutableListOf<Long>()
    private val swipeVelocityBuffer = mutableListOf<Float>()
 
    private var lastInteractionTime: Long = 0
    private var lastTouchX: Float? = null
    private var lastTouchY: Float? = null
    private var lastTouchTime: Long = 0
 
    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        gyroSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
    }
 
    // Call this from the hosting Activity's dispatchTouchEvent(event), e.g.:
    //   override fun dispatchTouchEvent(event: MotionEvent): Boolean {
    //       telemetryService?.recordTouchEvent(event)
    //       return super.dispatchTouchEvent(event)
    //   }
    fun recordTouchEvent(event: MotionEvent) {
        val currentTime = System.currentTimeMillis()
 
        if (lastInteractionTime != 0L) {
            interactionTimingBuffer.add(currentTime - lastInteractionTime)
        }
 
        lastTouchX?.let { prevX ->
            lastTouchY?.let { prevY ->
                val dt = (currentTime - lastTouchTime).coerceAtLeast(1).toFloat()
                val dx = event.x - prevX
                val dy = event.y - prevY
                val distance = sqrt(dx.pow(2) + dy.pow(2))
                swipeVelocityBuffer.add(distance / dt) // px/ms
            }
        }
 
        touchPressureBuffer.add(event.pressure)
        lastTouchX = event.x
        lastTouchY = event.y
        lastTouchTime = currentTime
        lastInteractionTime = currentTime
    }
 
    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_GYROSCOPE) {
            val magnitude = sqrt(
                event.values[0].pow(2) +
                event.values[1].pow(2) +
                event.values[2].pow(2)
            )
            gyroBuffer.add(magnitude)
        }
    }
 
    fun packageTelemetry(): TelemetryInput {
        return TelemetryInput(
            touchPressures = touchPressureBuffer.toFloatArray(),
            swipeVelocities = swipeVelocityBuffer.toFloatArray(),
            gyroVariance = gyroBuffer.toFloatArray(),
            interactionIntervalsMs = interactionTimingBuffer.toLongArray()
        )
    }
 
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* No-op */ }
    override fun onBind(intent: Intent?): IBinder? = null
 
    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)
        serviceScope.cancel()
    }
}
 
