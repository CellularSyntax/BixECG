package com.example.mockecgviewer

import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import androidx.activity.ComponentActivity
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import uk.me.berndporr.iirj.Butterworth
import java.io.BufferedReader
import java.io.InputStreamReader
import kotlin.math.sqrt
import androidx.core.graphics.toColorInt

class MainActivity : ComponentActivity() {

    private lateinit var reader: BufferedReader
    private lateinit var ecgChart: LineChart
    private lateinit var torchModelHandler: TorchModelHandler

    private val samplingRate = 250
    private val segmentLength = 300
    private val predictionStep = 100
    private val handler = Handler(Looper.getMainLooper())
    private val updateIntervalMs = 20L

    private val signalBuffer = mutableListOf<Float>()
    private val predictionBuffer = mutableListOf<Int>()
    private val maxBufferSize = 3000

    private lateinit var butterworth: Butterworth
    private var samplesSinceLastPrediction = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        setContentView(R.layout.activity_main)

        // Set status bar background color (dark color you like)
        window.statusBarColor = Color.parseColor("#222222") // very dark gray (almost black)

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            window.insetsController?.setSystemBarsAppearance(
                0, // No light status bar flags = light icons
                android.view.WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS
            )
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility =
                window.decorView.systemUiVisibility and android.view.View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR.inv()
        }



        try {
            reader = BufferedReader(InputStreamReader(assets.open("ecg.csv")))
            torchModelHandler = TorchModelHandler(this, "model.pt")
            setupCharts()
            setupFilter()
            startMockStreaming()
        } catch (e: Exception) {
            Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_LONG).show()
            e.printStackTrace()
        }
    }

    private fun setupCharts() {
        ecgChart = findViewById(R.id.lineChart)

        ecgChart.description.isEnabled = false
        ecgChart.setTouchEnabled(false)
        ecgChart.setScaleEnabled(false)
        ecgChart.setPinchZoom(false)
        ecgChart.axisRight.isEnabled = false
        ecgChart.legend.isEnabled = false

        ecgChart.axisLeft.axisMinimum = -5f
        ecgChart.axisLeft.axisMaximum = 10f
        ecgChart.axisLeft.gridColor = Color.LTGRAY
        ecgChart.axisLeft.isEnabled = false

        ecgChart.xAxis.isEnabled = true
        ecgChart.xAxis.position = XAxis.XAxisPosition.BOTTOM
        ecgChart.xAxis.gridColor = Color.TRANSPARENT
        ecgChart.xAxis.axisMinimum = 0f
        ecgChart.xAxis.axisMaximum = 1f // 1 second window

        ecgChart.setBackgroundColor("#FFFEFA".toColorInt())
    }

    private fun setupFilter() {
        butterworth = Butterworth()
        val order = 3
        val lowCut = 0.5
        val highCut = 50.0
        val centerFrequency = (lowCut + highCut) / 2
        val bandwidth = highCut - lowCut
        butterworth.bandPass(order, samplingRate.toDouble(), centerFrequency, bandwidth)
    }

    private fun zNormalize(signal: FloatArray): FloatArray {
        val mean = signal.average().toFloat()
        val std = sqrt(signal.fold(0f) { acc, v -> acc + (v - mean) * (v - mean) } / signal.size)
        return FloatArray(signal.size) { i -> (signal[i] - mean) / (std + 1e-6f) }
    }

    private fun startMockStreaming() {
        handler.post(object : Runnable {
            override fun run() {
                try {
                    val line = reader.readLine()
                    if (line != null) {
                        val value = line.trim().toFloat()
                        signalBuffer.add(value)
                        samplesSinceLastPrediction++

                        if (signalBuffer.size >= segmentLength && samplesSinceLastPrediction >= predictionStep) {
                            val segment = signalBuffer.takeLast(segmentLength).toFloatArray()
                            samplesSinceLastPrediction = 0

                            val filteredSegment = segment.map { sample ->
                                butterworth.filter(sample.toDouble()).toFloat()
                            }.toFloatArray()

                            val normalized = zNormalize(filteredSegment)
                            val logits = torchModelHandler.predict(normalized, segmentLength)

                            predictionBuffer.clear()
                            val numClasses = 4
                            for (i in 0 until segmentLength) {
                                var maxIdx = 0
                                var maxVal = logits[i * numClasses]
                                for (j in 1 until numClasses) {
                                    val v = logits[i * numClasses + j]
                                    if (v > maxVal) {
                                        maxVal = v
                                        maxIdx = j
                                    }
                                }
                                predictionBuffer.add(maxIdx)
                            }

                            val smoothedPredictions = applyMajorityVoting(predictionBuffer, windowSize = 9)
                            updateCharts(normalized, smoothedPredictions)
                        }

                        if (signalBuffer.size > maxBufferSize) {
                            signalBuffer.removeAt(0)
                        }

                    } else {
                        reader.close()
                        reader = BufferedReader(InputStreamReader(assets.open("ecg.csv")))
                        signalBuffer.clear()
                        predictionBuffer.clear()
                        samplesSinceLastPrediction = 0
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                }

                handler.postDelayed(this, updateIntervalMs)
            }
        })
    }

    private fun updateCharts(signal: FloatArray, predictions: List<Int>) {
        val ecgEntries = signal.mapIndexed { index, value ->
            Entry(index.toFloat() / samplingRate, value)
        }

        val ecgDataSet = LineDataSet(ecgEntries, "ECG").apply {
            color = Color.parseColor("#2A0592")
            lineWidth = 3.5f
            setDrawCircles(false)
            setDrawValues(false)
            setDrawHighlightIndicators(false)
            mode = LineDataSet.Mode.CUBIC_BEZIER
            cubicIntensity = 0.2f
        }

        val backgroundDataSets = mutableListOf<LineDataSet>()
        var lastClass = predictions.firstOrNull() ?: 0
        var segmentStart = 0

        for (i in 1 until predictions.size) {
            if (predictions[i] != lastClass || i == predictions.size - 1) {
                val segmentEntries = mutableListOf<Entry>()

                val startX = segmentStart.toFloat() / samplingRate
                val endX = i.toFloat() / samplingRate

                segmentEntries.add(Entry(startX, 8f))
                segmentEntries.add(Entry(endX, 8f))

                val color = when (lastClass) {
                    1 -> Color.parseColor("#8FAADC") // Light blue
                    2 -> Color.parseColor("#F4B183") // Light red
                    3 -> Color.parseColor("#A9D18E") // Light green
                    else -> Color.parseColor("#FFFEFA") // Light yellow
                }

                val segmentDataSet = LineDataSet(segmentEntries, null).apply {
                    setDrawFilled(true)
                    fillColor = color
                    fillAlpha = 100
                    setDrawCircles(false)
                    setDrawValues(false)
                    lineWidth = 0f
                    setDrawHighlightIndicators(false)
                    mode = LineDataSet.Mode.LINEAR
                    fillFormatter = com.github.mikephil.charting.formatter.IFillFormatter { _, _ -> -8f }
                }

                if (color != Color.TRANSPARENT) {
                    backgroundDataSets.add(segmentDataSet)
                }

                lastClass = predictions[i]
                segmentStart = i
            }
        }

        ecgChart.clear()

        val lineData = LineData()
        backgroundDataSets.forEach { lineData.addDataSet(it) }
        lineData.addDataSet(ecgDataSet)

        ecgChart.data = lineData
        ecgChart.data.notifyDataChanged()
        ecgChart.notifyDataSetChanged()
        ecgChart.invalidate()
    }

    private fun applyMajorityVoting(predictions: List<Int>, windowSize: Int): List<Int> {
        val halfWindow = windowSize / 2
        val smoothed = mutableListOf<Int>()

        for (i in predictions.indices) {
            val windowStart = (i - halfWindow).coerceAtLeast(0)
            val windowEnd = (i + halfWindow).coerceAtMost(predictions.size - 1)

            val window = predictions.subList(windowStart, windowEnd + 1)
            val majorityClass = window.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: predictions[i]
            smoothed.add(majorityClass)
        }

        return smoothed
    }
}
