package com.holos.demo

import android.app.Activity
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.text.InputType
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import com.holos.tts.HolosOnnx
import org.json.JSONObject
import java.io.File
import java.net.URL
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Speaks phonemes with HolosOnnx on the CPU, streaming the audio as it is
 * generated.
 *
 * On the first start it downloads the streaming files of the INT8 bundle from
 * the onnx-int8/ folder of the Hugging Face repository into the app's external
 * files directory; a bundle pushed there with adb is used as is.
 */
class MainActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var status: TextView
    private lateinit var phonemes: EditText
    private lateinit var voices: Spinner
    private lateinit var speak: Button
    private var tts: HolosOnnx? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val padding = (16 * resources.displayMetrics.density).toInt()
        status = TextView(this)
        phonemes = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            setText(SAMPLE_PHONEMES)
        }
        voices = Spinner(this)
        speak = Button(this).apply {
            text = "Speak"
            isEnabled = false
            setOnClickListener { speak() }
        }
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(padding, padding, padding, padding)
            addView(TextView(context).apply { text = "Phonemes" })
            addView(phonemes, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
            addView(voices, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
            addView(speak, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
            addView(status, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        }
        // Keeps the views clear of the status and navigation bars.
        setContentView(ScrollView(this).apply {
            fitsSystemWindows = true
            addView(layout)
        })
        worker.execute(::load)
    }

    override fun onDestroy() {
        worker.execute { tts?.close() }
        worker.shutdown()
        super.onDestroy()
    }

    private fun load() {
        try {
            val bundle = File(getExternalFilesDir(null), "onnx-int8")
            download(bundle)
            show("Loading the models…")
            val start = System.nanoTime()
            val loaded = HolosOnnx(bundle)
            tts = loaded
            val seconds = (System.nanoTime() - start) / 1e9
            runOnUiThread {
                voices.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, loaded.voices)
                speak.isEnabled = true
                status.text = String.format(Locale.US, "Loaded in %.1f s", seconds)
            }
        } catch (error: Exception) {
            show("Failed to load: $error")
        }
    }

    /** Downloads the streaming files of the bundle that are not in the directory yet. */
    private fun download(bundle: File) {
        bundle.mkdirs()
        val manifest = File(bundle, MANIFEST)
        if (!manifest.exists()) {
            fetch(MANIFEST, manifest)
        }
        val json = JSONObject(manifest.readText())
        val streaming = json.getJSONObject("streaming")
        val files = listOf(json.getString("encoder"), json.getString("prosody"), json.getString("voices"),
            streaming.getString("source"), streaming.getString("decoder"))
        for ((index, name) in files.withIndex()) {
            val file = File(bundle, name)
            if (!file.exists()) {
                show("Downloading $name (${index + 1}/${files.size})…")
                fetch(name, file)
            }
        }
    }

    private fun fetch(name: String, target: File) {
        val partial = File(target.path + ".part")
        URL(REPOSITORY + name).openStream().use { input ->
            partial.outputStream().use { output -> input.copyTo(output, 1 shl 20) }
        }
        check(partial.renameTo(target)) { "Cannot write $target" }
    }

    private fun speak() {
        val loaded = tts ?: return
        val text = phonemes.text.toString()
        val voice = voices.selectedItem as String
        speak.isEnabled = false
        worker.execute {
            val track = AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(loaded.sampleRate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                .setTransferMode(AudioTrack.MODE_STREAM)
                // Room for two chunks of 1 s: the next chunk is synthesized while
                // the buffer plays, so the playback does not run dry between chunks.
                .setBufferSizeInBytes(2 * loaded.sampleRate * 4)
                .build()
            try {
                val start = System.nanoTime()
                var firstChunk = 0L
                var playing = 0L
                var samples = 0
                var chunks = 0
                loaded.stream(text, voice) { chunk ->
                    if (firstChunk == 0L) firstChunk = System.nanoTime() - start
                    samples += chunk.size
                    chunks++
                    // A blocking write waits for the playback, so its time is not synthesis.
                    val before = System.nanoTime()
                    track.write(chunk, 0, chunk.size, AudioTrack.WRITE_BLOCKING)
                    // Starts once the first chunk is buffered, not on an empty buffer.
                    if (track.playState != AudioTrack.PLAYSTATE_PLAYING) track.play()
                    playing += System.nanoTime() - before
                }
                val synthesis = (System.nanoTime() - start - playing) / 1e9
                val audio = samples.toDouble() / loaded.sampleRate
                // Before the buffer runs out at the end, which counts as one more.
                val underruns = track.underrunCount
                show(String.format(Locale.US, "First chunk in %.0f ms, %.1f s of audio in %d chunks, RTF %.3f, underruns %d",
                    firstChunk / 1e6, audio, chunks, synthesis / audio, underruns))
                // Up to 2 s are still buffered, and release() would cut them off.
                while (track.playbackHeadPosition < samples) Thread.sleep(20)
                track.stop()
            } catch (error: Exception) {
                show("Failed: $error")
            } finally {
                track.release()
                runOnUiThread { speak.isEnabled = true }
            }
        }
    }

    private fun show(message: String) = runOnUiThread { status.text = message }

    private companion object {
        const val REPOSITORY = "https://huggingface.co/patriotyk/HolosTTS/resolve/main/onnx-int8/"
        const val MANIFEST = "holos_onnx.json"

        // "Будь ласка, зателефонуйте мені завтра після обіду, коли матимете вільну хвилинку."
        const val SAMPLE_PHONEMES = "ˈbudʲ ˈɫaskɐ, zɐteɫefoˈnui̯te meˈnʲi ˈzau̯trɐ ˈpʲisʲlʲɐ oˈbʲidʊ, " +
            "kɔˈɫɪ ˈmatɪmete ˈʋʲilʲnʊ xʋɪˈɫɪnkʊ."
    }
}
