package com.holos.tts

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.SplittableRandom
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tanh

/**
 * HolosTTS on the GPU with LiteRT, for the bundles in the litert/ folder of
 * the Hugging Face repository.
 *
 * The encoder, prosody and decoder graphs have static shapes (buckets for the
 * text and a fixed decoder window), so the GPU compiles each of them once.
 * The embedding lookups, durations, frame alignment, harmonic excitation and
 * decoder windows are computed here. The input is a phoneme string, as for
 * the Python packages.
 *
 * Loading a [bundle] directory compiles the GPU programs and caches them in
 * [cacheDirectory], so later loads are faster. [fp16] = false computes in
 * FP32, slower and with no audible difference. An instance is not
 * thread-safe, and every call blocks until the GPU is done, so call it off
 * the main thread.
 */
class HolosLiteRt @JvmOverloads constructor(
    context: Context,
    bundle: File,
    cacheDirectory: File = File(context.cacheDir, "holos_litert"),
    fp16: Boolean = true,
) : AutoCloseable {

    /** Receives the audio chunk by chunk, as float samples at [sampleRate]. */
    fun interface ChunkListener {
        fun onChunk(audio: FloatArray)
    }

    /** Features, F0, energy and durations of one utterance. */
    class Encoded internal constructor(
        internal val acoustic: FloatArray,  // [channels][frames]
        internal val f0: FloatArray,        // [2 * frames]
        internal val energy: FloatArray,    // [2 * frames]
        internal val voice: FloatArray,     // [ACOUSTIC_VOICE_DIM]
        private val tokenDurations: IntArray,
        /** Frames of audio, each [samplesPerFrame] samples. */
        val frames: Int,
    ) {
        /** Frames per token, including the boundary tokens at both ends. */
        val durations: IntArray get() = tokenDurations.clone()
    }

    private val manifest = JSONObject(File(bundle, "holos_litert.json").readText()).also {
        require(it.getString("format") == "holos-litert-gpu" && it.getInt("format_version") == 2) {
            "Unsupported bundle $bundle"
        }
    }

    /** Samples per second of the audio. */
    val sampleRate = manifest.getInt("sample_rate")

    /** Samples of audio per frame of features. */
    val samplesPerFrame = manifest.getInt("samples_per_frame")

    private val vocab = manifest.getJSONArray("vocab").let { names ->
        (0 until names.length()).associateBy { names.getString(it) }
    }
    private val symbols = manifest.getInt("n_symbols")
    private val channels = manifest.getInt("channels")
    private val prosodicTable: FloatArray
    private val acousticTable: FloatArray
    private val voiceDim: Int
    private val presets = LinkedHashMap<String, FloatArray>()
    private val tokenBuckets = manifest.getJSONObject("encoder").getJSONArray("buckets").ints()
    private val frameBuckets = manifest.getJSONObject("prosody").getJSONArray("buckets").ints()
    private val nFft = manifest.getJSONObject("stft").getInt("n_fft")
    private val environment: Environment
    private val encoder: CompiledModel
    private val prosody: CompiledModel
    private val decoder: CompiledModel
    private val buffers = HashMap<String, Io>()
    // Not synchronized, unlike Random, whose nextGaussian slowed the excitation.
    private val random = SplittableRandom()

    init {
        val tables = readFloats(File(bundle, manifest.getJSONObject("embeddings").getString("file")))
        prosodicTable = tables.copyOfRange(0, symbols * channels)
        acousticTable = tables.copyOfRange(symbols * channels, 2 * symbols * channels)
        val voiceInfo = manifest.getJSONObject("voices")
        voiceDim = voiceInfo.getInt("dim")
        val voiceValues = readFloats(File(bundle, voiceInfo.getString("file")))
        val names = voiceInfo.getJSONArray("names")
        for (i in 0 until names.length()) {
            presets[names.getString(i)] = voiceValues.copyOfRange(i * voiceDim, (i + 1) * voiceDim)
        }

        environment = Environment.create(context)
        cacheDirectory.mkdirs()
        val compiled = ArrayList<CompiledModel>()
        try {
            for (stage in listOf("encoder", "prosody", "decoder")) {
                compiled += compile(bundle, stage, cacheDirectory, fp16)
            }
        } catch (error: Exception) {
            compiled.forEach { it.close() }
            environment.close()
            throw error
        }
        encoder = compiled[0]
        prosody = compiled[1]
        decoder = compiled[2]
    }

    private fun compile(bundle: File, stage: String, cache: File, fp16: Boolean): CompiledModel {
        val file = File(bundle, manifest.getJSONObject(stage).getString("file"))
        val options = CompiledModel.Options(Accelerator.GPU, Accelerator.CPU)
        val precision = if (fp16) CompiledModel.GpuOptions.Precision.FP16 else CompiledModel.GpuOptions.Precision.FP32
        // The signatures (buckets) of a stage share its weights on the GPU. The
        // cache key changes with the file, so a new bundle is compiled again.
        val cacheKey = "${file.name}_${file.length()}_${file.lastModified()}_$precision"
        options.gpuOptions = CompiledModel.GpuOptions(
            true, null, null, precision, null, null,
            cache.absolutePath, cacheKey, true, null,
            null, null, CompiledModel.GpuOptions.Backend.OPENCL, null, null,
        )
        return CompiledModel.create(file.absolutePath, options, environment)
    }

    /** The names of the preset voices. */
    val voices: List<String> get() = presets.keys.toList()

    /** A preset voice's vector, for mixing voices. */
    fun getVoice(name: String): FloatArray =
        requireNotNull(presets[name]) { "Unknown voice $name" }.clone()

    /** The input and output buffers of one signature. */
    private class Io(val inputs: List<TensorBuffer>, val outputs: List<TensorBuffer>)

    private fun buffers(model: CompiledModel, signature: String): Io =
        buffers.getOrPut(signature) { Io(model.createInputBuffers(signature), model.createOutputBuffers(signature)) }

    /** Phonemes -> features per frame, F0, energy and durations, with a preset voice (null for the first). */
    @JvmOverloads
    fun encode(phonemes: String, voice: String? = null, speed: Float = 1f): Encoded =
        encode(phonemes, if (voice == null) presets.values.first() else getVoice(voice), speed)

    /** Phonemes -> features per frame, F0, energy and durations, with a voice vector. */
    @JvmOverloads
    fun encode(phonemes: String, voice: FloatArray, speed: Float = 1f): Encoded {
        require(speed > 0 && speed.isFinite()) { "speed must be a positive finite number" }
        require(voice.size == voiceDim) { "A voice has $voiceDim values" }
        val tokens = ArrayList<Int>()
        tokens += 0
        phonemes.codePoints().forEach { point -> vocab[String(Character.toChars(point))]?.let { tokens += it } }
        tokens += 0
        val length = tokens.size
        val capacity = bucket(tokenBuckets, length, "tokens")
        val prosodicEmbedded = FloatArray(capacity * channels)
        val acousticEmbedded = FloatArray(capacity * channels)
        for (i in 0 until capacity) {
            val token = if (i < length) tokens[i] else 0
            require(token < symbols) { "The text has symbols the model has no embedding for" }
            prosodicTable.copyInto(prosodicEmbedded, i * channels, token * channels, (token + 1) * channels)
            acousticTable.copyInto(acousticEmbedded, i * channels, token * channels, (token + 1) * channels)
        }
        var io = buffers(encoder, "encoder_$capacity")
        io.inputs[0].writeFloat(prosodicEmbedded)
        io.inputs[1].writeFloat(acousticEmbedded)
        io.inputs[2].writeInt(intArrayOf(length))
        io.inputs[3].writeFloat(voice)
        io.inputs[4].writeFloat(floatArrayOf(speed))
        encoder.run(io.inputs, io.outputs, "encoder_$capacity")
        val acousticTokens = io.outputs[0].readFloat()  // [channels][capacity]
        val prosodicTokens = io.outputs[1].readFloat()  // [capacity][channels]
        val expected = io.outputs[2].readFloat()

        // Half to even, like torch.round.
        val durations = IntArray(length) { Math.rint(expected[it].toDouble()).toInt() }
        val frames = durations.sum()
        require(frames > 0) { "The text produced no frames" }
        val index = IntArray(frames)
        var frame = 0
        for (token in 0 until length) {
            repeat(durations[token]) { index[frame++] = token }
        }
        val frameCapacity = bucket(frameBuckets, frames, "frames")
        val aligned = FloatArray(channels * frameCapacity)
        for (f in 0 until frames) {
            val row = index[f] * channels
            for (c in 0 until channels) {
                aligned[c * frameCapacity + f] = prosodicTokens[row + c]
            }
        }
        io = buffers(prosody, "prosody_$frameCapacity")
        io.inputs[0].writeFloat(aligned)
        io.inputs[1].writeInt(intArrayOf(frames))
        io.inputs[2].writeFloat(voice)
        prosody.run(io.inputs, io.outputs, "prosody_$frameCapacity")
        val f0 = io.outputs[0].readFloat().copyOf(2 * frames)
        val energy = io.outputs[1].readFloat().copyOf(2 * frames)

        val acoustic = FloatArray(channels * frames)
        for (f in 0 until frames) {
            for (c in 0 until channels) {
                acoustic[c * frames + f] = acousticTokens[c * capacity + index[f]]
            }
        }
        return Encoded(acoustic, f0, energy, voice.copyOf(ACOUSTIC_VOICE_DIM), durations, frames)
    }

    /**
     * Harmonic-plus-noise excitation of one utterance, as SineGen, computed
     * block by block on demand, so the first window does not wait for the
     * whole utterance. The phase at the start of every block of constant F0
     * is accumulated in double; inside a block each harmonic advances by a
     * fixed angle, so its sine follows by rotation instead of sin().
     */
    private inner class Excitation(val f0: FloatArray, private val noise: Boolean) {
        private val params = manifest.getJSONObject("source")
        private val upsample = params.getInt("upsample")
        private val harmonics = params.getInt("harmonics")
        private val sineAmp = params.getDouble("sine_amp").toFloat()
        private val noiseStd = params.getDouble("noise_std").toFloat()
        private val threshold = params.getDouble("voiced_threshold").toFloat()
        private val weight = params.getJSONArray("merge_weight").let { w -> FloatArray(harmonics) { w.getDouble(it).toFloat() } }
        private val norm: Double = sqrt(weight.fold(0.0) { sum, w -> sum + w * w })
        private val bias = params.getDouble("merge_bias").toFloat()
        private val start = DoubleArray(harmonics)
        private val merged = DoubleArray(upsample)
        private var blocks = 0
        val samples = FloatArray(f0.size * upsample)

        /** Computes the samples of the blocks before toBlock. */
        fun ensure(toBlock: Int) {
            while (blocks < min(toBlock, f0.size)) {
                val value = f0[blocks]
                val voiced = value > threshold
                merged.fill(bias.toDouble())
                for (h in 0 until harmonics) {
                    val cycles = value * (h + 1) / sampleRate.toFloat()
                    val rad = cycles - floor(cycles)
                    if (voiced) {
                        val step = 2 * PI * rad
                        val angle = 2 * PI * start[h] + step
                        var cosine = cos(angle)
                        var sine = sin(angle)
                        val stepCos = cos(step)
                        val stepSin = sin(step)
                        val scale = (weight[h] * sineAmp).toDouble()
                        for (j in 0 until upsample) {
                            merged[j] += scale * sine
                            val rotated = cosine * stepCos - sine * stepSin
                            sine = sine * stepCos + cosine * stepSin
                            cosine = rotated
                        }
                    }
                    val advance = rad.toDouble() * upsample
                    start[h] = (start[h] + advance - floor(advance)) % 1.0
                }
                val amplitude = if (voiced) noiseStd else sineAmp / 3
                val offset = blocks * upsample
                for (j in 0 until upsample) {
                    var sample = merged[j]
                    if (noise) {
                        sample += amplitude * norm * gaussian()
                    }
                    samples[offset + j] = tanh(sample).toFloat()
                }
                blocks++
            }
        }
    }

    private var spareGaussian = 0.0
    private var hasSpare = false

    /** A standard normal value (Marsaglia's polar method). */
    private fun gaussian(): Double {
        if (hasSpare) {
            hasSpare = false
            return spareGaussian
        }
        var u: Double
        var v: Double
        var radius: Double
        do {
            u = 2 * random.nextDouble() - 1
            v = 2 * random.nextDouble() - 1
            radius = u * u + v * v
        } while (radius >= 1 || radius == 0.0)
        val factor = sqrt(-2 * ln(radius) / radius)
        spareGaussian = v * factor
        hasSpare = true
        return u * factor
    }

    /** One decoder signature with its buffers. */
    private inner class Window(config: JSONObject) {
        val signature: String = config.getString("signature")
        val width = config.getInt("window_frames")
        val chunk = config.getInt("chunk_frames")
        val aligned = FloatArray(channels * width)
        val f0 = FloatArray(2 * width)
        val energy = FloatArray(2 * width)
        val padded = FloatArray(samplesPerFrame * width + nFft)
        val io = buffers(decoder, signature)
    }

    /** The chunk from start to end, computed by window from windowStart. */
    private class Plan(val window: Window, val start: Int, val end: Int, val windowStart: Int)

    private val decoderInfo by lazy(LazyThreadSafetyMode.NONE) { manifest.getJSONObject("decoder") }
    private val small by lazy(LazyThreadSafetyMode.NONE) { Window(decoderInfo.getJSONObject("stream")) }
    private val large by lazy(LazyThreadSafetyMode.NONE) { Window(decoderInfo.getJSONObject("generate")) }
    private val contextFrames by lazy(LazyThreadSafetyMode.NONE) { decoderInfo.getInt("context_frames") }

    /**
     * The decoder window for the chunk from start.
     *
     * Every window has context_frames of context before and after its chunk,
     * or reaches the edge of the utterance, so the chunks join seamlessly
     * whatever window computed them. The frames that fit in one small window
     * go into it in one piece. A window that runs past the end of a short
     * utterance is padded, which only approximates its end.
     */
    private fun planWindow(frames: Int, start: Int, allowLarge: Boolean): Plan {
        val window: Window
        val end: Int
        if (frames - max(0, start - contextFrames) <= small.width) {
            window = small
            end = frames
        } else if (!allowLarge) {
            window = small
            end = min(start + small.chunk, frames)
        } else {
            window = large
            end = min(start + large.chunk, frames)
        }
        return Plan(window, start, end, max(0, min(start - contextFrames, frames - window.width)))
    }

    /**
     * Decodes the utterance window by window. generate() uses the large
     * windows. stream() starts with a small one, to give the first chunk
     * sooner, and takes a large one, with less context recomputed per second
     * of audio, once the audio it has given out and not yet played covers
     * the time of a large window; on a slow GPU it stays with small windows,
     * so the playback does not run dry. It counts the audio as playing from
     * the first chunk on, without the skipped samples.
     */
    private fun decode(encoded: Encoded, stream: Boolean, noise: Boolean, skipped: Int, listener: ChunkListener) {
        val frames = encoded.frames
        val step = samplesPerFrame
        // Without padding, so a stream joins exactly like generate().
        val largeFits = frames >= large.width
        val excitation = Excitation(encoded.f0, noise)
        var plan = planWindow(frames, 0, !stream)
        paddedExcitation(excitation, plan)
        var firstChunk = 0L
        var given = -skipped.toLong()
        var smallSeconds = 0.0
        var largeSeconds = 0.0
        while (true) {
            val begin = System.nanoTime()
            val window = plan.window
            val width = window.width
            val windowStart = plan.windowStart
            val used = min(width, frames - windowStart)
            // Past the end: the last token and F0 repeated, no energy or excitation.
            for (c in 0 until channels) {
                for (i in 0 until width) {
                    window.aligned[c * width + i] = encoded.acoustic[c * frames + windowStart + min(i, used - 1)]
                }
            }
            for (i in 0 until 2 * width) {
                val at = 2 * windowStart + min(i, 2 * used - 1)
                window.f0[i] = encoded.f0[at]
                window.energy[i] = if (i < 2 * used) encoded.energy[at] else 0f
            }
            val io = window.io
            io.inputs[0].writeFloat(window.aligned)
            io.inputs[1].writeFloat(window.f0)
            io.inputs[2].writeFloat(window.energy)
            io.inputs[3].writeFloat(window.padded)
            io.inputs[4].writeFloat(encoded.voice)
            decoder.run(io.inputs, io.outputs, window.signature)
            // The excitation the next window needs at most, while the GPU decodes this one.
            if (plan.end < frames) {
                val next = planWindow(frames, plan.end, true)
                excitation.ensure(2 * (next.windowStart + min(large.width, frames - next.windowStart)))
            }
            val audio = io.outputs[0].readFloat()
            val seconds = (System.nanoTime() - begin) / 1e9
            if (window === large) largeSeconds = seconds else smallSeconds = seconds
            val offset = (plan.start - windowStart) * step
            if (firstChunk == 0L) {
                firstChunk = System.nanoTime()
            }
            given += (plan.end - plan.start).toLong() * step
            listener.onChunk(audio.copyOfRange(offset, offset + (plan.end - plan.start) * step))
            if (plan.end >= frames) {
                return
            }
            var allowLarge = !stream
            if (stream && largeFits) {
                val buffered = given / sampleRate.toDouble() - (System.nanoTime() - firstChunk) / 1e9
                val expected = if (largeSeconds > 0) largeSeconds else smallSeconds * large.width / small.width
                allowLarge = buffered >= LARGE_WINDOW_MARGIN * expected
            }
            plan = planWindow(frames, plan.end, allowLarge)
            paddedExcitation(excitation, plan)
        }
    }

    /** A window's excitation, reflect-padded by nFft / 2 for the decoder's STFT, into its input array. */
    private fun paddedExcitation(excitation: Excitation, plan: Plan) {
        val windowStart = plan.windowStart
        val used = min(plan.window.width, excitation.f0.size / 2 - windowStart)
        excitation.ensure(2 * (windowStart + used))
        val padded = plan.window.padded
        val pad = nFft / 2
        val length = padded.size - 2 * pad
        padded.fill(0f)
        excitation.samples.copyInto(padded, pad, samplesPerFrame * windowStart, samplesPerFrame * (windowStart + used))
        for (i in 0 until pad) {
            padded[i] = padded[2 * pad - i]
            padded[pad + length + i] = padded[pad + length - 2 - i]
        }
    }

    /** The whole utterance with a preset voice (null for the first). */
    @JvmOverloads
    fun generate(phonemes: String, voice: String? = null, speed: Float = 1f): FloatArray =
        generate(encode(phonemes, voice, speed))

    /** The whole utterance of an encoded text; noise = false gives reproducible output. */
    @JvmOverloads
    fun generate(encoded: Encoded, noise: Boolean = true): FloatArray {
        val audio = FloatArray(encoded.frames * samplesPerFrame)
        var position = 0
        decode(encoded, false, noise, 0) { chunk ->
            chunk.copyInto(audio, position)
            position += chunk.size
        }
        return audio
    }

    /** Streams the utterance with a preset voice (null for the first). */
    @JvmOverloads
    fun stream(phonemes: String, voice: String? = null, speed: Float = 1f, listener: ChunkListener) =
        stream(encode(phonemes, voice, speed), true, listener)

    /**
     * Streams an encoded utterance: the first chunk of 1 s as soon as
     * possible, then chunks of 1 s or, when the GPU is fast enough, 4 s. The
     * leading silence of the boundary token is dropped except 10 ms and
     * faded in over 5 ms. noise = false gives reproducible output.
     */
    @JvmOverloads
    fun stream(encoded: Encoded, noise: Boolean = true, listener: ChunkListener) {
        val skipped = max(0, encoded.durations[0] * samplesPerFrame
                - Math.round(SILENCE_MARGIN_SECONDS * sampleRate).toInt())
        var skip = skipped
        val fade = if (skipped > 0) FADE_IN_SAMPLES else 0
        var faded = 0
        decode(encoded, true, noise, skipped) { chunk ->
            val from = min(skip, chunk.size)
            skip -= from
            val trimmed = chunk.copyOfRange(from, chunk.size)
            var i = 0
            while (i < trimmed.size && faded < fade) {
                faded++
                trimmed[i] *= faded / fade.toFloat()
                i++
            }
            if (trimmed.isNotEmpty()) {
                listener.onChunk(trimmed)
            }
        }
    }

    override fun close() {
        for (io in buffers.values) {
            (io.inputs + io.outputs).forEach { it.close() }
        }
        buffers.clear()
        encoder.close()
        prosody.close()
        decoder.close()
        environment.close()
    }

    private companion object {
        const val SILENCE_MARGIN_SECONDS = 0.010
        const val FADE_IN_SAMPLES = 120
        const val ACOUSTIC_VOICE_DIM = 128

        /**
         * Seconds of buffered audio that stream() needs, relative to the
         * expected time of a large window, to switch to large windows.
         */
        const val LARGE_WINDOW_MARGIN = 1.5

        fun bucket(buckets: IntArray, length: Int, what: String): Int =
            buckets.firstOrNull { length <= it } ?: throw IllegalArgumentException(
                "$length $what exceed the largest bucket (${buckets.last()}); split the text into sentences")

        fun JSONArray.ints() = IntArray(length()) { getInt(it) }

        fun readFloats(file: File): FloatArray {
            val bytes = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return FloatArray(bytes.remaining()).also { bytes.get(it) }
        }
    }
}
