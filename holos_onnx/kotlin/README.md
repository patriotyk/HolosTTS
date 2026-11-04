# HolosTTS with ONNX Runtime in Kotlin

A Kotlin library that runs the ONNX bundles on Android with [ONNX Runtime](https://onnxruntime.ai), on the phone's CPU. It is a port of the [`holos-tts-onnx`](../python/README.md) Python package: the same bundles, the same streaming, and on a computer I checked that it gives the same audio as the Python package, sample for sample.

| Folder | Contents |
| --- | --- |
| `holos-onnx/` | The library in Kotlin: `com.holos.tts.HolosOnnx`, with `stream()` and `generate()`. Java can call it too. |
| `demo/` | A demo app in Kotlin: it downloads the INT8 model on the first start and speaks phonemes while streaming. |

The [LiteRT library](../../holos_litert/kotlin/README.md) is faster on a phone with a good GPU. This one needs only the CPU, so it runs on any phone with Android 10 or newer, and there is nothing to compile for the GPU on the first start.

## The model

On a phone I use the INT8 bundle from the [`onnx-int8/`](https://huggingface.co/patriotyk/HolosTTS/tree/main/onnx-int8) folder on Hugging Face: on the CPU it is faster than FP32 and half the size, and it sounds the same. The FP32 bundle in [`onnx/`](https://huggingface.co/patriotyk/HolosTTS/tree/main/onnx) works too. The library loads either the streaming files or `decoder.onnx`:

| File | Contents |
| --- | --- |
| `holos_onnx.json` | Vocabulary, sample rate and the stage files. |
| `voices.npz` | Preset voices. |
| `encoder.onnx`, `prosody.onnx` | Features and durations per token, then F0 and energy per frame. |
| `source.onnx`, `decoder_window.onnx` | The excitation and the decoder for one window, for streaming. |
| `decoder.onnx` | The decoder for the whole utterance, only for `streaming = false`. |

The streaming files of `onnx-int8/` take 320 MB. One utterance can have up to 1000 frames (25 s of speech), so synthesize long texts sentence by sentence.

## Speed on a Pixel 7

A sentence of 10.3 s with the INT8 bundle, on the CPU with ONNX Runtime's default threads:

| Metric | Value |
| --- | --- |
| First chunk of `stream()` | about 0.7 s |
| RTF of `stream()` and of `generate()` with `streaming = true` | about 0.4 |
| RTF of `generate()` with `streaming = false` | about 0.23 |
| Loading | 1.5–1.8 s |
| Memory (PSS) | about 430 MB loaded, 650–750 MB while synthesizing |

4 threads are as fast as the default. Chunks of 20 frames bring the first chunk only about 50 ms sooner, because most of that time goes to the encoder. A phone gets hot under this load, and once it throttles, the RTF grows to about 0.5. LiteRT on the GPU of the same phone gives the first chunk in 0.4 s and an RTF of 0.13. The audio on the phone is the same as on a computer, sample for sample.

## Usage

The library is not published yet. Add the module to your project in `settings.gradle`:

```groovy
include(":holos-onnx")
project(":holos-onnx").projectDir = file("path/to/HolosTTS/holos_onnx/kotlin/holos-onnx")
```

and depend on it with `implementation project(":holos-onnx")`. It brings `onnxruntime-android` with it.

```kotlin
import com.holos.tts.HolosOnnx

// A directory with the files of onnx-int8/ from Hugging Face.
val tts = HolosOnnx(File(context.getExternalFilesDir(null), "onnx-int8"))
val voice = tts.voices.first()
val phonemes = "ˈbudʲ ˈɫaskɐ, zɐteɫefoˈnui̯te meˈnʲi ˈzau̯trɐ ˈpʲisʲlʲɐ oˈbʲidʊ."

// The whole utterance: float samples at tts.sampleRate (24 kHz).
// The voice and speed are optional: the first voice and 1.0 by default.
val audio: FloatArray = tts.generate(phonemes, voice, speed = 1.0f)

// Or chunk by chunk, 1 s each. track is an AudioTrack with ENCODING_PCM_FLOAT and a
// buffer of 2 s, as in the demo: with the minimum buffer, the playback runs dry while
// the next chunk is synthesized.
tts.stream(phonemes, voice) { chunk -> track.write(chunk, 0, chunk.size, AudioTrack.WRITE_BLOCKING) }

tts.close()
```

- **Phonemes:** like the Python packages, the library takes phonemes, not text. Ukrainian stress marks and transcription are only available in Python for now ([ipa-uk](https://github.com/patriotyk/ipa-uk) and [ukrainian-word-stress](https://github.com/patriotyk/ukrainian-word-stress)).
- **Voices:** `getVoice(name)` returns a preset voice vector, and `encode()` also takes a vector, so voices can be mixed or cloned on a computer. `generate()` and `stream()` take what `encode()` returns.
- **Streaming:** each chunk of 40 frames (1 s) is decoded with 20 frames of context on both sides, so the chunks join into the `generate()` audio. `stream(encoded, chunkFrames)` takes smaller chunks, down to one frame, for the first audio sooner. `stream()` drops the boundary-token silence at the start except 10 ms and fades in over 5 ms, unless `trimSilence = false`.
- **Without streaming:** `HolosOnnx(bundle, streaming = false)` loads `decoder.onnx` instead of the window decoder. `generate()` then decodes the utterance in one call, about twice as fast, since the windows decode their context again. `stream()` is not available.
- **Threads:** by default ONNX Runtime picks the number of threads, which on a Pixel 7 is as fast as 4. The third argument takes your own `OrtSession.SessionOptions`, for example with `setIntraOpNumThreads(4)` or an execution provider. Run the synthesis off the main thread.
- **Java:** the same calls work from Java, with `getVoices()`, `getSampleRate()` and the overloads of the optional arguments.

## Building the demo

```bash
cd holos_onnx/kotlin
./gradlew :demo:installDebug
```

On the first start, the demo downloads the streaming files of `onnx-int8/` into `Android/data/com.holos.demo.onnx/files/onnx-int8`. To skip the download, start the app once so that Android creates its folder, stop it and push the files there:

```bash
hf download patriotyk/HolosTTS --include "onnx-int8/*" --local-dir holos
adb push holos/onnx-int8/. /sdcard/Android/data/com.holos.demo.onnx/files/onnx-int8/
```
