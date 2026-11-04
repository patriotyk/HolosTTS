# HolosTTS on Android

HolosTTS runs on the phone's GPU with [LiteRT](https://ai.google.dev/edge/litert). All of the encoder, prosody and decoder graphs have static shapes. The text goes into one of four buckets (64–512 phonemes) and the decoder works in fixed windows, so the GPU compiles every graph once and caches it, and synthesis never recompiles.

| Folder | Contents |
| --- | --- |
| `holos-litert/` | The library in Kotlin: `com.holos.tts.HolosLiteRt`, with `stream()` and `generate()`. Java can call it too. |
| `demo/` | A demo app in Kotlin: it downloads the model on the first start and speaks phonemes while streaming. |

## Speed on a Pixel 7

The 10 s sentence from the main [Benchmarks](../../README.md#benchmarks), on the Mali-G710 GPU in FP16:

| Metric | Value |
| --- | --- |
| RTF of `generate()` | 0.13 |
| First chunk of `stream()` | about 0.4 s |
| Loading, with the GPU programs cached | 3.5 s |
| Loading the first time (compiling for the GPU) | about 9 s |
| GPU memory | about 690 MB |

For comparison, the INT8 ONNX model on the CPU of the same phone gives its first chunk in about 0.63 s and streams with an RTF of about 0.37.

## The model

The bundle is in the [`litert/`](https://huggingface.co/patriotyk/HolosTTS/tree/main/litert) folder on Hugging Face, 270 MB with FP16 weights:

| File | Contents |
| --- | --- |
| `holos_litert.json` | Vocabulary, buckets, decoder windows and the excitation parameters. |
| `encoder.tflite` | Text encoder and durations, one signature per token bucket (64, 128, 256, 512). |
| `prosody.tflite` | F0 and energy, one signature per frame bucket (128, 256, 512, 1000). |
| `decoder.tflite` | The decoder for windows of 80 frames (streaming) and 200 frames (`generate()`). |
| `embeddings.bin`, `voices.bin` | Phoneme embeddings and preset voices, float32. |

One utterance can have up to 510 phonemes and 1000 frames (25 s of speech), so synthesize long texts sentence by sentence.

## Usage

The library is not published yet. Add the module to your project in `settings.gradle`:

```groovy
include(":holos-litert")
project(":holos-litert").projectDir = file("path/to/HolosTTS/holos_litert/kotlin/holos-litert")
```

and depend on it with `implementation project(":holos-litert")`. Your app also needs `jniLibs.useLegacyPackaging = true` in its `android { packaging { } }` block, because LiteRT loads its GPU library by file name.

LiteRT 2.3 is built with Kotlin 2.4, so the Kotlin compiler has to be 2.3 or newer. AGP 9.3 comes with Kotlin 2.2, and I add the newer one in the root `build.gradle`, as in [`build.gradle`](build.gradle):

```groovy
plugins {
    id "org.jetbrains.kotlin.android" version "2.4.20" apply false
}
```

```kotlin
import com.holos.tts.HolosLiteRt

// A directory with the files of litert/ from Hugging Face.
val tts = HolosLiteRt(context, File(context.getExternalFilesDir(null), "litert"))
val voice = tts.voices.first()
val phonemes = "ˈbudʲ ˈɫaskɐ, zɐteɫefoˈnui̯te meˈnʲi ˈzau̯trɐ ˈpʲisʲlʲɐ oˈbʲidʊ."

// The whole utterance: float samples at tts.sampleRate (24 kHz).
// The voice and speed are optional: the first voice and 1.0 by default.
val audio: FloatArray = tts.generate(phonemes, voice, speed = 1.0f)

// Or chunk by chunk: the first one of 1 s after about 0.4 s, then chunks of 4 s, or of 1 s
// on a GPU too slow to compute 4 s before the audio given out so far has played.
// track is an AudioTrack with ENCODING_PCM_FLOAT and a buffer of 2 s, as in the demo:
// with the minimum buffer, the playback runs dry while the next chunk is synthesized.
tts.stream(phonemes, voice) { chunk -> track.write(chunk, 0, chunk.size, AudioTrack.WRITE_BLOCKING) }

tts.close()
```

- **Phonemes:** like the Python packages, the library takes phonemes, not text. Ukrainian stress marks and transcription are only available in Python for now ([ipa-uk](https://github.com/patriotyk/ipa-uk) and [ukrainian-word-stress](https://github.com/patriotyk/ukrainian-word-stress)).
- **Voices:** `getVoice(name)` returns a preset voice vector, and `encode()` also takes a vector, so voices can be mixed or cloned on a computer. `generate()` and `stream()` take what `encode()` returns.
- **Java:** the same calls work from Java, with `getVoices()`, `getSampleRate()` and the overloads of the optional arguments.
- **Threads:** run the synthesis off the main thread, and use one `HolosLiteRt` from one thread at a time.
- **Loading:** the first load compiles the GPU programs into the app's cache directory, and later loads read them from there.

## Building the demo

```bash
cd holos_litert/kotlin
./gradlew :demo:installDebug
```

On the first start, the demo downloads the bundle into `Android/data/com.holos.demo/files/litert`. To skip the download, start the app once so that Android creates its folder, stop it and push the files there:

```bash
hf download patriotyk/HolosTTS --include "litert/*" --local-dir holos
adb push holos/litert/. /sdcard/Android/data/com.holos.demo/files/litert/
```
