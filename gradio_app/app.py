import os
import re
import tempfile
import numpy as np
import torch
from unicodedata import normalize

try:
    import gradio as gr
except ImportError:
    gr = None

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, *args, **kwargs):
            if func is not None:
                return func
            def decorator(f):
                return f
            return decorator

from ipa_uk import ipa
from ukrainian_word_stress import Stressifier, StressSymbol
from holos.models import HolosTTS, VoiceEncoder

try:
    from .verbalizer import Verbalizer
except ImportError:
    from verbalizer import Verbalizer

stressify = Stressifier()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = 'holos.pt' if os.path.isfile('holos.pt') else 'patriotyk/HolosTTS'
ve_path = 'voice_encoder.pt' if os.path.isfile('voice_encoder.pt') else 'patriotyk/HolosTTS'

model = HolosTTS(model_path, device=device)
voice_encoder = VoiceEncoder(ve_path, device=device)

# Loaded on the first click: it downloads its own model.
verbalizer = None


def split_to_parts(text, group=True):
    text = re.sub(r'(\w+[^.,!:?\-])\n', r'\1. ', text)
    text = text.replace('\n', ' ')
    split_symbols = '.?!:'
    parts = ['']
    index = 0
    last = len(text) - 1
    for i, s in enumerate(text):
        parts[index] += s
        if s in split_symbols and i < last and text[i + 1] == ' ':
            if group and len(parts[index]) <= 100:
                continue
            index += 1
            parts.append('')
    return parts


def verbalize(text):
    global verbalizer
    if verbalizer is None:
        verbalizer = Verbalizer()
    verbalized = ''
    for part in split_to_parts(text, group=False):
        if part.strip():
            verbalized += verbalizer.process_text(part.strip())[0] + ' '
    return verbalized.strip()


description = '''
<h1 style="text-align:center;">HolosTTS: синтез української мови</h1>

<p style="text-align:center;">
<a href="https://github.com/patriotyk/HolosTTS">GitHub</a> ·
<a href="https://huggingface.co/patriotyk/HolosTTS">Моделі на Hugging Face</a> ·
<a href="https://pypi.org/project/holos-tts/">PyPI</a>
</p>

HolosTTS — моя відкрита модель синтезу української мови з найкращим співвідношенням якості та швидкості:
на відеокарті RTX 3090 перший шматок аудіо готовий за 5 мс. Модель має набір готових голосів і може
клонувати голос з короткого аудіозразка. Вона працює на PyTorch або на ONNX Runtime без PyTorch:
на CPU, CUDA і TensorRT.

- **Наголоси** розставляються автоматично, але деякі можуть бути неправильними. Тоді поставте `+` після
  наголошеного голосного, наприклад «Му+дрого».
- **Цифри, дати і скорочення** модель читає погано. Кнопка «Вербалізувати» (бета) замінює їх словами.
- **Швидкість** більше одиниці робить мовлення швидшим, менше одиниці — повільнішим.
- **Клонування голосу:** завантажте або запишіть кілька секунд чистого мовлення без музики і сторонніх
  голосів (до 30 с). Отриманий голос можна завантажити як файл `.pt` і використовувати в своєму коді.
'''

examples = [
    ["Решта окупантів звернула на Вокзальну — центральну вулицю Бучі. Тільки уявіть їхній настрій, коли перед ними відкрилася ця пасторальна картина! Невеличкі котеджі й просторіші будинки шикуються обабіч, перед ними вивищуються голі липи та електро-стовпи, тягнуться газони й жовто-чорні бордюри. Доглянуті сади визирають із-поза зелених парканів, гавкотять собаки, співають птахи… На дверях будинку номер тридцять шість досі висить різдвяний вінок.", 1.0],
    ["Коли майданчик готують до зйомки, іноді виникає незручний момент. Команда розставляє все по місцях, а акторів — у певні положення, і вони мають дивитися одне на одного, щоб команда могла правильно освітити сцену.", 1.0],
    ["Я канєшна ізвіняюсь, таваріщ Інтєліґєнт, — досить уїдливо промовив Кєша. — НО ЕТА НІКАВО НЄ ЄБЬОТ!", 1.0],
    ["Дмитро завмер, відчуваючи, як по спині пробіг крижаний вітер. Забуття. Що це означає? Хто залишив цей файл у його особистому архіві? І чому саме це слово? Мерщій біжіть сюди! Мерщій біжіть сюди.", 1.0],
    ["Одна дівчинка стала королевою Франції. Звали її Анна, і була вона донькою Ярослава Му+дрого, великого київського князя. Він опі+кувався літературою та культурою в Київській Русі+, а тоді переважно про таке не дбали – більше воювали і споруджували фортеці.", 1.0],
    ["Одна дівчинка народилася і виросла в Америці, та коли стала дорослою, зрозуміла, що дуже любить українські вірші й найбільше хоче робити вистави про Україну. Звали її Вірляна. Дід Вірляни був український мовознавець і педагог Кость Кисілевський, котрий навчався в Лейпцизькому та Віденському університетах і, після Другої світової війни виїхавши до США, започаткував систему шкіл українознавства по всій Америці. Тож Вірляна зростала в українському середовищі, а окрім того – в середовищі вихідців з інших країн.", 1.0],
    ["За інформацією від Державної служби з надзвичайних ситуацій станом на 7 ранку 15 липня.", 1.0],
    ["Очікується, що цей застосунок буде запущено 22.08.2025.", 1.0],
]


@spaces.GPU
def encode_audio_prompt(audio_path):
    if not audio_path:
        return None, gr.update(value=None, visible=False)
    try:
        voice_tensor = voice_encoder.encode(audio_path)
        voice_tensor_cpu = voice_tensor.squeeze(0).cpu()

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        torch.save(voice_tensor_cpu, temp_file.name)
        temp_file.close()

        return voice_tensor_cpu, gr.update(value=temp_file.name, visible=True)
    except Exception as e:
        print(f"Error encoding audio prompt: {e}")
        raise gr.Error(f"Помилка кодування аудіо промта: {str(e)}")


@spaces.GPU
def synthesize(text, speed, voice_name, custom_voice=None, progress=gr.Progress()):
    if text.strip() == "":
        raise gr.Error("Введіть текст")
    if len(text) > 50000:
        raise gr.Error("Текст мусить бути <50k символів")
    print("*** saying ***")
    print(text)
    print("*** end ***")

    voice = custom_voice if custom_voice is not None else voice_name

    result_wav = []
    for t in progress.tqdm(split_to_parts(text)):
        t = t.strip()
        t = t.replace('"', '')
        if t:
            t = t.replace('+', StressSymbol.CombiningAcuteAccent)
            t = normalize('NFKC', t)

            t = re.sub(r'[᠆‐‑‒–—―⁻₋−⸺⸻]', '-', t)
            if t[-1] not in '.?!:-':
                t += '.'
            t = re.sub(r' - ', ': ', t)
            t = stressify(t)
            ps = ipa(t)

            if ps:
                print(ps)
                wav = model.generate(ps, voice, speed=speed).squeeze()
                result_wav.append(wav)

    data = torch.concatenate(result_wav).cpu().numpy()
    data = data / np.abs(data).max()
    data = data * 32767
    data = data.astype(np.int16)

    return 24000, data


def select_example(df, evt: gr.SelectData):
    return evt.row_value


def create_demo():
    if gr is None:
        raise ImportError("Gradio is not installed: pip install -r gradio_app/requirements.txt")

    with gr.Blocks(title="HolosTTS") as demo:
        gr.Markdown(description)
        style_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Text(label='Текст:', lines=5, max_lines=10)
                verbalize_button = gr.Button("Вербалізувати (бета)")
                speed = gr.Slider(label='Швидкість:', minimum=0.7, maximum=1.3, value=1.0, step=0.05)
                speaker = gr.Dropdown(
                    label="Голос (пресет):",
                    choices=list(model.voices.keys()),
                    value=list(model.voices.keys())[0] if model.voices else None
                )
                audio_prompt = gr.Audio(label="Аудіо промт (клонування голосу):", type="filepath", sources=["upload", "microphone"])
                style_download = gr.File(label="Завантажити згенерований голос (.pt):", visible=False)

            with gr.Column(scale=1):
                output_audio = gr.Audio(
                    label="Аудіо:",
                    autoplay=False,
                    streaming=False,
                    type="numpy",
                )
                synthesise_button = gr.Button("Синтезувати", variant="primary")

        verbalize_button.click(verbalize, inputs=[input_text], outputs=[input_text])
        synthesise_button.click(synthesize, inputs=[input_text, speed, speaker, style_state], outputs=[output_audio])
        audio_prompt.change(encode_audio_prompt, inputs=[audio_prompt], outputs=[style_state, style_download])

        with gr.Row():
            examples_table = gr.Dataframe(wrap=True, headers=["Текст", "Швидкість"], datatype=["str", "number"], value=examples, interactive=False, show_label=True, label="Приклади:")
            examples_table.select(select_example, inputs=[examples_table], outputs=[input_text, speed])

    return demo


def main():
    if gr is None:
        raise ImportError("Gradio is not installed: pip install -r gradio_app/requirements.txt")
    demo = create_demo()
    demo.queue(api_open=True, max_size=15).launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
