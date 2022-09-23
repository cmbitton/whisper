import gradio as gr
import whisper
import langs

lang_list = sorted(langs.LANGUAGES.values())

def transcribe(task, language, model_size, mic, file):
        args = {'task': task}
        if (model_size == 'tiny.en') or (model_size == 'base.en') or (model_size == 'small.en') or (model_size == 'medium.en'):
            args['language'] = 'english'
        elif (language == 'Detect'):
            args['language'] = None
        else:
            args['language'] = language
        model = whisper.load_model(model_size)
        if mic is not None:
            audio = mic
        elif file is not None:
            audio = file
        else:
            return "You must either provide a mic recording or a file"
        result = model.transcribe(audio, **args)
        return result["text"]



demo = gr.Interface(transcribe, 
    inputs=[
        gr.Radio(['transcribe', 'translate'], label= 'Task'), 
        gr.Dropdown(lang_list, value='Detect',  label='Audio Language'),
        gr.Dropdown(['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large'], value='small.en', label='Model Size'), 
        gr.Audio(label='Microphone Recording', source='microphone', type='filepath'), 
        gr.Audio(source='upload', type='filepath', optional=True, label='Audio File')
        ], 
    outputs="text")
demo.launch() 
