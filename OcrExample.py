import google.generativeai as genai
import whisper

model = whisper.load_model("tiny")
audio_path = "C:\Users/arlle/OneDrive/Área de Trabalho/IA-transcri-o-main/IA-transcri-o-main/gravação1.mp3"

API_KEY = 'AIzaSyCGW81sQiyNEp5ZjkreCU5l0rijOTBexAE'
genai.configure(api_key=API_KEY)

def pergunta():
    result = model.transcribe(audio_path)
    transcript = result["text"]

    modelo = genai.GenerativeModel('gemini-pro')
    pergunta = ('Qual o tema do discurso? \n'
                'Os dados mostrados são fake news? (Responda com sim ou não) \n'
                'Se não são fake news, mostre esses dados e as possíveis fontes que confirmam a veracidade: (As fontes não precisam ser links específicos, mas que levem a uma base confiável desses dados citados)'
                'Quem é citado no discurso:')
    transcricao = pergunta + "\n" + transcript
    resposta = modelo.generate_content(transcricao)

    print(f"Resposta: \n{resposta.text}\n")
    print("Transcrição: \n{}".format(transcript.replace('. ', '.\n').replace('? ', '?\n').replace('! ', '!\n')))

pergunta()
