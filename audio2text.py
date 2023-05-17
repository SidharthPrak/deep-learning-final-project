
import pyaudio
from queue import Queue
from threading import Thread
import json
from vosk import Model, KaldiRecognizer
import time
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import ftfy
import numpy as np
from keras.models import  load_model
import re
from nltk.tokenize import word_tokenize
import nltk

nltk.download("stopwords")
nltk.download("punkt")
vocab_f = 'glove.6B.50d.txt'
embeddings_index = {}
with open(vocab_f, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    
model = load_model("your_model_text20.h5")
class Transcriber:

    def __init__(self, channels, frame_rate, record_seconds, audio_format, sample_size, chunk , microphone_index):

        self.messages = Queue()
        self.recordings = Queue()

        self.model = Model(model_name="vosk-model-en-us-0.22")
        self.rec = KaldiRecognizer(self.model, frame_rate)
        self.rec.SetWords(True)

        self.channels = channels
        self.frame_rate = frame_rate
        self.record_seconds = record_seconds
        self.audio_format = audio_format
        self.sample_size = sample_size
        self.chunk = chunk
        self.microphone_index = microphone_index


    def record_microphone(self, chunk=1024):
        p = pyaudio.PyAudio()

        stream = p.open(format=self.audio_format,
                        channels=self.channels,
                        rate=self.frame_rate,
                        input=True,
                        input_device_index=self.microphone_index,
                        frames_per_buffer=self.chunk)

        frames = []

        while not self.messages.empty():
            data = stream.read(chunk)
            frames.append(data)
            if len(frames) >= (self.frame_rate * self.record_seconds) / self.chunk:
                self.recordings.put(frames.copy())
                frames = []

        stream.stop_stream()
        stream.close()
        p.terminate()

    def speech_recognition(self):

        while not self.messages.empty():
            frames = self.recordings.get()

            self.rec.AcceptWaveform(b''.join(frames))
            result = self.rec.Result()
            text = json.loads(result)["text"]


            print(text)
            predict(text)
            time.sleep(1)

    def run(self):
        print("Recording started")
        record = Thread(target=self.record_microphone)
        record.start()
        self.messages.put(True)
        transcribe = Thread(target=self.speech_recognition)
        transcribe.start()

    def stop(self):
        self.messages.get()

    



transcriber = Transcriber(channels=1,
                          frame_rate=48000,
                          record_seconds=5,
                          audio_format=pyaudio.paInt16,
                          sample_size=3,
                          chunk=1024,
                          microphone_index=0)

transcriber.run()
def padd(arr):
    for i in range(20-len(arr)):
        arr.append('<pad>')
    return arr[:20]

def predict(text):

    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

    #stop words
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    text = ' '.join(filtered_sentence)
        
    feel_arr = word_tokenize(text)
      
    
   
# call the padd function for each sentence in feel_arr
    feel_arr=padd(feel_arr)

    
    embedded_feel_arr = []
    for word in feel_arr:
        if word.lower() in embeddings_index:
            embedded_feel_arr.append(embeddings_index[word.lower()])
        else:
            embedded_feel_arr.append([0]*50)

    X=np.array(embedded_feel_arr)
  
    X=np.reshape(X,(1,20,50))

    
    
    y=model.predict(X)
    y=np.argmax(y[0])
    labels=['anger', 'fear', 'happy', 'sadness', 'surprise']
    print(labels[y])

predict("i am scared")