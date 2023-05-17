# deep-learning-final-project
Real-Time Emotion Recognition: Improving Social Interaction for Individuals on the Autism Spectrum

Problem Statement:
One of the many everyday hardships faced by
individuals afflicted with ASL or other
neuro-divergent disorders is the inability to pick
up on social cues or trouble engaging in regular
social interactions. This stems from their having
difficulty recognizing emotions in face and tone.
Our problem statement is to develop a model that
would gauge the other person’s emotion in
real-time using a three-step classification process
analyzing facial expressions, tone of speech, and
language used in the interaction.

Model Description:
We use a CNN with a Bi-directional
LSTM for real-time facial expression recognition
along with speech emotion classification.



Steps to train the text emotion recognition model:
1. download the glove file given in the link below
   The link to the Golve file: https://drive.google.com/file/d/1yaG1D-kYMEDzmz0HnK6Wyt6nqWicMPJu/view?usp=sharing
2. Upload the dataset(emotion_final.csv) and the glove file to your drive and run each cell in the file.
3. To train the model open the TextEmotion.ipynb file run each cell.


Steps to execute the existing text emotion recoognition model:
1. download the glove file given in the link below and place in the same folder.
   The link to the Golve file: https://drive.google.com/file/d/1yaG1D-kYMEDzmz0HnK6Wyt6nqWicMPJu/view?usp=sharing
2.  run the audio2text.py file. It will take a few seconds to initialize. Once it is initialized it will record the speecha nd convert to text and also predict the emotion of the text       generated.
