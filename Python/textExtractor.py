import os
import pandas as pd
from pydub import AudioSegment
from deepgram import Deepgram
import asyncio
import json
from config import DEEPGRAM_API_KEY
import time
import mimetypes
import certifi

# print(certifi.where())  # This prints the path to the certifi CA bundle
# print('certificate location')
os.environ['SSL_CERT_FILE'] = certifi.where()

class text_features:
    
    def __init__(self, data):
        
        self.data = data
        
    async def transcribe_audio_with_deepgram(self, audio_path):

        # Transcribe using Deepgram
        try:
            dg_client = Deepgram(DEEPGRAM_API_KEY)
            
            # Read audio file for Deepgram
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                
            response = await dg_client.transcription.prerecorded(
                {'buffer': audio_data, 'mimetype': 'audio/wav'},  # Ensure it's in WAV format
                {'punctuate': True, 'utterances': True, 'language': 'en'}  # Additional options
            )
        
            return response
        
        except Exception as e:
            print(f"Error during transcription: {e}")
            
            return None

    def extract_transcription_data(self, transcription_result):
        
        # process transcriptions and extract word-level timestamps
        
        transcriptions = []
        if transcription_result and 'results' in transcription_result:
            utterances = transcription_result['results']['utterances']
            for utterance in utterances:
                words = utterance.get('words', [])
                for word in words:
                    word_text = word['word']
                    start_time = word['start']
                    end_time = word['end']
                    transcriptions.append([word_text, start_time, end_time])
        return transcriptions

    # Function to save transcriptions to CSV
    def save_transcriptions_to_csv(self, transcriptions, output_csv, filename):
        # Create a DataFrame from the list of transcriptions
        df = pd.DataFrame(transcriptions, columns=["Word", "Start Time", "End Time"])
        
        # Add filename to the CSV
        # df.insert(0, "Filename", filename)

        # Save the DataFrame to CSV
        df.to_csv(output_csv, index=False)
        print(f"Transcriptions saved to {output_csv}")

    # Main function to run the transcription for a single file
    async def transcribe_single_file(self, input_audio_file):
        # Perform transcription using Deepgram
        print(f"Processing {input_audio_file}...")

        transcription_result = await self.transcribe_audio_with_deepgram(input_audio_file)
        transcription_data = self.extract_transcription_data(transcription_result)
        
        transcription_data = pd.DataFrame(transcription_data, columns=["Word", "Start Time", "End Time"])
        
        # Save the results to CSV
        # self.save_transcriptions_to_csv(transcription_data, output_csv_file, os.path.basename(input_audio_file))
        return transcription_data


if __name__ == "__main__":
    
    start_time = time.time()
    print('working directory:')
    print(os.getcwd())

    
    dirpath = 'Data_super_May22'
    group = '20240522_1325_S3WBLM9W4J66'
    filename = '1716394969658-1a569948-8381-409b-9a04-fbb484b191a4-cam-audio-1716394970693'

    # Replace with your .webm file path and desired output CSV path
    input_file =  os.path.join(dirpath, group, filename) # Path to your input .webm file
    output_csv = os.path.join('Output', 'super_May22', 'Text', filename)  # Path to save the output CSV file
    
    # Run the transcription
    tf = text_features(input_file)
    asyncio.run(tf.transcribe_single_file(input_file, output_csv))
    
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"The code took {elapsed_time:.2f} minutes to run.")



