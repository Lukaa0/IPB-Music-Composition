import glob
import pickle
from music21 import instrument, converter, note, chord
class PreProcessor(object):
    @staticmethod
    def generate_notes():
        notes = []
    
        for file in glob.glob("dataset/*.mid"):
            midi_file = converter.parse(file)
    
            print("Processing %s" % file)
    
            preprocess_notes = None
    
            try:
                s2 = instrument.partitionByInstrument(midi_file)
                preprocess_notes = s2.parts[0].recurse() 
            except:
                preprocess_notes = midi_file.flat.notes
    
            for element in preprocess_notes:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    
        with open('preprocessed_data/notes', 'wb') as filepath:
            pickle.dump(notes, filepath)
        print("Preprocessing complete.")
            
PreProcessor.generate_notes()

