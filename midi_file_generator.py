from midiutil.MidiFile import MIDIFile
from loguru import logger

class MidiFileGenerator:
    def __init__(self, name, tempo, notes):
        print('Generating Midi File')
        self.mf = MIDIFile(1)  # only 1 track
        self.mf.addTrackName(track=0, time=0, trackName=name)
        self.mf.addTempo(track=0, time=0, tempo=tempo)

        for note in notes:
            q_time = note.start_second * (tempo / 60)
            q_duration = note.duration_second * (tempo / 60)

            self.mf.addNote(track=0, channel=0, pitch=note.midi_note_number, time=q_time, duration=q_duration, volume=note.volume)

    def write(self, path):
        logger.info('Writting midi file for ' + path)
        with open(path, 'wb') as outf:
            self.mf.writeFile(outf)
