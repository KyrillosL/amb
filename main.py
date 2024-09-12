import sys
from loguru import logger
import utils
from labeled_audio import LabeledAudio
from preprocess import resave_to_mono

if __name__ == "__main__":
    logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
    logger.info('Start')

    source_folder = 'Example Files'
    preprocessed_folder = 'preprocessed'

    # Preprocess -> Remove stereo sides, but don't filter right now (only for the pitch detection)
    files = utils.get_files(source_folder)
    for f in files:
        resave_to_mono(source_folder, preprocessed_folder, f.name)

    # Get the notes
    files = utils.get_files(preprocessed_folder)
    for f in files:
        logger.info(f)
        labeled_audio = LabeledAudio(f)
        logger.info(labeled_audio)
        labeled_audio.generate_midi_file()
