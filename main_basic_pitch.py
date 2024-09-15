from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model
from basic_pitch.inference import predict_and_save
from loguru import logger

import utils

if __name__ == "__main__":
    logger.info('Start Basic Pitch')
    basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)

    source_folder = 'Example Files'

    files = utils.get_files(source_folder)
    output_folder = 'output_basic_pitch'

    predict_and_save(audio_path_list=files, output_directory=output_folder, model_or_model_path=basic_pitch_model,
                     save_midi=True, sonify_midi=True, save_model_outputs=False, save_notes=False)
