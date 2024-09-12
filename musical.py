from loguru import logger
from scipy.io import wavfile
import crepe

def get_pitch(file):
    logger.info('Detecting Pitch')
    # Try with Librosa CQT chromagram to double check
    sr, audio = wavfile.read(file)
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
    return time, frequency


# Todo : rework
def get_pitch_torch(file):
    """
    #Using the default implem from the repo
    hop_length = int(constants.SAMPLERATE / 200.)

    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # This would be a reasonable range for speech
    fmin = 50
    fmax = 300

    # Select a model capacity--one of "tiny" or "full"
    model = 'full'

    # Choose a device to use for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pick a batch size that doesn't cause memory errors on your gpu
    batch_size = 2048
    #audio_torch_tensor = torch.Tensor([audio_buffer])

    audio_tensor, sr = torchaudio.load(str(file))

    print('shape', audio_tensor.shape)
    print('quotient ', 1 + (111410 // hop_length))
    #Should be 499

    pitch, periodicity = torchcrepe.predict(audio_tensor,
                               constants.SAMPLERATE,
                               hop_length,
                               fmin,
                               fmax,
                               model,
                               batch_size=batch_size,
                               device=device,
                               pad=True,
                               return_periodicity=True)
    print(pitch)
    print(pitch.shape)

    figure, axis = plt.subplots(1, 1)
    sample_rate = 22050
    hopsize = 256
    seconds = pitch.shape[-1] * hopsize / sample_rate
    upsampled_pitch = 2 ** torch.nn.functional.interpolate(
        torch.log2(pitch)[None],
        scale_factor=10,
        mode='linear')[0]
    upsampled_periodicity = torch.nn.functional.interpolate(
        periodicity[None],
        scale_factor=10,
        mode='linear')[0]
    axis.scatter(
        torch.linspace(0., seconds, upsampled_pitch.shape[-1]),
        upsampled_pitch.squeeze(),
        c=matplotlib.cm.RdYlGn(upsampled_periodicity[0] / upsampled_periodicity.max()), edgecolor='none')
    #figure.show()
    #plt.show()

    #total_frames = 1 + int((audio.size(1) - WINDOW_SIZE) // hop_length)

    time = np.arange(pitch.shape[1]) * (hop_length)
    print(time)
    final_pitch = np.linspace(0, stop, num=50)
    #1 + int(time // hop_length)
    #Pitch = 499

    #111410
    """
