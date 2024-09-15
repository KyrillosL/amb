# Audio to Midi (Bass)
Convert a wav file to MIDI

### I used a 4 bar loop of the 'bass_2' to test the script. The best results are with this file

---

## How to use 

- Install deps `pip install -r requirements.txt`
- Run the `main.py` file to test with the demo audio bass file
MIDI files will be generated in the `midi` folder
- Place the files you want to try in the `Examples Files` folder

## Output 
- Generated MIDI Files (`midi` folder)
- Audio files split at onset (`cut_files` folder)
- Preprocessed to mono files (`preprocessed` folder)
- Preprocessed (filtered) files (`filtered` folder)

## Missing Features 
The level (note velocity) hasn't been implemented yet

## Issues

These issues have been found with the cut audio bass file provided

### Onset Detection
There is an offset in the onset detection, it should be improved, the backtracking isn't perfect. 
Todo -> Trim the silence / test other backend such as madmom (wip)

### Pitch Tracking
In addition to the inaccuracy of crepe, and due to the onset detection, we sometimes retrieve the wrong note pitch.
It seems that due to the librosa backtracking, the notes are shifted by an amount of frames, thus, when looking at the pitch in the table we look at the wrong note (shifted)
Todo -> trim start silence to get the correct index 

### Note duration
The note duration is wrong with some files (ex `bass_0`). The threshold to detect the silence should be adaptive

### MIDI File is trimmed 
The generated midi file is trimmed to the first note (must be from the used lib). For the Bass 1 it musts be realigned
 
### Legato
Legato notes are not supported yet (`bass_4`). I shouldn't rely only on onset detection to create note, but also on the pitch content variation

## Todo
- [X] [Feature] Implement Level (note velocity)
- [ ] [Feature] Adaptive threshold to extract note duration
- [ ] [Feature] Legato
- [ ] [Evaluation] Compare different backend (madmom vs librosa) and method to extract features
- [ ] [Evaluation] Compute a score for onset detection and pitch detection
- [ ] [Evaluation] Regenerate the audio file from the MIDI file to compare them (in progress)
- [ ] [Preprocessing] Remove stereo sides
- [ ] [MISC] Output the LabeledAudio class to csv
- [ ] [MISC] Display all features extracted in a plot
- [ ] [MISC] Finish to type variables and methods
- [ ] [MISC] Set back the samplerate to 22050 for the MIDI Extraction (keep 44100 for audio generation ?)

