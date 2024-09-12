# Audio to Midi (Bass)
Convert a wav file to MIDI

---

## How to use 

- Install deps `pip install -r requirements.txt`
- Run the `main.py` file to test with the demo audio bass file
- The MIDI file will be generated in the `folder`
- Place the files you want to try in the `Examples Files` folder

## Issues

These issues have been found with the cut audio bass file provided

### First note is wrong
The first note is a E2 (note 40, freq 82) instead of a f2 (41, 87).
It seems that due to the librosa backtracking, the notes are shifted by an amount of frames, thus, when looking at the pitch in the table we look at the wrong note (shifted)
Todo -> trim start silence to get the correct index

## Todo
- [ ] [Feature] Extract note level to get MIDI note duration
- [ ] [EVALUATION] Compare different backend (madmom vs librosa) and method to extract features
- [ ] [EVALUATION] Compute a score for onset detection and pitch detection
- [ ] [EVALUATION] Regenerate the audio file from the MIDI file to compare them (in progress)
- [ ] [Preprocessing] Remove stereo sides
- [ ] [MISC] Output the LabeledAudio class to csv
- [ ] [MISC] Display all features extracted in a plot
- [ ] [MISC] Finish to type variables and methods

