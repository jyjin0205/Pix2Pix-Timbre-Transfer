import argparse
import os
import subprocess
import sys

import numpy as np
import pandas

import librosa
from config import DEFAULT_SAMPLING_RATE, NSYNTH_SAMPLE_RATE, NSYNTH_VELOCITIES
from data import files_within, init_directory
from lib.NoteSynthesizer import NoteSynthesizer
from scipy.io.wavfile import write as write_wav

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_path', required=True)
    ap.add_argument('--audios_path', required=True)
    ap.add_argument('--playback_speed', required=False, default=1)
    ap.add_argument('--duration_rate', required=False, default=4)
    ap.add_argument('--transpose', required=False, default=0)
    args = ap.parse_args()

    wavfiles = list(files_within(args.train_path, '*.wav'))
    init_directory(args.audios_path)

"""
    print()
    print("Instruments: \t", len(instruments), [instrument['name'] for instrument in instruments])
    print("MIDI files: \t", len(midifiles))
    print()
"""

    for wav in wavfiles:
        _, seq_name = os.path.split(wav)
        output_name = os.path.join(args.audios_path, os.path.splitext(seq_name)[0]+'.wav')

        print("Sequence: \t", wav)
        print("Output: \t", output_name, '\n')

        if(not os.path.isfile(output_name)):
            audio, _ = synth.render_sequence(
                                                sequence=str(wav),
                                                playback_speed=float(args.playback_speed),
                                                duration_scale=float(args.duration_rate),
                                            )

            if(DEFAULT_SAMPLING_RATE != NSYNTH_SAMPLE_RATE):
                audio = librosa.core.resample(audio, NSYNTH_SAMPLE_RATE, DEFAULT_SAMPLING_RATE)
            # write_audio(output_name, audio, DEFAULT_SAMPLING_RATE)
            write_wav(output_name, DEFAULT_SAMPLING_RATE, np.array(32000.*audio, np.short))