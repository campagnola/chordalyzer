import pyqtgraph as pg
import fluidsynth
import numpy as np
from scipy.ndimage import gaussian_filter


def synth(notes, levels):
    fs = fluidsynth.Synth()
    sfid = fs.sfload("sf/JR_String2.sf2")
    fs.program_select(0, sfid, 0, 0)
    for note, level in zip(notes, levels):
        fs.noteon(0, note, level)
    samp = fs.get_samples(44100)[::2]
    return samp


def filter(sample):
    # take first half of power spectrum
    fft = abs(np.fft.rfft(sample))
    fft = fft[:len(fft)//2]

    # smooth and downsample fft
    ds = 5
    smooth = gaussian_filter(fft, ds)
    n_ds = (len(fft) // ds)
    return smooth[:n_ds*ds].reshape(n_ds, ds).mean(axis=1)


if __name__ == '__main__':
    n = 10000
    labels = np.zeros((n, 15), dtype='ubyte')
    feature_size = len(filter(synth([], [])))
    features = np.zeros((n, feature_size), dtype='float32')

    for i in range(n):
        note = np.random.randint(0, 13)
        chord_type = np.random.randint(0, 3)  # major, minor, 7
        labels[i, note] = 1
        labels[i, 12+chord_type] = 1

        octave = np.random.randint(2, 6)
        tonic = note + octave * 12
        if chord_type == 0:
            notes = [tonic, tonic + 4, tonic + 7]
        elif chord_type == 1:
            notes = [tonic, tonic + 3, tonic + 7]
        elif chord_type == 2:
            notes = [tonic, tonic + 4, tonic + 7, tonic + 11]

        levels = np.random.randint(10, 30, size=len(notes))
        features[i] = filter(synth(notes, levels))

        if (i % 1000) == 0:
            print("%d / %d" % (i, n))

    np.save('labels.npy', labels)
    np.save('features.npy', features)

