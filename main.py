import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
from pydub import AudioSegment
from alive_progress import alive_bar
from scipy.io.wavfile import write
import re
import os
from mutagen.id3 import ID3, APIC


class File:
    rate = None
    type = None
    filename = None
    tags = {
        "Title": "TIT2",
        "BPM": "TBPM",
        "Key": "TKEY",
        "Artist": "TPE1",
        "Label": "TPUB",
        "Encoder settings": "TSSE",
        "Track number": "TRCK",
        "Genre": "TCON",
    }

    
def check_file(file):
    name, ext = os.path.splitext(file)
    if ext.lower() not in {'.mp3', '.wav'}:
        return False
    try:
        with open(file):
            pass
        File.filename = name
        File.type = ext
        return True
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return False


def open_audio(file):
    if File.type == ".mp3":
        audio = AudioSegment.from_mp3(file)
        File.rate = audio.frame_rate
    elif File.type == ".wav":
        audio = AudioSegment.from_wav(file)
        File.rate = audio.frame_rate
    return np.array(audio.get_array_of_samples(), dtype=np.int16).reshape(-1, 2)


def remove_backround_noise(signal_arr):
    signal_arr[signal_arr == 1] = 0
    signal_arr[signal_arr == -1] = 0


def split_signal_arr(signal_arr):

    def replace_negative(arr, value):
        arr[arr < 0] = value

    def replace_positive_equal(arr, value):
        arr[arr >= 0] = value

    def negative_to_positive_arr(arr):
        return abs(arr)

    signal_left_org, signal_right_org = np.split(signal_arr, 2, axis=1)

    # Replace positve ints with 1 and vice versa
    # In negative arr, 0 is also replaced for later merging
    # Turn negative ints to postive for simpliciy
    left_positive = np.array(signal_left_org, dtype=np.int32)
    left_negative = np.array(signal_left_org, dtype=np.int32)
    right_positive = np.array(signal_right_org, dtype=np.int32)
    right_negative = np.array(signal_right_org, dtype=np.int32)
    
    replace_negative(left_positive, 1)
    replace_positive_equal(left_negative, -1)
    replace_negative(right_positive, 1)
    replace_positive_equal(right_negative, -1)
    
    left_negative = negative_to_positive_arr(left_negative)
    right_negative = negative_to_positive_arr(right_negative)

    split_signal = [
        left_positive,
        left_negative,
        right_positive,
        right_negative,
    ]
    split_signal_copy = [
        np.array(left_positive, dtype=np.int32),
        np.array(left_negative, dtype=np.int32),
        np.array(right_positive, dtype=np.int32),
        np.array(right_negative, dtype=np.int32),
    ]

    return split_signal, split_signal_copy


def find_transient_limit(signal):
    duration = len(signal) / File.rate / 60
    if duration < 0.03:
        return 1
    number_of_transients = (6 / 8) * duration
    transient_limit = int(number_of_transients * 1.6 * (60 * duration))
    return transient_limit


def find_transients(split_signal, transient_limit):
    transients = []
    threshold = 16383
    threshold_old = 0
    step = 8192
    reached_above = 0
    reached_below = 0
    with alive_bar(len(split_signal), title="Searching transients") as bar:
        for signal in split_signal:
            while step > 0:
                found_transients = np.extract(signal >= threshold, signal).size
                if found_transients > transient_limit:
                    if reached_below:
                        step /= 2
                    threshold += step
                    reached_above = 1
                    reached_below = 0
                if found_transients < transient_limit:
                    if reached_above:
                        step /= 2
                    threshold -= step
                    reached_below = 1
                    reached_above = 0
                if int(threshold_old) == int(threshold):
                    transients.append(np.where(signal >= threshold)[0])
                    break
                threshold_old = threshold
            bar()
    return transients


def find_amplitudes(split_signal, transients):
    amplitudes = [[], [], [], []]

    def find_amplitude(split_signal, x, i, transients):
        start, end = None, None
        for j in range(transients[x][i], -1, -1):
            if split_signal[x][j] <= 1:
                start = j
                break
        for j in range(transients[x][i], len(split_signal[x])):
            if split_signal[x][j] <= 1:
                end = j
                break
        if start is not None and end is not None:
            amplitudes[x].append([start, end])

    for x in range(4):
        for i in range(len(transients[x])):
            if i > 1 and amplitudes[x]:
                prev_end = amplitudes[x][-1][1]
                if transients[x][i] <= prev_end:
                    continue
            find_amplitude(split_signal, x, i, transients)

        if len(amplitudes[x]) > 1 and amplitudes[x][0][1] >= amplitudes[x][1][0]:
            del amplitudes[x][0]
    return amplitudes


def amplify(split_signal, split_signal_copy, amplitudes):

    def find_global_peak(split_signal_copy, amplitudes, x):
        for i in range(len(amplitudes[x])):
            split_signal_copy[x][amplitudes[x][i][0] : amplitudes[x][i][1]] = 0
        return 32767 / split_signal_copy[x].max()

    def _amplify(split_signal, start, end, factor):
        split_signal[start:end][split_signal[start:end] > 2] = (
            split_signal[start:end][split_signal[start:end] > 2] * factor
        )
    with alive_bar(4, title="Normalizing") as bar:
        for x in range(4):
            global_peak = find_global_peak(split_signal_copy, amplitudes, x)
            first_amp = amplitudes[x][0][0]
            _amplify(split_signal[x], 0, first_amp, global_peak)
            for i in range(len(amplitudes[x])):
                start_amp = amplitudes[x][i][0]
                end_amp = amplitudes[x][i][1]
                last_amp = amplitudes[x][len(amplitudes[x]) - 1][1]
                end_audio = len(split_signal[x])
                try:
                    end_between = amplitudes[x][i + 1][0]
                except IndexError:
                    pass
                peak_area_values = split_signal[x][start_amp:end_amp]
                normalised_peak_area = split_signal[x][start_amp:end_amp] * global_peak
                if normalised_peak_area.max() > 32767:
                    local_peak = 32767 / peak_area_values.max()
                    _amplify(split_signal[x], start_amp, end_amp, local_peak)
                else:
                    _amplify(split_signal[x], start_amp, end_amp, global_peak)
                _amplify(split_signal[x], end_amp, end_between, global_peak)
            bar()
        # Insert normalised block from last peak until end
        _amplify(split_signal[x], last_amp, end_audio, global_peak)


def check_for_cliping(split_signal):
    for i in range(4):
        split_signal[i][split_signal[i] > 32767] = 32767


def merge_split_signal(split_signal):

    def delete_values(arr, values):
        mask = np.in1d(arr, values)
        return arr[~mask]

    def negative_arr(arr):
        return -arr

    split_signal[1] = negative_arr(split_signal[1])
    split_signal[3] = negative_arr(split_signal[3])

    signal_left = np.array([split_signal[0], split_signal[1]], dtype=np.int16).T.flatten()
    signal_right = np.array([split_signal[2], split_signal[3]], dtype=np.int16).T.flatten()

    signal_left = delete_values(signal_left, [1, -1])
    signal_right = delete_values(signal_right, [1, -1])

    return np.column_stack((signal_left, signal_right))


def write_tags(file):
    tags_old = ID3(file)
    tags_new = ID3(f"./Normalised Files/{file}")
    try:
        # Album art
        pict = tags_old.getall("APIC")[0].data
        tags_new.add(APIC(encoding=3, mime="image/jpg", type=3, desc="Cover", data=pict))
    except:
        print("Album Cover not found")
    for tag in File.tags:
        try:
            tags_new[File.tags[tag]] = tags_old[File.tags[tag]]
        except:
            print(f"{tag} tag not found")
    tags_new.save(f"./Normalised Files/{file}", v2_version=3)


def save(split_signal):
    if File.type == ".mp3":
        song = AudioSegment(split_signal.tobytes(), frame_rate=File.rate, sample_width=2, channels=2,)
        song.export(f"./Normalised Files/{File.filename}{File.type}", format="mp3", bitrate="320k",)
    elif File.type == ".wav":
        write(f"./Normalised Files/{File.filename}{File.type}", File.rate, split_signal)


def normalize(file):
    try:
        signal_arr = open_audio(file)
        remove_backround_noise(signal_arr)
        split_signal, split_signal_copy = split_signal_arr(signal_arr)
        transients = find_transients(split_signal, find_transient_limit(split_signal[0]))
        if len(transients[0]) == len(split_signal[0]):
            print("\033[91m SIGNAL TO LOW \033[0m")
            return
        amplify(split_signal, split_signal_copy, find_amplitudes(split_signal, transients))
        check_for_cliping(split_signal)
        signal_arr = merge_split_signal(split_signal)
        save(signal_arr)
        write_tags(f"{File.filename}{File.type}")
        print("\033[92m SUCCESS \033[0m")
    except: print("\033[91m FAILED \033[0m")


def main():
    try:
        os.makedirs("Normalised Files")
    except FileExistsError: pass
    done_files = os.listdir("./Normalised Files")
    for file in os.listdir("./"):
        if file not in done_files:
            if check_file(file) == True:
                print(file)
                normalize(file)


if __name__ == "__main__":
    main()
