import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
from pydub import AudioSegment
from alive_progress import alive_bar
from scipy.io.wavfile import write
import re
import os
from mutagen.id3 import ID3, APIC

class _file:
    rate = 0
    _type = ""
    filename = ""

def open_audio(file):
    _file.rate = 0
    _file._type = ""
    _file.filename = ""
    if matches := re.search(r"(.+)\.(.+)$", file):
        _file.filename = matches.group(1)
        try:
            if matches.group(2) == "wav":
                audio = AudioSegment.from_wav(file)
                _file.rate = audio.frame_rate
                _file._type = matches.group(2)
            elif matches.group(2) == "mp3":
                audio = AudioSegment.from_mp3(file)
                _file.rate = audio.frame_rate
                _file._type = matches.group(2)
            else:
                print(f"{_file.filename} not mp3 or wav")
                return 0
        except FileNotFoundError:
            print("File not found")
            return 0
    else:
        print("Unsupported filetype")
        return 0
    
    # return audio signal as 2D numpy array
    return np.array(audio.get_array_of_samples(), dtype=np.int16).reshape(-1, 2)
    

def extract_audio(audio_arr):
    
    # Replace all -1s and 1s with zero to avoid backround noise
    audio_arr[audio_arr == 1] = 0
    audio_arr[audio_arr == -1] = 0
    
    # Split original signal
    audio_left_org, audio_right_org = np.split(audio_arr, 2, axis=1)

    left_positive = np.array(audio_left_org, dtype=np.int32)
    left_negative = np.array(audio_left_org, dtype=np.int32)
    right_positive = np.array(audio_right_org, dtype=np.int32)
    right_negative = np.array(audio_right_org, dtype=np.int32)

    left_positive[left_positive < 0] = 1
    left_negative[left_negative >= 0] = -1
    right_positive[right_positive < 0] = 1
    right_negative[right_negative >= 0] = -1

    splited_signal = [
        left_positive,
        abs(left_negative),
        right_positive,
        abs(right_negative),
    ]
    splited_signal_copy = [
        np.array(left_positive, dtype=np.int32),
        np.array(abs(left_negative), dtype=np.int32),
        np.array(right_positive, dtype=np.int32),
        np.array(abs(right_negative), dtype=np.int32),
    ]
    
    return splited_signal, splited_signal_copy


def find_peak_limit(audio):
    duration = (len(audio) / _file.rate / 60)
    if duration < 0.03:
        return 1
    number_of_peaks = (6 / 5) * duration
    peak_limit = number_of_peaks * 1.6 * (60 * duration)

    return peak_limit


def find_peaks(splited_signal, peak_limit):
    peaks = []
    max_volume = 32767
    with alive_bar(len(splited_signal), title="Searching Peaks") as bar:
        for i, signal in enumerate(splited_signal):
            np.extract(signal > max_volume, signal)
            while True:
                found_peaks = np.extract(signal >= max_volume, signal).size
                max_volume -= 100
                if found_peaks > peak_limit:
                    peaks.append(np.where(signal >= max_volume)[0])
                    max_volume = 32767
                    break
            bar()
    return peaks


def find_transients(splited_signal, peaks):
    transients = [[], [], [], []]
    def find_transient(x, i, splited_signal):
        _i = 0
        while True:
            if splited_signal[x][peaks[x][i] - _i] <= 1:
                start = peaks[x][i] - _i
                _i = 0
                break
            _i += 1
        while True:
            if splited_signal[x][peaks[x][i] + _i] <= 1:
                end = peaks[x][i] + _i
                break
            _i += 1
        transients[x].append([start, end])

    for x in range(4):
        for i in range(len(peaks[x])):
            if i > 1:
                if (
                    transients[x][len(transients[x]) - 1][0]
                    <= peaks[x][i]
                    <= transients[x][len(transients[x]) - 1][1]
                ):
                    pass
                else:
                    find_transient(x, i, splited_signal)
            else:
                find_transient(x, i, splited_signal)

        if transients[x][0][1] >= transients[x][1][0]:
            del transients[x][0]
            
    return transients


def find_global_peak(splited_signal_copy, transients, x):
    for i in range(len(transients[x])):
        splited_signal_copy[x][transients[x][i][0] : transients[x][i][1]] = 0
    return 32767 / splited_signal_copy[x].max()


def normalise(splited_signal, splited_signal_copy, transients):
    def _normalise(splited_signal, start, end, factor):
        splited_signal[start:end][splited_signal[start:end] > 2] = (
            splited_signal[start:end][splited_signal[start:end] > 2] * factor
        )

    for x in range(4):
        global_peak = find_global_peak(splited_signal_copy, transients, x)
        first_tran = transients[x][0][0]
        _normalise(splited_signal[x], 0, first_tran, global_peak)
        with alive_bar(len(transients[x]), title="Normalising") as bar:
            for i in range(len(transients[x])):
                start_tran = transients[x][i][0]
                end_tran = transients[x][i][1]
                last_tran = transients[x][len(transients[x]) - 1][1]
                end_audio = len(splited_signal[x])
                try:
                    end_between = transients[x][i + 1][0]
                except IndexError:
                    pass
                peak_area_values = splited_signal[x][start_tran:end_tran]
                normalised_peak_area = peak_area_values * global_peak

                if normalised_peak_area.max() > 32767:
                    local_peak = 32767 / peak_area_values.max()
                    _normalise(splited_signal[x], start_tran, end_tran, local_peak)
                else:
                    _normalise(splited_signal[x], start_tran, end_tran, global_peak)
                _normalise(splited_signal[x], end_tran, end_between, global_peak)
                bar()
        # Insert normalised block from last peak until end
        _normalise(splited_signal[x], last_tran, end_audio, global_peak)


def unsplit_signal(splited_signal):
    for i in range(4):
        splited_signal[i][splited_signal[i] > 32767] = 32767

    splited_signal[1] = np.negative(splited_signal[1])
    splited_signal[3] = np.negative(splited_signal[3])

    audio_left = (
        np.array([splited_signal[0], splited_signal[1]], dtype=np.int16)
        .transpose()
        .flatten()
    )
    audio_right = (
        np.array([splited_signal[2], splited_signal[3]], dtype=np.int16)
        .transpose()
        .flatten()
    )

    audio_left = np.delete(audio_left, np.where(audio_left == -1))
    audio_right = np.delete(audio_right, np.where(audio_right == -1))
    audio_left = np.delete(audio_left, np.where(audio_left == 1))
    audio_right = np.delete(audio_right, np.where(audio_right == 1))

    audio_left = np.array(audio_left, dtype=np.int16)
    audio_right = np.array(audio_right, dtype=np.int16)

    return np.column_stack((audio_left, audio_right))


def write_tags(file):
    tags_old = ID3(file)
    tags_new = ID3(f"./Normalised Files/{file}")
    pict = tags_old.getall('APIC')[0].data
    #title
    tags_new["TIT2"] = tags_old["TIT2"]
    #BPM
    tags_new["TBPM"] = tags_old["TBPM"]
    #Key
    tags_new["TKEY"] = tags_old["TKEY"]
    #Artist
    tags_new["TPE1"] = tags_old["TPE1"]
    #Label           
    tags_new["TPUB"] = tags_old["TPUB"]
    #Encoder settings
    tags_new["TSSE"] = tags_old["TSSE"]
    #track_number
    tags_new["TRCK"] = tags_old["TRCK"]
    #Genre
    tags_new["TCON"] = tags_old["TCON"]
    #Album art
    tags_new.add(APIC(encoding=3, mime='image/jpg', type=3, desc=u'Cover', data=pict))
    tags_new.save(f"./Normalised Files/{file}", v2_version=3)


def save(splited_signal):
    if _file._type == "mp3":
        song = AudioSegment(unsplit_signal(splited_signal).tobytes(), frame_rate=_file.rate, sample_width=2, channels=2)
        song.export(f"./Normalised Files/{_file.filename}.{_file._type}", format="mp3", bitrate="320k")
    elif _file._type == "wav":
        write(f"./Normalised Files/{_file.filename}.{_file._type}", _file.rate, unsplit_signal(splited_signal))


def normalise_audio(file):
    audio_arr = open_audio(file)
    if type(audio_arr) == np.ndarray:
        splited_signal, splited_signal_copy = extract_audio(audio_arr)
        peaks = find_peaks(splited_signal, find_peak_limit(splited_signal[0]))
        normalise(splited_signal, splited_signal_copy, find_transients(splited_signal, peaks))
        save(splited_signal)
        write_tags(f"{_file.filename}.{_file._type}")
        print("\033[92m SUCCESS \033[0m")
        

def main():
    try:
        os.makedirs("Normalised Files")
    except FileExistsError:
        pass
    for file in os.listdir("./"):
        normalise_audio(file)


main()
