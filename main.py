import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
from pydub import AudioSegment
from alive_progress import alive_bar
from scipy.io.wavfile import write
import os
from mutagen.id3 import ID3, APIC


def remove_adjacent_by_difference(arrays: list, diff=1) -> None:
    for arr_index in range(len(arrays)):
        i = 0
        while i < arrays[arr_index].size - 1:
            if abs(arrays[arr_index][i] - arrays[arr_index][i+1]) == diff:
                arrays[arr_index] = np.delete(arrays[arr_index], i)
            else:
                i += 1


def replace_equal(array, equal: list, value: list) -> None:
    for equal, value in zip(equal, value):
        array[array == equal] = value


def replace_negative(arrays: list, indices: list, value) -> None:
    for i in indices:
        arrays[i][arrays[i] < 0] = value
        

def replace_positive(arrays: list, indices: list, value) -> None:
    for i in indices:
        arrays[i][arrays[i] >= 0] = value
        

def invert_values(arrays: list, indices: list) -> None:
    for i in indices:
        arrays[i] = np.where(arrays[i] >= 0, np.negative(arrays[i]), np.abs(arrays[i]))


def delete_values(arrays: list, values) -> None:
    for arr in range(len(arrays)):
        mask = np.in1d(arrays[arr], values)
        arrays[arr] = arrays[arr][~mask]
        
        
def combine_arrays(data_pair: list):
    return np.column_stack((data_pair[0], data_pair[1]))


def open_audio(file) -> dict:
    name, ext = os.path.splitext(file)
    if ext.lower() not in {'.mp3', '.wav'}:
        return None
    try:
        if ext.lower() == '.mp3':
            audio = AudioSegment.from_mp3(file)
        elif ext.lower() == '.wav':
            audio = AudioSegment.from_wav(file)
        signal_arr = np.array(audio.get_array_of_samples(), dtype=np.int16).reshape(-1, 2)
        return {"filename": f"{name}{ext}", "type": ext, "frame_rate": audio.frame_rate, "signal_arr": signal_arr}
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return None
    
    
def split_signal_array(signal_arr) -> list:
    signal_left_org, signal_right_org = np.split(signal_arr, 2, axis=1)
    split_signals = [
        np.array(signal_left_org, dtype=np.float32),
        np.array(signal_left_org, dtype=np.float32),
        np.array(signal_right_org, dtype=np.float32),
        np.array(signal_right_org, dtype=np.float32),
    ]
    
    return split_signals


def find_transient_limit(signal_arr, frame_rate: int) -> int:
    duration = len(signal_arr) / frame_rate / 60
    if duration < 0.03:
        return 1
    number_of_transients = (6 / 10) * duration
    
    return int(number_of_transients * 1.6 * (60 * duration))


def find_transients(split_signals, transient_limit) -> list:
    transients_per_split_signal = []
    
    for split_signal in split_signals:
        threshold = 16383
        step = 8192
        reached_above, reached_below = 0, 0
        
        while step > 0:
            found_transients = np.extract(split_signal >= threshold, split_signal).size
            if found_transients > transient_limit:
                if reached_below:
                    step /= 2
                threshold += step
                reached_above, reached_below = 1, 0
            if found_transients < transient_limit:
                if reached_above:
                    step /= 2
                threshold -= step
                reached_below, reached_above = 1, 0
            if step <= 2:
                transients_per_split_signal.append(np.where(split_signal >= threshold)[0])
                break
            
    return transients_per_split_signal


def find_amplitudes(split_signals, transients_per_split_signal):
    amplitudes_per_split_signal = [[], [], [], []]
     
    for split_signal_index in range(len(split_signals)):
        split_signal = split_signals[split_signal_index]
        transients = transients_per_split_signal[split_signal_index]
        
        for transient in transients:
            start, end = None, None
            for trans in range(transient, -1, -1):
                if split_signal[trans] <= 1:
                    start = trans
                    break
            for trans in range(transient, len(split_signal)):
                if split_signal[trans] <= 1:
                    end = trans
                    break
            if amplitudes_per_split_signal[split_signal_index]:
                if amplitudes_per_split_signal[split_signal_index][-1][1] >= start:
                    continue
            amplitudes_per_split_signal[split_signal_index].append([start, end])

    return amplitudes_per_split_signal


def find_global_amplification_factors(split_signals: list, amplitudes: list) -> list:
    global_amplification_factors = []
    
    for split_signal_index in range(len(split_signals)):
        masked_split_signal = split_signals[split_signal_index].copy()
        for start, end in amplitudes[split_signal_index]:
            masked_split_signal[start : end] = 0
        global_amplification_factors.append(np.float32(32767 / np.max(masked_split_signal)))
        
    return global_amplification_factors


def amplify(split_signals: list, global_amplification_factors: list, amplitudes_per_split_signal: list) -> None:
    START = 0
    END = 1
    
    def _amplify(split_signal, start, end, factor):
        split_signal[start:end][split_signal[start:end] > 2] *= factor
        
    for split_signal_index in range(len(split_signals)):
        split_signal = split_signals[split_signal_index]
        global_amplification_factor = global_amplification_factors[split_signal_index]
        amplitudes = amplitudes_per_split_signal[split_signal_index]
        FIRST_AMP = amplitudes[split_signal_index][START]
        
        _amplify(split_signal, START, FIRST_AMP, global_amplification_factor)
        
        for amplitude in range(len(amplitudes)):
            START_AMP = amplitudes[amplitude][START]
            END_AMP = amplitudes[amplitude][END]
            LAST_AMP = amplitudes[len(amplitudes) - 1][END]
            END_OF_AUDIO = len(split_signal)
            try:
                START_NEXT_AMP = amplitudes[amplitude + 1][START]
            except IndexError:
                pass
            amplitude_area_values = split_signal[START_AMP:END_AMP]
            normalised_amplitude_area = split_signal[START_AMP:END_AMP] * global_amplification_factor
            
            if normalised_amplitude_area.max() > 32767:
                amplitude_amplification_factor = 32767 / amplitude_area_values.max()
                _amplify(split_signal, START_AMP, END_AMP, amplitude_amplification_factor)
            else:
                _amplify(split_signal, START_AMP, END_AMP, global_amplification_factor)
                
            _amplify(split_signal, END_AMP, START_NEXT_AMP, global_amplification_factor)

        _amplify(split_signal, LAST_AMP, END_OF_AUDIO, global_amplification_factor)


def check_for_cliping(split_signals: list) -> None:
    
    for split_signal_index in range(len(split_signals)):
        split_signals[split_signal_index][split_signals[split_signal_index] > 32767] = 32767
        
        
def merge_split_signal(split_signal: list) -> list:

    signal_left = np.array([split_signal[0], split_signal[1]], dtype=np.int16).T.flatten()
    signal_right = np.array([split_signal[2], split_signal[3]], dtype=np.int16).T.flatten()

    return [signal_left, signal_right]


def write_tags(file: str) -> None:
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
    print(file)
    tags_old = ID3(file)
    tags_new = ID3(f"./Normalised Files/{file}")
    try:
        # Album art
        pict = tags_old.getall("APIC")[0].data
        tags_new.add(APIC(encoding=3, mime="image/jpg", type=3, desc="Cover", data=pict))
    except:
        print("Album Cover not found")
    for tag in tags:
        try:
            tags_new[tags[tag]] = tags_old[tags[tag]]
        except:
            print(f"{tag} tag not found")
    tags_new.save(f"./Normalised Files/{file}", v2_version=3)
    
    
def save(audio_signal, file_name: str, file_type: str, frame_rate: int) -> None:
    if file_type == ".mp3":
        song = AudioSegment(audio_signal.tobytes(), frame_rate=frame_rate, sample_width=2, channels=2,)
        song.export(f"./Normalised Files/{file_name}", format="mp3", bitrate="320k",)
    elif file_type == ".wav":
        write(f"./Normalised Files/{file_name}", frame_rate, audio_signal)
        
        
def normalize(file: str) -> None:
    SIGNAL_LEFT_POSITIVE = 0
    SIGNAL_LEFT_NEGATIVE = 1
    SIGNAL_RIGHT_POSITIVE = 2
    SIGNAL_RIGHT_NEGATIVE = 3
    if audio_file := open_audio(file):
        replace_equal(audio_file['signal_arr'], [1, -1], [0, 0])
        split_signal_arrays = split_signal_array(audio_file['signal_arr'])
        replace_negative(split_signal_arrays, [SIGNAL_LEFT_POSITIVE, SIGNAL_RIGHT_POSITIVE], 1)
        replace_positive(split_signal_arrays, [SIGNAL_LEFT_NEGATIVE, SIGNAL_RIGHT_NEGATIVE], -1)
        invert_values(split_signal_arrays, [SIGNAL_LEFT_NEGATIVE, SIGNAL_RIGHT_NEGATIVE])
        transient_limit = find_transient_limit(audio_file['signal_arr'], audio_file['frame_rate'])
        transients_per_split_signal = find_transients(split_signal_arrays, transient_limit)
        remove_adjacent_by_difference(transients_per_split_signal)
        amplitudes_per_split_signal = find_amplitudes(split_signal_arrays, transients_per_split_signal)
        global_amplification_factors = find_global_amplification_factors(split_signal_arrays, amplitudes_per_split_signal)
        amplify(split_signal_arrays, global_amplification_factors, amplitudes_per_split_signal)
        check_for_cliping(split_signal_arrays)
        invert_values(split_signal_arrays, [SIGNAL_LEFT_NEGATIVE, SIGNAL_RIGHT_NEGATIVE])
        signals_left_right = merge_split_signal(split_signal_arrays)
        delete_values(signals_left_right, [1, -1])
        audio_signal = combine_arrays(signals_left_right)
        save(audio_signal, audio_file['filename'], audio_file['type'], audio_file['frame_rate'])
        write_tags(audio_file['filename'])
        print("\033[92m SUCCESS \033[0m")
    else:
        pass
        
        
def main():
    try:
        os.makedirs("Normalised Files")
    except FileExistsError: pass
    done_files = os.listdir("./Normalised Files")
    for file in os.listdir("./"):
        if file not in done_files:
            normalize(file)


if __name__ == "__main__":
    main()
