import concurrent.futures
from mutagen.id3 import ID3, APIC
import os
from scipy.io.wavfile import write
from alive_progress import alive_bar
from pydub import AudioSegment
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


SIGNAL_LEFT_POSITIVE = 0
SIGNAL_LEFT_NEGATIVE = 1
SIGNAL_RIGHT_POSITIVE = 2
SIGNAL_RIGHT_NEGATIVE = 3
EQUAL_VALUES = [1, -1]
REPLACE_VALUES = [0, 0]
DELETE_VALUES = [1, -1]


class ArrayModifiers:

    def remove_adjacent_by_difference(arrays: list, diff=1) -> None:
        for arr_index in range(len(arrays)):
            i = 0
            while i < arrays[arr_index].size - 1:
                if abs(arrays[arr_index][i] - arrays[arr_index][i+1]) == diff:
                    arrays[arr_index] = np.delete(arrays[arr_index], i)
                else:
                    i += 1

    def replace_equals_with_values(array, equals: list, values: list) -> None:
        for equals, values in zip(equals, values):
            array[array == equals] = values

    def replace_negatives_with_value(arrays: list, indices: list, value) -> None:
        for i in indices:
            arrays[i][arrays[i] < 0] = value

    def replace_positives_with_value(arrays: list, indices: list, value) -> None:
        for i in indices:
            arrays[i][arrays[i] >= 0] = value

    def invert_values(arrays: list, indices: list) -> None:
        for i in indices:
            arrays[i] = np.where(arrays[i] >= 0, np.negative(
                arrays[i]), np.abs(arrays[i]))

    def delete_values(arrays: list, values) -> None:
        for arr in range(len(arrays)):
            mask = np.in1d(arrays[arr], values)
            arrays[arr] = arrays[arr][~mask]

    def combine_arrays(array_pair: list) -> np.ndarray:
        return np.column_stack((array_pair[0], array_pair[1]))

    def split_signal_array(signal_array) -> list:
        signal_left_org, signal_right_org = np.split(signal_array, 2, axis=1)
        split_signals = [
            np.array(signal_left_org, dtype=np.float32),
            np.array(signal_left_org, dtype=np.float32),
            np.array(signal_right_org, dtype=np.float32),
            np.array(signal_right_org, dtype=np.float32),
        ]
        return split_signals

    def merge_split_signal(split_signal: list) -> list:
        signal_left = np.array(
            [split_signal[0], split_signal[1]], dtype=np.int16).T.flatten()
        signal_right = np.array(
            [split_signal[2], split_signal[3]], dtype=np.int16).T.flatten()
        return [signal_left, signal_right]


class Normalizer:

    def find_transient_limit(signal_array: np.ndarray, frame_rate: int) -> int:
        duration = len(signal_array) / frame_rate / 60
        if duration < 0.03:
            return 1
        number_of_transients = (6 / 10) * duration
        return int(number_of_transients * 1.6 * (60 * duration))

    def find_transients(signal_array: np.ndarray, transient_limit: int) -> np.ndarray:
        threshold = 16383
        step = 8192
        reached_above, reached_below = 0, 0

        while True:
            found_transients = np.extract(
                signal_array >= threshold, signal_array).size
            if found_transients > transient_limit:
                if reached_below:
                    step /= 2
                threshold += step
                reached_above, reached_below = 1, 0
            elif found_transients < transient_limit:
                if reached_above:
                    step /= 2
                threshold -= step
                reached_below, reached_above = 1, 0
            else:
                return np.where(signal_array >= threshold)[0]
            if step <= 1:
                return np.where(signal_array >= threshold)[0]

    def find_amplitudes(signal_array: np.ndarray, transients: np.ndarray) -> list:
        amplitudes = []
        for transient in transients:
            start, end = None, None
            for trans in range(transient, -1, -1):
                if signal_array[trans] <= 1:
                    start = trans
                    break
            for trans in range(transient, len(signal_array)):
                if signal_array[trans] <= 1:
                    end = trans
                    break
            if amplitudes:
                if amplitudes[-1][1] >= start:
                    continue
            amplitudes.append([start, end])
        return amplitudes

    def find_amplification_factor(signal_array: np.ndarray, amplitudes: list) -> list:
        masked_signal_array = signal_array.copy()

        for start, end in amplitudes:
            masked_signal_array[start: end] = 0

        amplification_factor = np.float32(
            32767 / np.max(masked_signal_array))
        if amplification_factor < 1:
            amplification_factor = 1.0
        return amplification_factor

    def amplify(signal_array: np.ndarray, amplification_factor: float, amplitudes: list) -> None:
        AUDIO_END = len(signal_array)
        FIRST_AMP = amplitudes[0][0]
        LAST_AMP = amplitudes[-1][1]

        def amplify_segment(start: int, end: int, factor: float) -> None:
            segment = signal_array[start:end]
            above_threshold = segment > 2
            segment[above_threshold] *= factor

        amplify_segment(0, FIRST_AMP, amplification_factor)

        for i, (start, end) in enumerate(amplitudes):
            segment = signal_array[start:end] * amplification_factor
            if segment.max() > 32767:
                factor = 32767 / segment.max()
            else:
                factor = amplification_factor
            amplify_segment(start, end, factor)
            if i + 1 < len(amplitudes):
                next_start = amplitudes[i + 1][0]
                amplify_segment(end, next_start, amplification_factor)

        amplify_segment(LAST_AMP, AUDIO_END, amplification_factor)

    def check_for_cliping(signal_array: np.ndarray) -> None:
        signal_array[signal_array > 32767] = 32767


class File:

    def open_audio(file) -> dict:
        name, ext = os.path.splitext(file)
        if ext.lower() not in {'.mp3', '.wav'}:
            return None
        try:
            if ext.lower() == '.mp3':
                audio = AudioSegment.from_mp3(file)
            elif ext.lower() == '.wav':
                audio = AudioSegment.from_wav(file)
            signal_array = np.array(
                audio.get_array_of_samples(), dtype=np.int16).reshape(-1, 2)
            file_data = {
                "filename": f"{name}{ext}",
                "type": ext,
                "frame_rate": audio.frame_rate,
                "signal_arr": signal_array}
            return file_data
        except (FileNotFoundError, PermissionError, IsADirectoryError):
            return None

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
        tags_old = ID3(file)
        tags_new = ID3(f"./Normalised Files/{file}")
        try:
            # Album art
            pict = tags_old.getall("APIC")[0].data
            tags_new.add(APIC(encoding=3, mime="image/jpg",
                         type=3, desc="Cover", data=pict))
        except:
            print("Album Cover not found")

        for tag in tags:
            try:
                tags_new[tags[tag]] = tags_old[tags[tag]]
            except:
                print(f"{tag} tag not found")

        tags_new.save(f"./Normalised Files/{file}", v2_version=3)

    def save(signal_array, file_name: str, file_type: str, frame_rate: int) -> None:
        if file_type == ".mp3":
            audio = AudioSegment(signal_array.tobytes(
            ), frame_rate=frame_rate, sample_width=2, channels=2,)
            audio.export(
                f"./Normalised Files/{file_name}", format="mp3", bitrate="320k",)
        elif file_type == ".wav":
            write(f"./Normalised Files/{file_name}", frame_rate, signal_array)


def open_file_and_modulate_signal_arrays(file: str) -> list:
    if audio_file_data := File.open_audio(file):
        ArrayModifiers.replace_equals_with_values(
            audio_file_data['signal_arr'], EQUAL_VALUES, REPLACE_VALUES)
        split_signal_arrays = ArrayModifiers.split_signal_array(
            audio_file_data['signal_arr'])
        ArrayModifiers.replace_negatives_with_value(
            split_signal_arrays, [SIGNAL_LEFT_POSITIVE, SIGNAL_RIGHT_POSITIVE], 1)
        ArrayModifiers.replace_positives_with_value(
            split_signal_arrays, [SIGNAL_LEFT_NEGATIVE, SIGNAL_RIGHT_NEGATIVE], -1)
        ArrayModifiers.invert_values(
            split_signal_arrays, [SIGNAL_LEFT_NEGATIVE, SIGNAL_RIGHT_NEGATIVE])
        return split_signal_arrays, audio_file_data
    return None


def normalize(split_signal_arrays: list, audio_file_data: dict) -> list:
    for signal_array_index in range(len(split_signal_arrays)):
        signal_array = split_signal_arrays[signal_array_index]
        transient_limit = Normalizer.find_transient_limit(
            signal_array, audio_file_data['frame_rate'])
        transients = Normalizer.find_transients(signal_array, transient_limit)
        if type(transients) == np.ndarray:
            ArrayModifiers.remove_adjacent_by_difference(transients)
            amplitudes = Normalizer.find_amplitudes(signal_array, transients)
            amplification_factor = Normalizer.find_amplification_factor(
                signal_array, amplitudes)
            Normalizer.amplify(signal_array, amplification_factor, amplitudes)
            Normalizer.check_for_cliping(signal_array)
            return split_signal_arrays, audio_file_data
        else:
            return None, None


def modulate_signal_arrays_and_save(split_signal_arrays: list, audio_file_data: dict) -> None:
    ArrayModifiers.invert_values(
        split_signal_arrays, [SIGNAL_LEFT_NEGATIVE, SIGNAL_RIGHT_NEGATIVE])
    signals_left_right = ArrayModifiers.merge_split_signal(split_signal_arrays)
    ArrayModifiers.delete_values(signals_left_right, DELETE_VALUES)
    audio_signal = ArrayModifiers.combine_arrays(signals_left_right)
    File.save(audio_signal, audio_file_data['filename'],
              audio_file_data['type'], audio_file_data['frame_rate'])
    File.write_tags(audio_file_data['filename'])
    print("\033[92m SUCCESS \033[0m")


def main():
    try:
        os.makedirs("Normalised Files")
    except FileExistsError:
        pass
    done_files = os.listdir("./Normalised Files")
    for file in os.listdir("./"):
        if file not in done_files:
            ...

if __name__ == "__main__":
    main()
