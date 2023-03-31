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

    def split_array(array) -> list:
        split_array1, split_array2 = np.split(array, 2, axis=1)
        split_arrays = [
            np.array(split_array1, dtype=np.float32),
            np.array(split_array1, dtype=np.float32),
            np.array(split_array2, dtype=np.float32),
            np.array(split_array2, dtype=np.float32),
        ]
        return split_arrays

    def merge_split_arrays(split_arrays: list) -> list:
        signal_left = np.array(
            [split_arrays[0], split_arrays[1]], dtype=np.int16).T.flatten()
        signal_right = np.array(
            [split_arrays[2], split_arrays[3]], dtype=np.int16).T.flatten()
        return [signal_left, signal_right]


class Normalizer:

    # todo # make search threshold smarter
    def find_search_treshold(signal_array: np.ndarray, frame_rate: int) -> int:
        duration = len(signal_array) / frame_rate / 60
        if duration < 0.03:
            return 1
        number_of_transients = (6 / 15) * duration
        return int(number_of_transients * 1.6 * (60 * duration))

    def find_transients(signal_array: np.ndarray, search_treshold: int) -> np.ndarray:
        threshold = 16383
        step = 8192
        reached_above, reached_below = 0, 0

        while True:
            found_transients = np.extract(
                signal_array >= threshold, signal_array).size
            if found_transients > search_treshold:
                if reached_below:
                    step /= 2
                threshold += step
                reached_above, reached_below = 1, 0
            elif found_transients < search_treshold:
                if reached_above:
                    step /= 2
                threshold -= step
                reached_below, reached_above = 1, 0
            if step <= 1 or found_transients == search_treshold:
                if threshold < 5:
                    return np.array([])
                else:
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
                "signal_array": signal_array}
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

    def save(signal_array: np.ndarray, audio_file_data: dict) -> None:
        if audio_file_data["type"] == ".mp3":
            audio = AudioSegment(signal_array.tobytes(
            ), frame_rate=audio_file_data["frame_rate"], sample_width=2, channels=2,)
            audio.export(
                f"./Normalised Files/{audio_file_data['filename']}", format="mp3", bitrate="320k",)
        elif audio_file_data["type"] == ".wav":
            write(
                f"./Normalised Files/{audio_file_data['filename']}", audio_file_data["frame_rate"], signal_array)


def split_array_and_modify(array: np.ndarray) -> list:
    ArrayModifiers.replace_equals_with_values(
        array, EQUAL_VALUES, REPLACE_VALUES)
    arrays = ArrayModifiers.split_array(array)
    ArrayModifiers.replace_negatives_with_value(
        arrays, [SIGNAL_LEFT_POSITIVE, SIGNAL_RIGHT_POSITIVE], 1)
    ArrayModifiers.replace_positives_with_value(
        arrays, [SIGNAL_LEFT_NEGATIVE, SIGNAL_RIGHT_NEGATIVE], -1)
    ArrayModifiers.invert_values(
        arrays, [SIGNAL_LEFT_NEGATIVE, SIGNAL_RIGHT_NEGATIVE])
    return arrays


def normalize(signal_array: list, audio_file_data: dict) -> bool:
    search_treshold = Normalizer.find_search_treshold(
        signal_array, audio_file_data['frame_rate'])
    transients = Normalizer.find_transients(signal_array, search_treshold)
    if transients.any():
        ArrayModifiers.remove_adjacent_by_difference(transients)
        amplitudes = Normalizer.find_amplitudes(signal_array, transients)
        amplification_factor = Normalizer.find_amplification_factor(
            signal_array, amplitudes)
        Normalizer.amplify(signal_array, amplification_factor, amplitudes)
        Normalizer.check_for_cliping(signal_array)
        return 1
    else:
        return 0


def modify_and_merge_arrays(signal_arrays: list) -> np.ndarray:
    ArrayModifiers.invert_values(
        signal_arrays, [SIGNAL_LEFT_NEGATIVE, SIGNAL_RIGHT_NEGATIVE])
    signals_left_right = ArrayModifiers.merge_split_arrays(signal_arrays)
    ArrayModifiers.delete_values(signals_left_right, DELETE_VALUES)
    signal_array = ArrayModifiers.combine_arrays(signals_left_right)
    return signal_array


def run_multithreaded(function: callable, audio_file_data: dict, signal_arrays: list) -> bool:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(function, signal_array, audio_file_data)
                   for signal_array in signal_arrays]
        done, _ = concurrent.futures.wait(futures)
        results = [future.result() for future in done]
        return all(results)


def main():
    try:
        os.makedirs("Normalised Files")
    except FileExistsError:
        pass
    done_files = os.listdir("./Normalised Files")
    for file in os.listdir("./"):
        if file not in done_files:
            if audio_file_data := File.open_audio(file):
                print(file)
                signal_arrays = split_array_and_modify(
                    audio_file_data["signal_array"])
                if not run_multithreaded(normalize, audio_file_data, signal_arrays):
                    print("\033[96m SILENCE OR NO NORMALISATION NEEDED \033[0m")
                    continue
                signal_array = modify_and_merge_arrays(signal_arrays)
                File.save(signal_array, audio_file_data)
                File.write_tags(file)
                print("\033[92m NORMALISED \033[0m")


if __name__ == "__main__":
    main()
