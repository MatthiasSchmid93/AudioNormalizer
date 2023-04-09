from mutagen.id3 import ID3, APIC
import os
from scipy.io.wavfile import write
from pydub import AudioSegment
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

EQUAL_VALUES = [1, -1]
REPLACE_VALUES = [0, 0]
DELETE_VALUES = [1, -1]
MAX_VOLUME = 32767


class ArrayModifiers:

    def remove_adjacent_by_difference(array: np.ndarray, diff=1) -> np.ndarray:
        diff_array = np.abs(np.diff(array)) == diff
        return np.concatenate(([array[0]], array[1:][~diff_array]))

    def replace_equals_with_values(array: np.ndarray, equals: list, values: list) -> None:
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

    def delete_values(arrays: list, values: list) -> None:
        for arr in range(len(arrays)):
            mask = np.in1d(arrays[arr], values)
            arrays[arr] = arrays[arr][~mask]

    def combine_arrays(array_pair: list) -> np.ndarray:
        return np.column_stack((array_pair[0], array_pair[1]))

    def split_audio_array(array: np.ndarray) -> list:
        def create_split_arrays(signal):
            return [np.array(signal, dtype=np.float32), np.array(signal, dtype=np.float32)]

        if array.ndim == 1:  # MONO
            split_arrays = create_split_arrays(array)
        elif array.ndim == 2:  # STEREO
            signal_left, signal_right = np.hsplit(array, 2)
            split_arrays = create_split_arrays(signal_left) + create_split_arrays(signal_right)

        return split_arrays

    def merge_split_arrays(split_arrays: list) -> list:
        signal_left = np.array(
            [split_arrays[0], split_arrays[1]], dtype=np.int16).T.flatten()
        signal_right = np.array(
            [split_arrays[2], split_arrays[3]], dtype=np.int16).T.flatten()
        return [signal_left, signal_right]
    

class Normalizer:

    def find_transient_threshold(signal_array: np.ndarray, frame_rate: int) -> list:
        blocks_max = []
        for i in range(0, signal_array.size, frame_rate):
            block = signal_array[i:i+frame_rate]
            blocks_max.append(int(block.max()))
            threshold = int(sum(blocks_max) / len(blocks_max))
        return threshold
    
    def find_transients(signal_array, threshold):
        return np.where(signal_array >= threshold)[0]

    def find_amplitudes(signal_array: np.ndarray, transients: np.ndarray) -> list:
        indices_of_ones = np.where(signal_array == 1)[0]
        all_indices = np.searchsorted(indices_of_ones, transients)
        before_indices = all_indices - 1
        after_indices = all_indices
        amplitudes = np.column_stack((indices_of_ones[before_indices], indices_of_ones[after_indices]))
        amplitudes = np.delete(amplitudes, 0, axis=0)
        return amplitudes.tolist()

    def find_amplification_factor(signal_array: np.ndarray, amplitudes: list) -> list:
        masked_signal_array = signal_array.copy()
        
        for start, end in amplitudes:
            masked_signal_array[start: end] = 0
            
        amplification_factor = np.float32(
            MAX_VOLUME / np.max(masked_signal_array))
        if amplification_factor < 1:
            amplification_factor = 1.0
        return amplification_factor

    def amplify(signal_array: np.ndarray, amplification_factor: float, amplitudes: list) -> None:
        AUDIO_END = len(signal_array)
        FIRST_AMP = amplitudes[0][0]
        LAST_AMP = amplitudes[-1][1]

        def amplify_segment(start: int, end: int, factor: float) -> None:
            segment = signal_array[start:end]
            above_threshold = segment > 3
            segment[above_threshold] *= factor

        amplify_segment(0, FIRST_AMP, amplification_factor)

        for i, (start, end) in enumerate(amplitudes):
            segment = signal_array[start:end] * amplification_factor
            if segment.max() > MAX_VOLUME:
                factor = MAX_VOLUME / (segment.max() / amplification_factor)
            else:
                factor = amplification_factor
            amplify_segment(start, end, factor)
            if i + 1 < len(amplitudes):
                next_start = amplitudes[i + 1][0]
                amplify_segment(end, next_start, amplification_factor)

        amplify_segment(LAST_AMP, AUDIO_END, amplification_factor)

    def check_for_cliping(signal_array: np.ndarray) -> None:
        signal_array[signal_array > MAX_VOLUME] = MAX_VOLUME


class File:

    def open_audio(file: str) -> dict:
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
        try:
            tags_old = ID3(file)
            tags_new = ID3(f"./Normalised Files/{file}")
        except:
            print("No ID3 Tags found")
        try:
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


def split_array_and_modify(signal_array: np.ndarray) -> list:
    ArrayModifiers.replace_equals_with_values(signal_array, EQUAL_VALUES, REPLACE_VALUES)
    signal_arrays = ArrayModifiers.split_audio_array(signal_array)
    for i in range(0, len(signal_arrays), 2):
        ArrayModifiers.replace_negatives_with_value(signal_arrays, [i], 1)
        ArrayModifiers.replace_positives_with_value(signal_arrays, [i + 1], -1)
        ArrayModifiers.invert_values(signal_arrays, [i + 1])
    return signal_arrays


def normalize(signal_array: list, audio_file_data: dict) -> bool:
    threshold = Normalizer.find_transient_threshold(signal_array, audio_file_data["frame_rate"])
    transients = Normalizer.find_transients(signal_array, threshold)
    transients = ArrayModifiers.remove_adjacent_by_difference(transients)
    amplitudes = Normalizer.find_amplitudes(signal_array, transients)
    amplification_factor = Normalizer.find_amplification_factor(
        signal_array, amplitudes)
    Normalizer.amplify(signal_array, amplification_factor, amplitudes)
    Normalizer.check_for_cliping(signal_array)
    return 1


def modify_and_merge_arrays(signal_arrays: list) -> np.ndarray:
    for i in range(0, len(signal_arrays), 2):
        ArrayModifiers.invert_values(signal_arrays, [i + 1])
    signals_left_right = ArrayModifiers.merge_split_arrays(signal_arrays)
    ArrayModifiers.delete_values(signals_left_right, DELETE_VALUES)
    if len(signal_arrays) == 4: # STEREO
        return ArrayModifiers.combine_arrays(signals_left_right)
    return signals_left_right


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
                for signal_array in signal_arrays:
                    if not normalize(signal_array, audio_file_data):
                        print("\033[96m SILENCE OR NO NORMALISATION NEEDED \033[0m")
                        continue
                signal_array = modify_and_merge_arrays(signal_arrays)
                File.save(signal_array, audio_file_data)
                File.write_tags(file)
                print("\033[92m NORMALISED \033[0m")


if __name__ == "__main__":
    main()
