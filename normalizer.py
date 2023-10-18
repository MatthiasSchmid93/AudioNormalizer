"""
This is a normalizer for mp3 and wav files. Its a final project for Havard CS50P.

The approach in this program to normalize audio differs in comparison to most other programs that implement this task. 
Most normalizers amplify an audio signal until the highest transient reaches 0 dB.
The downside of this approach is that the audio signal can only be amplified based on a single maximum transient. 
In other words, the whole normalizing process depends on one transient, which is inefficient.

This program, however, doesn't depend on a single transient. 
It rather splits the original signal into four separate signals (left positive, left negative, right positive, and right negative). 
Then it uses an algorithm that finds several transients that exceed a calculated value based on the audio file's duration and actual volume. 
For each split signal, a copy is being created, a dummy signal. In the dummy signals, the found transients are turned into the value of zero. 
The next task is to find the new maximum transient in each modified dummy signal like any other simple normalizer. 
The new maximum value is then saved for the final amplification.

At this stage, the program finds all amplitudes in the original split signals where the previously found transients are sitting. 
The split audio signals are then amplified based on the new maximum transient. 
All amplitudes that exceed (clip) above 0 dB are lowered to 0 dB without manipulating the original signal except for the volume.
Finally, all the amplified split signals are merged together and are saved as the original file name and type. 
All the ID3 tags, including the album cover, are also maintained in the new normalized file.

As a result, the user gets an audiofile normalized to 0 dB without losing the dynamic range or the overall audio quality.
"""

from mutagen.id3 import ID3, APIC, TIT2, TBPM, TKEY, TPE1, TPUB, TSSE, TRCK, TCON , APIC
from mutagen.aiff import AIFF
import os
from pydub import AudioSegment
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ProgressHandler:
    """
    access to current state of the normalizing process
    """ 
    def __init__(self) -> None:
        self.bar = 0
        self.running = False
        self.terminate = False
        self.current_file = ""

    def reset(self) -> None:
        self.bar = 0
        self.running = False
        self.terminate = False
        self.current_file = ""
     

progress = ProgressHandler()


def update_bar(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        progress.bar += 1
        return result

    return wrapper

class ArrayModifiers:
    @staticmethod
    def array_to_numpy_array(array, data_type: np.dtype) -> np.ndarray:
        return np.array(array, dtype=data_type)
    
    @staticmethod
    def reshape_to_channels(signal_array: np.ndarray, channels: int) -> None:
        return signal_array.reshape(-1, channels)
    
    @staticmethod
    def remove_adjacent_by_difference(array: np.ndarray, diff=1) -> np.ndarray:
        diff_array = np.abs(np.diff(array)) == diff
        return np.concatenate(([array[0]], array[1:][~diff_array]))

    @staticmethod
    def replace_equals_with_values(
        array: np.ndarray, equals: list, values: list
    ) -> None:
        """
        Replace elements equal to specified values with new ones in an array.
        """
        for equal, value in zip(equals, values):
            array[array == equal] = value

    @staticmethod
    def replace_negatives_with_value(arrays: list, indices: list, value) -> None:
        for i in indices:
            arrays[i][arrays[i] < 0] = value

    @staticmethod
    def replace_positives_with_value(arrays: list, indices: list, value) -> None:
        for i in indices:
            arrays[i][arrays[i] >= 0] = value

    @staticmethod
    def invert_values(arrays: list, indices: list) -> None:
        """
        Invert the sign of values at specified indices in arrays.
        """
        for i in indices:
            arrays[i] = np.where(
                arrays[i] >= 0, np.negative(arrays[i]), np.abs(arrays[i])
            )

    @staticmethod
    def delete_values(arrays: list, values: list) -> list:
        new_arrays = []
        
        for arr in arrays:
            mask = np.in1d(arr, values)
            new_arrays.append(arr[~mask])
        
        return new_arrays

    @staticmethod
    def combine_arrays(array_pair: list) -> np.ndarray:
        """
        Combines two arrays into a single array with paired columns.
        """
        return np.column_stack((array_pair[0], array_pair[1]))

    @staticmethod
    def split_audio_array(array: np.ndarray, channels: int, data_type: np.dtype) -> list:
        """
        Splits a given audio array into its left and right channels.
        Returns 4 arrays if Stereo: left_negative, left_positive and vice versa
        Returns 2 arrays if Mono: left_negative, left_positive
        """

        def create_split_arrays(signal):
            return [
                np.array(signal, dtype=data_type),
                np.array(signal, dtype=data_type),
            ]

        if channels == 1:  # MONO
            split_arrays = create_split_arrays(array)
            
        elif channels == 2:  # STEREO
            signal_left, signal_right = np.hsplit(array, 2)
            split_arrays = create_split_arrays(signal_left) + create_split_arrays(
                signal_right
            )

        return split_arrays

    @staticmethod
    def merge_split_arrays(split_arrays: list, channels: int, data_type: np.dtype) -> list:
        """
        Merges previously split audio channels back into their original array structure.
        """
        signal_left = np.array(
            [split_arrays[0], split_arrays[1]], dtype=data_type
        ).T.flatten()
        
        if channels == 2: # STEREO
            signal_right = np.array(
                [split_arrays[2], split_arrays[3]], dtype=data_type
            ).T.flatten()
            
            return [signal_left, signal_right]
        
        return [signal_left] # NONO


class Normalizer:
    @staticmethod
    def get_data_type(sample_width: int) -> np.dtype:
        """
        Gets the data type for all the arrays based on the origin signal data type.
        Sample width 1 == 1 byte == 8 bit,...
        """
        data_types = {1: np.int8, 2: np.int16, 4: np.int32}
        return data_types[sample_width]
    
    @staticmethod
    def get_max_value(sample_width: int) -> int:
        """
        (int(((1 << bits) - 1) / 2)
        Maximum value that can be represented given the sample width
        """
        max_values = {1: 127, 2: 32767, 4: 2147483647}
        return max_values[sample_width]
        
    @staticmethod
    def find_transient_threshold(signal_array: np.ndarray, frame_rate: int) -> list:
        """
        Calculate a transient threshold based on the maximum values in equal-sized frames of a signal array.
        """
        blocks_max = []
        
        for i in range(0, signal_array.size, frame_rate):
            block = signal_array[i : i + frame_rate]
            blocks_max.append(int(block.max()))
            threshold = int(sum(blocks_max) / len(blocks_max))
        return threshold

    @staticmethod
    def find_transients(signal_array: np.ndarray, threshold: int) -> np.ndarray:
        """
        Find the indices where the signal array values exceed a given threshold.
        """
        return np.where(signal_array >= threshold)[0]

    @staticmethod
    def find_amplitudes(signal_array: np.ndarray, transients: np.ndarray) -> list:
        """
        Determine amplitudes within a signal array given specific transients.
        """
        indices_of_ones = np.where(signal_array == 1)[0]
        all_indices = np.searchsorted(indices_of_ones, transients)
        before_indices = all_indices - 1
        after_indices = all_indices
        
        try:
            amplitudes = np.column_stack(
                (indices_of_ones[before_indices], indices_of_ones[after_indices])
            )
        except IndexError:
            return None
        
        amplitudes = np.delete(amplitudes, 0, axis=0)
        return amplitudes.tolist()

    @staticmethod
    def find_amplification_factor(signal_array: np.ndarray, amplitudes: list, maximum_value: int) -> list:
        """
        Find a factor to amplify a signal array while considering designated amplitude regions.
        """
        masked_signal_array = signal_array.copy()

        for start, end in amplitudes:
            masked_signal_array[start:end] = 0

        amplification_factor = np.float32(
            maximum_value / np.max(masked_signal_array)
        )
        
        if amplification_factor < 1:
            amplification_factor = 1.0
            
        return amplification_factor

    @staticmethod
    @update_bar
    def amplify(
        signal_array: np.ndarray, amplification_factor: float, amplitudes: list, max_value: int
    ) -> None:
        """
        Amplify a signal array while considering designated amplitude regions and an amplification factor.
        """
        AUDIO_END = len(signal_array)
        FIRST_AMP = amplitudes[0][0]
        LAST_AMP = amplitudes[-1][1]

        def amplify_segment(start: int, end: int, factor: float) -> None:
            segment = signal_array[start:end].astype(float)
            above_threshold = segment > 3
            segment[above_threshold] *= factor
            segment = np.round(segment)
            signal_array[start:end] = segment

        amplify_segment(0, FIRST_AMP, amplification_factor)

        for i, (start, end) in enumerate(amplitudes):
            segment = signal_array[start:end] * amplification_factor
            
            if segment.max() > max_value:
                factor = max_value / (segment.max() / amplification_factor)
            else:
                factor = amplification_factor
                
            amplify_segment(start, end, factor)
            
            if i + 1 < len(amplitudes):
                next_start = amplitudes[i + 1][0]
                amplify_segment(end, next_start, amplification_factor)

        amplify_segment(LAST_AMP, AUDIO_END, amplification_factor)

    @staticmethod
    def check_for_clipping(signal_array: np.ndarray, max_value: int) -> None:
        """
        Adjusts the signal array values to ensure they do not exceed the maximum allowed volume.
        """
        signal_array[signal_array > max_value] = max_value
    
    @staticmethod
    @update_bar
    def prepare_signal(signal_array: np.ndarray, channels: int, data_type: np.dtype) -> list:
        """
        Prepare original signal array for the normalizing process.
        """
        EQUAL_VALUES = [1, -1]
        REPLACE_VALUES = [0, 0]

        ArrayModifiers.replace_equals_with_values(
            signal_array, EQUAL_VALUES, REPLACE_VALUES
        )
        signal_arrays = ArrayModifiers.split_audio_array(signal_array, channels, data_type)
        
        # 1s are placeholders to know where positive or negative values have been
        # It keeps all the samples in original order
        # The original length is also maintained
        # All arrays are processed unsighned. left negative for example contains only unsighned values. 
        # Its converted back to negative after normalizing.
        for i in range(0, len(signal_arrays), 2):
            ArrayModifiers.replace_negatives_with_value(signal_arrays, [i], 1)
            ArrayModifiers.replace_positives_with_value(signal_arrays, [i + 1], -1)
            ArrayModifiers.invert_values(signal_arrays, [i + 1])
            
        return signal_arrays

    @staticmethod
    @update_bar
    def undo_prepare_signal(signal_arrays: list, channels: int, data_type: np.dtype) -> np.ndarray:
        """
        Convert the prepared signal arrays back to the original structure
        """
        DELETE_VALUES = [1, -1]
        
        for i in range(0, len(signal_arrays), 2):
            ArrayModifiers.invert_values(signal_arrays, [i + 1])
            
        signals_left_right = ArrayModifiers.merge_split_arrays(signal_arrays, channels, data_type)
        signals_left_right = ArrayModifiers.delete_values(signals_left_right, DELETE_VALUES)
        
        if channels == 2:  # STEREO
            return ArrayModifiers.combine_arrays(signals_left_right)
        
        return signals_left_right[0].reshape(-1, 1) # MONO
    
    
class File:
    @staticmethod
    @update_bar
    def open_audio(file: str, folder: str) -> dict:
        name, ext = os.path.splitext(file)
        name = name.replace("â€“", "&")
        if ext.lower() not in {".mp3", ".wav"}:
            return None
        
        try:
            
            if ext.lower() == ".mp3":
                audio = AudioSegment.from_mp3(f"{folder}/{file}")
                
            elif ext.lower() == ".wav":
                audio = AudioSegment.from_wav(f"{folder}/{file}")
    
        except (FileNotFoundError, PermissionError, IsADirectoryError, ValueError):
            return None

        file_data = {
            "filename": name,
            "signal_array": audio.get_array_of_samples(),
            "channels": audio.channels,
            "sample_width": audio.sample_width,
            "frame_rate": audio.frame_rate,
        }
        
        return file_data

    @staticmethod
    @update_bar
    def write_tags(file: str, folder: str) -> None:
        file, ext = os.path.splitext(file)
        
        try:
            tags_old = ID3(f"{folder}/{file}{ext}")
        except:
            print("No tags found")
            return None
            
        file = file.replace("â€“", "&")
        tags = AIFF()
        tag_map = {
            "TIT2": TIT2,
            "TBPM": TBPM,
            "TKEY": TKEY,
            "TPE1": TPE1,
            "TPUB": TPUB,
            "TSSE": TSSE,
            "TRCK": TRCK,
            "TCON": TCON,
        }
        
        for tag, TagClass in tag_map.items():
            try:
                tag_text = str(tags_old[tag]).replace("â€“", "&")
                tags[tag] = TagClass(encoding=3, text=f"{tag_text}")
            except:
                print(f"{tag} not found")
                
        try:
            pict = tags_old.getall("APIC")[0].data
            tags["APIC"] = APIC(encoding=3, mime="image/jpg", type=3, desc="Cover", data=pict)
        except IndexError:
            print("Album Cover not found")
        
        tags.save(f"{folder}/Normalized Files/{file}.aiff", v2_version=3)


    @staticmethod
    @update_bar
    def save_as_aiff(signal_array: np.ndarray, file_data: dict, folder: str) -> None:
        new_audio = AudioSegment(
        signal_array.tobytes(),
        frame_rate=file_data["frame_rate"],
        sample_width=file_data["sample_width"],
        channels=file_data["channels"]
        )

        # Export the audio to an AIFF file
        new_audio.export(f"{folder}/Normalized Files/{file_data['filename']}.aiff", format="aiff")

    @staticmethod
    def count_availible_files(folder: str) -> int:
        count = 0
        
        for file in os.listdir(f"{folder}/"):
            _, ext = os.path.splitext(file)
            if ext.lower() in {".mp3", ".wav"}:
                count += 1
                
        return count


    @staticmethod
    def check_folder(folder: str) -> list[str]:
        files = []
        try:
            os.makedirs(f"{folder}/Normalized Files")
        except FileExistsError:
            pass
        
        for file in os.listdir(f"{folder}/Normalized Files"):
            file, _ = os.path.splitext(file)
            files.append(file)
            
        return files


@update_bar
def normalize_signal(signal_array: any, channels: int, frame_rate: int, sample_width: int) -> np.ndarray:
    
    data_type = Normalizer.get_data_type(sample_width)
    max_value = Normalizer.get_max_value(sample_width)
    
    signal_array = ArrayModifiers.array_to_numpy_array(signal_array, data_type)
    signal_array = ArrayModifiers.reshape_to_channels(signal_array, channels)
    signal_arrays = Normalizer.prepare_signal(signal_array, channels, data_type)
    
    for signal_array in signal_arrays:
        threshold = Normalizer.find_transient_threshold(
            signal_array, frame_rate
        )
        transients = Normalizer.find_transients(signal_array, threshold)
        
        # Removes transients that are directly next to each other. 
        # Its only important to know one transient per amplitude that exceeded the threshold
        transients = ArrayModifiers.remove_adjacent_by_difference(transients)
        
        if amplitudes := Normalizer.find_amplitudes(signal_array, transients):
            amplification_factor = Normalizer.find_amplification_factor(
                signal_array, amplitudes, max_value
            )
            
            # Nothing to amplify
            if amplification_factor == 1.0:
                continue
            
            try:
                Normalizer.amplify(signal_array, amplification_factor, amplitudes, max_value)
            except:
                continue
            
            Normalizer.check_for_clipping(signal_array, max_value)

    normalized_signal = Normalizer.undo_prepare_signal(signal_arrays, channels, data_type)
    return normalized_signal


def normalize_file(file: str, folder: str) -> None:
    if audio_file_data := File.open_audio(file, folder):
        progress.current_file = audio_file_data["filename"]
        normalized_signal = normalize_signal(
            audio_file_data["signal_array"],
            audio_file_data["channels"],
            audio_file_data["frame_rate"],
            audio_file_data["sample_width"]
        )
        
        File.save_as_aiff(normalized_signal, audio_file_data, folder)
        File.write_tags(file, folder)
        
        
def normalize_folder(folder) -> None:
    progress.running = True
    done_files = File.check_folder(folder)
    
    try:
        os.listdir(f"{folder}")
    except FileNotFoundError:
        progress.reset()
        return 1

    for file in os.listdir(f"{folder}"):
        file, ext = os.path.splitext(file)

        
        if progress.terminate:
            progress.reset()
            return None
        
        if file not in done_files:
            normalize_file(f"{file}{ext}", folder)
                
        progress.bar = 0
        
    progress.reset()
    return None


if __name__ == "__main__":
    normalize_folder("./")
