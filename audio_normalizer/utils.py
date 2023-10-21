from mutagen.id3 import ID3, APIC, TIT2, TBPM, TKEY, TPE1, TPUB, TSSE, TRCK, TCON , APIC
from mutagen.aiff import AIFF
import os
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt


class ProgressHandler:
    """
    access to current state of the normalizing process
    """
    _instance = None 

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProgressHandler, cls).__new__(cls)
            cls._instance._format = "aiff"
            cls._instance.reset() 
        return cls._instance  

    def reset(self):
        self.bar = 0  # Represents the progress bar's current value.
        self.running = False  # A flag indicating whether the process is running.
        self.terminate = False  # A flag indicating whether the process should be stopped.
        self.current_file = ""  # The name of the file currently being processed.

    @staticmethod
    def update_bar(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            ProgressHandler()._instance.bar += 1
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
    def positive_to_negative(arrays: list, indices: list) -> None:
        """
        All positive values are converted to negative.
        """
        for i in indices:
            arrays[i] = np.where(arrays[i] > 0, -arrays[i], arrays[i])
    
    @staticmethod
    def negative_to_positive(arrays: list, indices: list, max_value: int) -> None:
        """
        All negative values are converted to positive.
        """
        def int_overflow_protection(signal_array: np.ndarray, max_value: int) -> np.ndarray:
            return np.where(
                signal_array == -(max_value + 1), -max_value, signal_array
            )
            
        for i in indices:
            arrays[i] = int_overflow_protection(arrays[i], max_value)
            arrays[i] = np.where(arrays[i] < 0, np.abs(arrays[i]), arrays[i])
            
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
    
    
class File:
    @staticmethod
    @ProgressHandler.update_bar
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
    @ProgressHandler.update_bar
    def write_tags(file: str, folder: str, format: str) -> None:
        file, ext = os.path.splitext(file)
        
        try:
            tags_old = ID3(f"{folder}/{file}{ext}")
        except:
            print("No tags found")
            return None
            
        file = file.replace("â€“", "&")
        tags = {"aiff": AIFF(), "mp3": ID3()}[format]
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
        
        tags.save(f"{folder}/Normalized Files/{file}.{format}", v2_version=3)

    @staticmethod
    @ProgressHandler.update_bar
    def save_as(signal_array: np.ndarray, file_data: dict, folder: str, format: str) -> None:
        new_audio = AudioSegment(
        signal_array.tobytes(),
        frame_rate=file_data["frame_rate"],
        sample_width=file_data["sample_width"],
        channels=file_data["channels"]
        )

        new_audio.export(f"{folder}/Normalized Files/{file_data['filename']}.{format}", format=format, bitrate="320k")

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
    

class Plot:
    @staticmethod
    def plot_find_transient_threshold(signal_array: np.ndarray, threshold: int) -> None:

        x_values = np.arange(len(signal_array))

        plt.figure()
        plt.plot(x_values, signal_array.ravel())
        plt.axhline(y=threshold, color='r', linestyle='-')
        plt.title('Plot of the Array') 
        plt.xlabel('Time') 
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_signal_array(signal_array: np.ndarray) -> None:

        x_values = np.arange(len(signal_array))

        plt.figure()
        plt.plot(x_values, signal_array.ravel()) 
        plt.title('Plot of the Array') 
        plt.xlabel('Time') 
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()
