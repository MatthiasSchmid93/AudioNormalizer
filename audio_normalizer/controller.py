from PySimpleGUI import WINDOW_CLOSED
from threading import Thread
from .normalizer import progress, normalize_folder
from .user_interface import _main_window, RefreshWindow

 
def main():
    while True:
        event, _ = _main_window.read()
        
        if event == WINDOW_CLOSED:
            progress.terminate = True
            _main_window.close()
            return 0
        
        if event == "normalize":
            if not progress.running:
                RefreshWindow.on_click_normalize()
                Thread(target=normalize_folder, args=(user_folder,)).start()
                Thread(target=RefreshWindow.normalizer_progress).start()
            else:
                progress.terminate = True
                RefreshWindow.on_click_normalize()
                
        if event == "choose_folder":
            user_folder = RefreshWindow.on_click_choose_folder()
            
        if event == "aiff":
            RefreshWindow.on_click_aiff()
            progress._format = "aiff"
            
        if event == "mp3":
            RefreshWindow.on_click_mp3()
            progress._format = "mp3"


if __name__ == "__main__":
    main()
