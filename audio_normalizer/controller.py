from PySimpleGUI import WINDOW_CLOSED
from threading import Thread
import normalizer
from user_interface import _main_window, RefreshWindow


def main():
    
    while True:
        event, _ = _main_window.read()
        
        if event == WINDOW_CLOSED:
            normalizer.progress.terminate = True
            _main_window.close()
            return 0
        
        if event == "normalize":
            if not normalizer.progress.running:
                RefreshWindow.on_click_normalize()
                Thread(target=normalizer.normalize_folder, args=(user_folder,)).start()
                Thread(target=RefreshWindow.normalizer_progress).start()
            else:
                normalizer.progress.terminate = True
                RefreshWindow.on_click_normalize()
                
        if event == "choose_folder":
            user_folder = RefreshWindow.on_click_choose_folder()


if __name__ == "__main__":
    main()
