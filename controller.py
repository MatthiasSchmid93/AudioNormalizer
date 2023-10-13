import PySimpleGUI as sg
from threading import Thread
import normalizer
from user_interface import _main_window, RefreshWindow


class ThreadingEvents:
    def __init__(self) -> None:
        self.normalizing_thread = None
        self.refresh_window_thread = None

    def start_normalizing(self, user_folder: str) -> None:
        self.normalizing_thread = Thread(
            target=normalizer.normalize_folder, args=(user_folder,)
        )
        self.normalizing_thread.start()

    def start_refresh_window(self) -> None:
        self.refresh_window_thread = Thread(target=RefreshWindow.normalizer_progress)
        self.refresh_window_thread.start()


def main():
    threading_events = ThreadingEvents()
    while True:
        event, _ = _main_window.read()
        if event == sg.WINDOW_CLOSED:
            normalizer.progress.terminate = True
            _main_window.close()
            return 0
        if event == "normalize":
            if not normalizer.progress.running:
                RefreshWindow.on_click_normalize()
                threading_events.start_normalizing(user_folder)
                threading_events.start_refresh_window()
            else:
                normalizer.progress.terminate = True
                RefreshWindow.on_click_normalize()
        if event == "choose_folder":
            user_folder = RefreshWindow.on_click_choose_folder()


if __name__ == "__main__":
    main()
