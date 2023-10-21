from PyQt5.QtWidgets import QApplication
import PySimpleGUI as sg
import sys
from time import sleep
import normalizer
from pathlib import Path

sg.theme("Black")
sg.set_options(dpi_awareness=True)


repo_root_dir = Path(__file__).resolve().parent.parent
ICON = repo_root_dir / "assets" / "nor.png"


# switch object sizes from the ui for low resolution or high resolution screens
class win_lay:
    in_win_lst = [[25, 27], [50, 27]]
    in_win_win = [[280, 139], [600, 200]]
    main_win_win = [[590, 230], [1150, 435]]
    main_win_txt_bpm = [[10], [40]]
    main_win_txt_bpmind = [[10], [20]]
    main_win_txt_info = [[9], [29]]
    main_win_bar_size = [[52, 3], [86, 5]]
    main_win_bar_pad = [[10, 10], [20, 20]]
    main_win_but_start = [[10], [20]]
    main_win_but_once = [[5], [20]]
    main_win_but_link = [[10], [20]]
    main_win_but_learn = [[5], [20]]
    make_win_txt_setup = [[100, 15], [211, 20]]
    make_win_lstin = [[25, 10], [50, 20]]
    make_win_lstout = [[25, 10], [50, 30]]
    make_win_txt_info = [[5], [20]]
    make_win_but_learn_pad = [[5], [50]]
    make_win_but_learn_size = [[15], [18]]
    make_win_but_exit = [[10], [35]]
    make_win_win = [[300, 185], [600, 360]]


def check_screen_resolution() -> int:
    app = QApplication(sys.argv)
    screen = app.screens()[0]
    dpi = screen.physicalDotsPerInch()
    app.quit()
    if dpi > 150:
        return int(1)
    else:
        return int(0)


def main_window(resolution: int) -> sg.Window:
    layout = [
        [
            sg.Text(
                " 0",
                key="number_files",
                font=("", 77),
                pad=(win_lay.main_win_txt_bpm[resolution][0], 1),
            ),
            sg.Text(
                "FILES FOUND",
                key="files",
                font=("", 26),
                text_color="blue",
                pad=(win_lay.main_win_txt_bpmind[resolution][0], 1),
            ),
        ],
        [
            sg.Text(
                key="info",
                pad=(win_lay.main_win_txt_info[resolution][0], 1),
                font=("", 9),
            ),
            sg.Text(
                key="completed",
                pad=(win_lay.main_win_txt_info[resolution][0], 1),
                font=("", 9),
            ),
        ],
        [
            sg.ProgressBar(
                8,
                orientation="h",
                size=(
                    win_lay.main_win_bar_size[resolution][0],
                    win_lay.main_win_bar_size[resolution][1],
                ),
                pad=(
                    win_lay.main_win_bar_pad[resolution][0],
                    win_lay.main_win_bar_pad[resolution][1],
                ),
                border_width=0,
                key="-PROGRESS_BAR-",
                bar_color=("Blue", "Blue"),
            )
        ],
        [
            sg.Button(
                disabled=True,
                key="normalize",
                button_text="NORMALIZE FILES",
                border_width=0,
                size=(16, 1),
                pad=(win_lay.main_win_but_link[resolution][0], 2),
            ),
            sg.Button(
                key="choose_folder",
                button_text="CHOOSE FOLDER",
                border_width=0,
                pad=(win_lay.main_win_but_learn[resolution][0], 2),
                size=(15, 1),
            ),
            sg.Text(
                "Save as",
                pad=(win_lay.main_win_txt_info[resolution][0], 1),
                font=("", 9),
            ),
            sg.Button(
                key="aiff",
                button_text="AIFF",
                button_color=(("white on blue")),
                border_width=0,
                pad=(win_lay.main_win_but_learn[resolution][0], 2),
                size=(5, 1),
            ),
            sg.Button(
                key="mp3",
                button_text="MP3",
                border_width=0,
                pad=(win_lay.main_win_but_learn[resolution][0], 2),
                size=(5, 1),
            ),
        ],
    ]
    return sg.Window(
        "Audio Normalizer",
        layout,
        no_titlebar=False,
        finalize=True,
        titlebar_icon=ICON,
        size=(win_lay.main_win_win[resolution][0], win_lay.main_win_win[resolution][1]),
        titlebar_text_color="#ffffff",
        use_custom_titlebar=True,
        titlebar_background_color="#000000",
        titlebar_font=("Arial", 12),
        grab_anywhere=True,
    )


_main_window = main_window(check_screen_resolution())


class RefreshWindow:
    @staticmethod
    def popup_explorer() -> str:
        _main_window.Element("choose_folder").Update(disabled=True)
        _main_window.Element("normalize").Update(disabled=True)
        
        user_folder = sg.popup_get_folder("Choose a folder", no_window=True)
        
        _main_window.Element("choose_folder").Update(disabled=False)
        _main_window.Element("normalize").Update(disabled=False)
        
        return user_folder

    @staticmethod
    def on_click_choose_folder() -> str:
        _main_window.Element("completed").Update("", text_color="green")
        
        user_folder = RefreshWindow.popup_explorer()
        
        _main_window.Element("number_files").Update(
            normalizer.File.count_availible_files(user_folder)
        )
        _main_window.Element("info").Update(user_folder)
        _main_window.Element("normalize").Update(disabled=False)
        
        return user_folder

    @staticmethod
    def on_click_normalize() -> None:
        if not normalizer.progress.terminate:
            _main_window.Element("completed").Update("", text_color="green")
            _main_window.Element("choose_folder").Update(disabled=True)
            _main_window.Element("normalize").Update(
                ("NORMALIZING..."),
                button_color=(("white on blue")),
            )
            
        else:
            _main_window.Element("choose_folder").Update(disabled=True)
            _main_window.Element("normalize").Update(
                ("TERMINATING..."), button_color=(("white on blue")), disabled=True
            )
            
    @staticmethod
    def on_click_aiff() -> None:
        _main_window.Element("aiff").Update(button_color=(("white on blue")))
        _main_window.Element("mp3").Update(button_color=(("black on white")))
        
    @staticmethod
    def on_click_mp3() -> None:
        _main_window.Element("mp3").Update(button_color=(("white on blue")))
        _main_window.Element("aiff").Update(button_color=(("black on white")))

    @staticmethod
    def normalizer_progress() -> None:
        while normalizer.progress.running:
            sleep(0.2)
            _main_window.Element("-PROGRESS_BAR-").Update(
                normalizer.progress.bar, bar_color=("Blue", "White")
            )
            _main_window.Element("info").Update(normalizer.progress.current_file)
            
        _main_window.Element("info").Update("")
        _main_window.Element("completed").Update("DONE", text_color="green")
        _main_window.Element("normalize").Update(
            ("NORMALIZE"), button_color=(("black on white")), disabled=False
        )
        _main_window.Element("-PROGRESS_BAR-").Update(bar_color=("Blue", "Blue"))
        _main_window.Element("choose_folder").Update(disabled=False)
        
        return 0
