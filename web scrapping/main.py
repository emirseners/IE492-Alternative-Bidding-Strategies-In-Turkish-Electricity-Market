import pyautogui
import time
from datetime import datetime, timedelta
import pyperclip

x1, y1 = 226, 307
x2, y2 = 1378, 344
x3, y3 = 1278, 415
x5, y5 = 316, 312

current_datetime = datetime.strptime("24.07.2024 15:00", "%d.%m.%Y %H:%M")

num_iterations = 5000

hours = []

for _ in range(num_iterations):
    date_str = current_datetime.strftime("%d.%m.%Y %H:%M")
    hours.append(date_str)
    current_datetime += timedelta(hours=1)

pyautogui.click(1221, 314)
time.sleep(0.5)

for i in hours:
    pyperclip.copy(i)
    time.sleep(0.3)

    pyautogui.click(x1, y1)
    time.sleep(0.3)

    pyautogui.hotkey('command', 'a')
    time.sleep(0.3)

    pyautogui.press('delete')
    time.sleep(0.3)

    pyautogui.hotkey('command', 'v')
    time.sleep(0.3)

    pyautogui.click(x5, y5)
    time.sleep(10)

    pyautogui.click(x2, y2)
    time.sleep(0.5)

    pyautogui.click(x3, y3)
    time.sleep(1)