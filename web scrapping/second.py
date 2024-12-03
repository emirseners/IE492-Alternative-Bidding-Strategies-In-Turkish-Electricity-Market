import pyautogui
import time

time.sleep(1)

current_mouse_x, current_mouse_y = pyautogui.position()
print(f"The current mouse position is ({current_mouse_x}, {current_mouse_y})")