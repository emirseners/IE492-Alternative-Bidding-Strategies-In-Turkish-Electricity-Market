import time
from datetime import datetime, timedelta
current_datetime = datetime.strptime("01.01.2024 00:00", "%d.%m.%Y %H:%M")



for _ in range(100):

    date_str = current_datetime.strftime("%d.%m.%Y %H:%M")
    print(date_str)
    print(type(date_str))
    current_datetime += timedelta(hours=1)