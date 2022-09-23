import requests
import os
import time

face_index = 2
while True:
    print(face_index)
    r = requests.get("https://thispersondoesnotexist.com/image")
    if r.status_code == 200:
        fname = os.path.join("input", f"face-{face_index:04}.jpeg")
        with open(fname, "wb") as f:
            f.write(r.content)
        face_index += 1
    time.sleep(1.2)
