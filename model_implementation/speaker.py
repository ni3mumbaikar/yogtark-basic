import pyttsx3


class speaker:


    def __init__(self):
        print('Speaker called')

    def speak(self):
        for i in range(0, 5):
            eng = pyttsx3.init()
            eng.say(str(5 - i))
            eng.runAndWait()
