import speech_recognition as sr

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Adjusting for ambient noise. Please wait.")
    r.adjust_for_ambient_noise(source)  # Listen for 1 second to calibrate the energy threshold for ambient noise levels
    print("Say something!")
    audio = r.listen(source)

try:
    print("Google Speech Recognition thinks you said: " + r.recognize_google(audio))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")
