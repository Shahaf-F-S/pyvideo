# test.py

from pyvideo import AudioArray, volume, speed

SOURCE = "media/videos/input/milo.mp4"
DESTINATION = "media/videos/output/milo.wav"

def main() -> None:
    """A function to run the main test."""

    audio = AudioArray.load(SOURCE).copy()

    for _ in volume(audio, factor=5.5):
        pass

    speed(audio, factor=1.5)

    audio.save(DESTINATION)

if __name__ == "__main__":
    main()
