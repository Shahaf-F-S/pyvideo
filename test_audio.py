# test.py

from pyvideo2 import Audio

SOURCE = "media/videos/input/milo.mp4"
DESTINATION = "media/videos/output/milo.wav"

def main() -> None:
    """A function to run the main test."""

    audio = Audio.load(SOURCE)

    audio = (
        audio.
        copy().
        volume(factor=5.5).
        speed(factor=1.5)
    )

    audio.save(DESTINATION)

if __name__ == "__main__":
    main()
