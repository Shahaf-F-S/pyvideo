# test.py

from pyvideo.video import Video

SOURCE = "media/videos/input/milo.mp4"
DESTINATION = "media/videos/output/milo.mp4"

def main() -> None:
    """A function to run the main test."""

    Video.load(SOURCE).save(DESTINATION)
# end main

if __name__ == "__main__":
    main()
# end if