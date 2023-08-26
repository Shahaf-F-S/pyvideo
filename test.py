# test.py

from pyvideo.video import Video

def main() -> None:
    """A function to run the main test."""

    video = Video.load("media/videos/input/milo.mp4")

    video.save("media/videos/output/milo.mp4")
# end main

if __name__ == "__main__":
    main()
# end if