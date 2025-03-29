# test.py

from pyvideo2 import Video

SOURCE = "media/videos/input/milo.mp4"
DESTINATION = "media/videos/output/milo.mp4"

def main() -> None:
    """A function to run the main test."""

    video = Video.load(SOURCE)

    video = (
        video.
        copy().
        cut(start=100).
        crop(upper_left=(0, 0), lower_right=(368, 656)).
        rescale(factor=0.75).
        color(contrast=1.25, brightness=0.75).
        volume(factor=5.5).
        speed(factor=1.5).
        flip(horizontally=True)
    )

    video.save(DESTINATION)

if __name__ == "__main__":
    main()
