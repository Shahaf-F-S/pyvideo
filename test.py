# test.py

from pyvideo import VideoArray, speed, volume, cut, crop, rescale, color, flip


SOURCE = "media/videos/input/milo.mp4"
DESTINATION = "media/videos/output/milo.mp4"


def main() -> None:
    """A function to run the main test."""

    video = VideoArray.load(SOURCE)

    operations = [
        cut(video, start=100),
        crop(video, upper_left=(0, 0), lower_right=(368, 656)),
        rescale(video, factor=0.75),
        color(video, contrast=1.25, brightness=0.75),
        speed(video, factor=1.5),
        flip(video, horizontally=True),
        volume(video.audio, factor=5.5)
    ]

    for operation in operations:
        for _ in operation:
            pass

    video.save(DESTINATION)

if __name__ == "__main__":
    main()
