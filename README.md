# pyvideo

> A lightweight module for simple handling of video frames and audio.

## Installation
```
pip install python-video
```

## example
```python
from pyvideo import Video

SOURCE = "media/videos/input/milo.mp4"
DESTINATION = "media/videos/output/milo.mp4"

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
```