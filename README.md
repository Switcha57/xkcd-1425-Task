# birdcheck

This repo is about the joke in [xkcd 1425: "Tasks"](https://xkcd.com/1425/) (September 24, 2014) by Randall Munroe aging badly. In the comic, checking whether a photo was taken in a national park is an "easy GIS lookup", while checking whether the photo is of a bird would take "a research team and five years" — the 1966 Minsky summer vision project, still unsolved half a century later.

Now both are simple problems: a pretrained MobileNetV2 classifies the photo (bird or not), and EXIF GPS coordinates plus point-in-polygon on bundled U.S. national park boundaries handle the park check when the metadata allows it.
