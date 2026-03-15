# cat-tracker
A cat video tracker. All this program does is turn on an LED if a cat is in the picture and turns it off when there is no cat.

## Hardware
- Raspberry Pi 4
- Logitech USB Webcam (Old 720p webcam)
- LED

## Setup
- Plug a webcam into a USB port on your Pi
- Use GPIO 17 (Pin 11) as Output to your LED
- Attach your LED to ground using any ground pin

Load the program onto your Raspberry Pi, if all goes well when a cat enters your camera's field of view, the LED should light up.
When it leaves, it will turn off.