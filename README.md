# Comma Intern Challenge

## How it works

My solution uses OpenCV, and parts of their sample files to find points of interest on-screen. Then, I use
optical flow to plot the change in position of these features. Since horizontal movement doesn't determine
vertical speed, I ignored the changes in x and focused on changes in y. Then I took the average of all of
this movement change across the screen for each frame.

At this point, I integrated the training data to these average y movement values, and used Naive Bayes machine
learning to establish the relationship between the two sets of data. I run the program again with the test video,
and get speed values outputted to a data file.

### How it could be improved

The first thing that comes to mind is decreasing the amount outliers, as well as improving the accuracy of the
somewhat correct data. This could be improved by applying a mask to each frame of the video that blocked out
objects like cars or other items that are not stationary. These have the potential to throw off the data because
if the car is at 0 mph, and cars are rushing by it, it may think it is moving. Inversely, if the car is moving quickly
and there are many datapoints on cars moving at the same speed, there will be little change in position and it may
think it is moving slowly.

Another way to improve the data integrity would be looking at the trend of data that comes before it. If you have
a long line of increasing speeds, it would not make sense for a speed to suddenly decrease or become zero. Establishing
a way to learn from this trend and spot inconsistencies would be a great addition.

### How I came up with it

When I first received the problem, I had no clue where to start. I knew I need to know computer vision and machine
learning, but didn't know how to tie everything together, or even integrate it into the program.

I took a step back and watched the videos a few times to understand how I would
calculate speed. No program, just watching it. My eyes were drawn to items on the screen that flew by the car, and
as the car accelerated, the focus points flew by faster.

This is what inspired my approach to the feature detection, and plotting the change in position of these points as
I discussed above.

## Setup and Run

To run this, you need to install OpenCV on your computer, as well as scikit-learn, commonly abbreviated sklearn.
OpenCV is responsible for the computer vision, and sklearn is responsible for the machine learning using Naive Bayes.

To run:

```
$ python lk_track.py trainingfile trainingvideo trainingvideoframes testvideo testvideoframes > outputfile.txt
```

`trainingfile` is the path to the text file contianing speeds for each frame of the video.

`trainingvideo` is the path to the video associated with the data file.

`trainingvideoframes` is the amount of frames in the video, or the amount of frames you'd like it to run through.

`testvideo` is the video you pass in you'd like the program to predict the speed for.

`testvideoframes` is the same as `trainingvideoframes`

`outputfile` is the name of the file you'd like to output the data to

For example, to run the script for the speed challenge:

```
$ python lk_track.py train.txt train.mp4 20400 test.mp4 10798 > results.txt
```

## Reflection

This was by far the most challenging programming challenge I've had to complete for an internship. In fact, I'd go so far as to say
it was the most challenging program I've written. Unlike learning a new language or building a new web app, this forced
me outside of my comfort zone and required me to learn computer vision, and take a crash course into machine learning.

While it may not provide the most accurate results, I believe my solution is unique, and can be iterated
upon with more knowledge of the libraries I integrated to be very powerful.

Realizing that if I spent 12 hours working on any project or learning any tool as intently as I worked on this project, I would be able
to create some really amazing things, and expand my skills in fields I never considered entering.
