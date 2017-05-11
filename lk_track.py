#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

from sklearn.naive_bayes import GaussianNB
import numpy as np
import cv2
from common import anorm2, draw_str
from time import clock
from os import sys

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0

    def run(self, frames):
        values = []
        mvmts = []
        while True:
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                values.append(abs(p0-p0r))

                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                track_sums = []
                for track in new_tracks:
                    # find change in y position from intial position to ending position of featured element
                    track_sums.append(track[0][1]-track[len(track)-1][1])
                mvmts.append(track_sums)
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
            ch = cv2.waitKey(1)

            # run as long as we have more frames (frames-1 to make sure we stay in bounds)
            if len(mvmts) == frames-1:
                return mvmts


def main():
    # grab values from args
    training_file = sys.argv[1]
    training_video = sys.argv[2]
    training_frames = int(sys.argv[3])
    test_video = sys.argv[4]
    test_frames = int(sys.argv[5])

    # position changes (movements) from training video
    training_mvmts = App(training_video).run(training_frames)

    # get the average position change from training_mvmts (1 average / 1 frame)
    training_averages = []
    for sum_set in training_mvmts:
        avg = []
        avg.append(sum(sum_set)/len(sum_set))
        training_averages.append(avg)

    # grab training data from
    speeds = []
    with open(training_file) as f:
        data = f.readlines()
        # multiply each speed by 1000000 because np.array doesn't support floats - maintains accuracy of decimals
        data = [int(float(x.strip())*1000000) for x in data]
        speeds = data
    speeds = speeds[:len(training_mvmts)]

    # create naive bayes model from sklearn using optical flow averages and speed data
    x = np.array(training_averages)
    y = np.array(speeds)

    model = GaussianNB()

    # train the model using the training sets
    model.fit(x, y)

    # position changes (movements) from test video
    test_mvmts = App(test_video).run(test_frames)

    test_averages = []
    for sum_set in test_mvmts:
        avg = []
        avg.append(sum(sum_set)/len(sum_set))
        test_averages.append(avg)

    predicted= model.predict(test_averages)
    for speed in predicted:
        # divide by 1000000.0 to reverse *1000000 from earlier because np.array doesn't support floats - maintains accuracy of decimals
        print(speed/1000000.0)


if __name__ == '__main__':
    main()
