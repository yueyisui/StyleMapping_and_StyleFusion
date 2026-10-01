import cv2
import argparse
import os


def get_video_duration(filename):
    cap = cv2.VideoCapture(filename)
    if cap.isOpened():
        rate = cap.get(5)
        frame_num = cap.get(7)
        duration = frame_num / rate
        return duration
    return -1


def parse_args(video_file, picture_path, default):
    parser = argparse.ArgumentParser(description='Process pic')
    parser.add_argument('--input', help='video to process', dest='input', default=None, type=str)
    parser.add_argument('--output', help='pic to store', dest='output', default=None, type=str)
    parser.add_argument('--skip_frame', dest='skip_frame', help='skip number of video', default=default, type=int)
    args = parser.parse_args(['--input', video_file, '--output', picture_path])
    return args


def process_video(i_video, o_video, num):
    cap = cv2.VideoCapture(i_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:", fps)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    expand_name = '.png'
    if not cap.isOpened():
        print("Please check the path.")
    cnt = 0
    count = 0
    while 1:
        ret, frame = cap.read()
        cnt += 1
        if cnt % num == 0:
            count += 1
            cv2.imwrite(os.path.join(o_video, str(count).zfill(4) + expand_name), frame)

        if not ret:
            break


if __name__ == '__main__':
    video_file = ".MP4"
    picture_path = ""
    frame = 1
    cap = cv2.VideoCapture(video_file)

    frames_num = cap.get(7)
    video_duration = get_video_duration(video_file)
    print("iamges number: ", int(frames_num / frame))

    args = parse_args(video_file, picture_path, default=frame)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    process_video(args.input, args.output, args.skip_frame)
