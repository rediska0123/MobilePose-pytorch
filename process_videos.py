import cv2
import numpy as np
import os
from estimator import ResEstimator
from network import CoordRegressionNetwork
from dataloader import crop_camera
import moviepy.editor as mpe
import ffmpy
from numpy.fft import fft, ifft


def load_model(model='resnet18', inp_dim=224):
    model_path = os.path.join("./models", model + "_%d_adam_best.t7" % inp_dim)
    net = CoordRegressionNetwork(n_locations=16, backbone=model).to("cpu")
    return ResEstimator(model_path, net, inp_dim)


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ('Error opening video {}'.format(path))
    return cap


def concat_images(img1, img2, height=500):
    def compress_img(img):
        w = round(img.shape[1] * height / img.shape[0])
        return cv2.resize(img, (w, height), interpolation=cv2.INTER_AREA)

    c1 = compress_img(img1)
    c2 = compress_img(img2)
    return np.concatenate((c1, c2), axis=1)


def normalize_pose(p):
    p = p.astype('float64')
    p -= np.repeat(np.expand_dims(p.mean(axis=0), axis=0), p.shape[0], axis=0)
    k = np.median(p[:, 0] ** 2 + p[:, 1] ** 2)
    if k != 0:
        p /= k
    return p


def count_pos_error(pos1, pos2):
    p1 = pos1.copy()
    p2 = pos2.copy()

    p1 = normalize_pose(p1)
    p2 = normalize_pose(p2)
    return round(((p1 - p2) ** 2).sum(axis=1).mean().item() * 1000000)


def add_error_on_frame(frame, err):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.0
    thickness = 2
    h, w = frame.shape[: 2]

    def get_text_start_point(center_point, text):
        center_point_x, center_point_y = center_point
        text_sz, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_sz_x, text_sz_y = text_sz
        return (center_point_x - text_sz_x // 2,
                center_point_y + text_sz_y // 2)

    label = str(err)
    x, y = w // 2, h - 30
    cv2.rectangle(frame, (x - 50, y - 20), (x + 50, y + 20), color=[255, 255, 255], thickness=-1)
    cv2.putText(frame, label, get_text_start_point((x, y), label),
                font, thickness=thickness, color=[0, 0, 0], fontScale=font_scale)

    return frame


def print_grade(total_err):
    print()
    print('Total err: {}'.format(total_err))

    grades = [(75, 5), (150, 4), (225, 3), (300, 2), (np.inf, 1)]

    for bound, grade in grades:
        if total_err <= bound:
            print('Grade: {}'.format(grade))
            return total_err, grade


def modify_two_videos(path1, path2, shift, frame_modifier, out_path=None, logger=None):
    cap1 = open_video(path1)
    cap2 = open_video(path2)
    fps = cap1.get(cv2.CAP_PROP_FPS)
    cap2.set(cv2.CAP_PROP_FPS, fps)
    shift = round(cap1.get(cv2.CAP_PROP_FRAME_COUNT) * shift)
    frames = round(min(
        cap1.get(cv2.CAP_PROP_FRAME_COUNT) - max(0, shift),
        cap2.get(cv2.CAP_PROP_FRAME_COUNT) + min(0, shift)
    )) + abs(shift)
    out = None

    i = 0
    while cap1.isOpened() and cap2.isOpened():
        print('\rFrame {}/{}'.format(i, frames), end='')
        if logger is not None:
            logger.log(i, frames)
        i += 1
        if shift >= 1:
            cap1.read()
            shift -= 1
            continue
        if shift <= -1:
            cap2.read()
            shift += 1
            continue
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        frame = frame_modifier(frame1, frame2)

        if out is None and out_path is not None:
            h, w = frame.shape[: 2]
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if out is not None:
            out.write(frame)

    cap1.release()
    cap2.release()
    if out is not None:
        out.release()
    print()


def smooth_poses(poses, logger=None):
    cnt = 0
    l = len(poses) - 1
    for i in range(1, l):
        prv, cur, nxt = tuple(poses[i - 1:i + 2])
        e1 = count_pos_error(prv, cur)
        e2 = count_pos_error(cur, nxt)
        e3 = count_pos_error(prv, nxt)
        if e1 + e2 > 1.5 * e3:
            cnt += 1
            poses[i] = (prv + nxt) // 2
        if logger is not None:
            logger.log(i, l)
    print('Smoothed {} out of {} frames'.format(cnt, len(poses)))
    return poses


class Logger:
    def __init__(self, callback, l_threshold, r_threshold):
        self.callback = callback
        self.l_threshold = l_threshold
        self.r_threshold = r_threshold

    def log(self, x, y):
        """x out of y done"""
        if self.callback is not None:
            self.callback((self.l_threshold * (y - x) + self.r_threshold * x) / y)


def make_video(path1, path2, out_path, res_estimator, processing_log=None):
    shift = calculate_shift(path1, path2)
    print('Shift: {}%'.format(shift * 100))
    poses1, poses2 = [], []

    def get_human_pose(frame):
        frame = crop_camera(frame)
        return res_estimator.inference(frame)

    def collect_human_poses(frame1, frame2):
        poses1.append(get_human_pose(frame1))
        poses2.append(get_human_pose(frame2))

    print('Calculating positions...')
    modify_two_videos(path1, path2, shift, collect_human_poses, None, Logger(processing_log, 0, 60))

    print('Smoothing...')
    poses1 = smooth_poses(poses1, Logger(processing_log, 60, 70))
    poses2 = smooth_poses(poses2, Logger(processing_log, 70, 80))

    frame_number = 0
    errors = []
    c = out_path.split('.')
    tmp_path = '.'.join(c[:-1]) + '_tmp.' + c[-1]

    def assemble_final_video(frame1, frame2):
        frame1 = crop_camera(frame1)
        frame2 = crop_camera(frame2)

        nonlocal frame_number
        pos1, pos2 = poses1[frame_number], poses2[frame_number]
        frame_number += 1

        ResEstimator.draw_humans(frame1, pos1, imgcopy=False)
        ResEstimator.draw_humans(frame2, pos2, imgcopy=False)
        frame = concat_images(frame1, frame2)

        err = count_pos_error(pos1, pos2)
        errors.append(err)

        return add_error_on_frame(frame, err)

    print('Assembling the final video...')
    modify_two_videos(path1, path2, shift, assemble_final_video, tmp_path, Logger(processing_log, 80, 100))

    # set audio
    output_video = mpe.VideoFileClip(tmp_path)
    dur = mpe.VideoFileClip(path1).duration
    audio_background = mpe.VideoFileClip(path1).audio.subclip(
        t_start=max(0, shift * dur),
        t_end=max(0, shift * dur) + output_video.duration
    )
    final_video = output_video.set_audio(audio_background)
    final_video.write_videofile(out_path)
    os.remove(tmp_path)

    if len(errors) != 0:
        total = round(np.mean(errors).item())
        return print_grade(total)


def convert_video(video_path, out_path):
    flags = '-r 24 -codec copy'
    ff = ffmpy.FFmpeg(inputs={video_path: None}, outputs={out_path: flags})
    ff.run()


def calculate_shift(path1, path2):
    def polymult(a, b):
        n, m = len(a), len(b)
        a1 = np.pad(a, (0, m), 'constant')
        b1 = np.pad(b, (0, n), 'constant')
        a_f, b_f = fft(a1), fft(b1)
        c_f = a_f * b_f
        return np.rint(np.real(ifft(c_f)[:n + m - 1])).astype('int64')

    def get_shift_errors(arr1, arr2):
        def get_sq_array(a, m):
            sq = np.concatenate((
                np.repeat(0, m),
                a,
                np.repeat(0, m - 1)
            )) ** 2
            s = np.cumsum(sq)
            return s[m:] - s[:len(s) - m]

        n, m = len(arr1), len(arr2)
        sq1 = get_sq_array(arr1[::-1], m)
        sq2 = get_sq_array(arr2, n)
        diff = sq1 + sq2 - 2 * polymult(arr1[::-1], arr2)
        elements_count = np.concatenate((
            np.arange(1, min(n, m) + 1),
            np.repeat(min(n, m), max(n, m) - min(n, m)),
            np.arange(min(n, m) - 1, 0, -1)
        ))
        return diff / elements_count

    audio1 = mpe.VideoFileClip(path1).audio
    audio2 = mpe.VideoFileClip(path2).audio
    if audio1 is None or audio2 is None:
        return 0
    audio1 = audio1.to_soundarray()[::10, 0]
    audio2 = audio2.to_soundarray()[::10, 0]
    k1, k2 = audio1.mean(), audio2.mean()
    audio1 *= k2
    audio2 *= k1
    fft1, fft2 = fft(audio1), fft(audio2)

    diff = get_shift_errors(fft1, fft2)
    l = len(diff)
    a, b = l // 5, (4 * l + 3) // 5
    shift = len(fft1) - 1 - (np.argmin(diff[a:b]) + a)
    return shift / len(fft1)


if __name__ == '__main__':
    path1 = 'examples/kek1.mp4'
    path2 = 'examples/kek4.mp4'
    out_path = 'examples/kek5.mp4'

    e = load_model()

    make_video(path1, path2, out_path, e)
