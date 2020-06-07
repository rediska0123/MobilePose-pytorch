import cv2
import numpy as np
import os
from estimator import ResEstimator
from network import CoordRegressionNetwork
from dataloader import crop_camera
import moviepy.editor as mpe


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


def modify_two_videos(path1, path2, frame_modifier, out_path=None):
    cap1 = open_video(path1)
    cap2 = open_video(path2)
    fps = cap1.get(cv2.CAP_PROP_FPS)
    cap2.set(cv2.CAP_PROP_FPS, fps)
    frames = round(min(
        cap1.get(cv2.CAP_PROP_FRAME_COUNT),
        cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    out = None

    i = 0
    while cap1.isOpened() and cap2.isOpened():
        print('\rFrame {}/{}'.format(i, frames), end='')
        i += 1
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


def smooth_poses(poses):
    cnt = 0
    for i in range(1, len(poses) - 1):
        prv, cur, nxt = tuple(poses[i - 1:i + 2])
        e1 = count_pos_error(prv, cur)
        e2 = count_pos_error(cur, nxt)
        e3 = count_pos_error(prv, nxt)
        if e1 + e2 > 1.5 * e3:
            cnt += 1
            poses[i] = (prv + nxt) // 2
    print('Smoothed {} out of {} frames'.format(cnt, len(poses)))
    return poses


def make_video(path1, path2, out_path, res_estimator):
    poses1, poses2 = [], []

    def get_human_pose(frame):
        frame = crop_camera(frame)
        return res_estimator.inference(frame)

    def collect_human_poses(frame1, frame2):
        poses1.append(get_human_pose(frame1))
        poses2.append(get_human_pose(frame2))

    print('Calculating positions...')
    modify_two_videos(path1, path2, collect_human_poses)

    print('Smoothing...')
    poses1 = smooth_poses(poses1)
    poses2 = smooth_poses(poses2)

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
    modify_two_videos(path1, path2, assemble_final_video, tmp_path)

    # set audio
    output_video = mpe.VideoFileClip(tmp_path)
    audio_background = mpe.VideoFileClip(path1).audio.subclip(t_end=output_video.duration)
    final_video = output_video.set_audio(audio_background)
    final_video.write_videofile(out_path)
    os.remove(tmp_path)

    if len(errors) != 0:
        total = round(np.mean(errors).item())
        return print_grade(total)


if __name__ == '__main__':
    path1 = 'examples/kek1.mp4'
    path2 = 'examples/kek4.mp4'
    out_path = 'examples/kek5.mp4'

    e = load_model()

    make_video(path1, path2, out_path, e)
