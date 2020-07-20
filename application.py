from PyQt5.QtWidgets import QPushButton, QApplication, QWidget, QVBoxLayout, QSlider, QStyle, \
    QHBoxLayout, QFileDialog, QLabel, QProgressBar
from PyQt5.QtCore import QUrl, Qt, QRunnable, pyqtSignal, QObject, QThreadPool
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import process_videos
import os
import time
import uuid


def position_to_time(pos):
    return time.strftime('%M:%S', time.gmtime(pos // 1000))


class VideoPlayer(QWidget):
    def __init__(self, path):
        super().__init__()

        video_widget = QVideoWidget()

        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
        self.player.setVideoOutput(video_widget)
        self.player.stateChanged.connect(self.state_changed)
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_video)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)

        self.time_elapsed_label = QLabel()
        self.duration_label = QLabel()

        video_control_layout = QHBoxLayout()
        video_control_layout.setContentsMargins(0, 0, 0, 0)
        video_control_layout.addWidget(self.time_elapsed_label)
        video_control_layout.addWidget(self.slider)
        video_control_layout.addWidget(self.play_button)
        video_control_layout.addWidget(self.duration_label)

        main_layout = QVBoxLayout()
        main_layout.addWidget(video_widget)
        main_layout.addLayout(video_control_layout)

        self.setLayout(main_layout)

    def reset(self, path):
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))

    def play_video(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def state_changed(self, _):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.play_button.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))

    def position_changed(self, position):
        self.slider.setValue(position)
        self.time_elapsed_label.setText(position_to_time(position))

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)
        self.duration_label.setText(position_to_time(duration))

    def set_position(self, position):
        self.player.setPosition(position)
        self.time_elapsed_label.setText(position_to_time(position))


class VideoProcesserSignals(QObject):
    result = pyqtSignal(tuple)


class VideoProcesser(QRunnable):
    def __init__(self, fn, args):
        super().__init__()
        self.fn = fn
        self.args = args
        self.signals = VideoProcesserSignals()

    def run(self):
        res = self.fn(*self.args)
        self.signals.result.emit(res)


class ImageShow(QWidget):
    def __init__(self):
        super().__init__()

        self.img = QPixmap()

        self.label = QLabel(self)
        self.label.setPixmap(self.img)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)

        self.setLayout(self.layout)

    def set_image(self, inp_img):
        img = QImage(inp_img.data, inp_img.shape[1], inp_img.shape[0], inp_img.shape[1] * 3, QImage.Format_RGB888)
        img = img.rgbSwapped()
        self.img.convertFromImage(img)
        self.label.setPixmap(self.img)


class AppWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(250, 100, 800, 600)

        self.train_path_label = QLabel()
        self.train_path_label.setStyleSheet("border: 1px solid black;")
        self.load_train_button = QPushButton('Load train video')

        self.start_button = QPushButton('Start!')

        self.image_show = None
        self.video_player = None
        self.grade_label = QLabel()

        self.train_path = None

        self.tmp_dir = os.getcwd() + '/DanceApp'
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        self.out_path = self.tmp_dir + '/out.mp4'

        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)

        self.load_train_button.clicked.connect(self.load_train_clicked)
        self.start_button.clicked.connect(self.start_clicked)

        train_layout = QHBoxLayout()
        train_layout.addWidget(self.train_path_label)
        train_layout.addWidget(self.load_train_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setHidden(True)

        self.layout = QVBoxLayout()
        self.layout.addLayout(train_layout)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.grade_label)
        self.layout.addWidget(self.progress_bar)
        self.setLayout(self.layout)

        self.model = process_videos.load_model()

        self.video_formats = 'Video Files (*.mp4 *.flv *.ts *.mts *.avi *.mov)'

    def load_train_clicked(self):
        fname = QFileDialog.getOpenFileName(self, 'Load video file', '~/', self.video_formats)
        if fname[0] == '':
            return
        self.train_path = fname[0]
        self.load_train_button.setStyleSheet("background-color: green")
        self.train_path_label.setText(self.train_path)

    def set_image(self, img):
        self.image_show.set_image(img)

    def start_clicked(self):
        if self.train_path is None:
            return
        self.grade_label.setText('Processing...')
        self.progress_bar.setHidden(False)
        self.load_train_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.image_show = ImageShow()
        self.layout.addWidget(self.image_show)
        pr = VideoProcesser(make_video, args=(self.train_path, self.out_path, self.model, self.tmp_dir,
                                              self.progress_bar.setValue, process_videos.VideoPlayer(self.set_image)))
        pr.signals.result.connect(self.process_result)
        self.pool.start(pr)

    def process_result(self, res):
        total_err, grade = res
        self.progress_bar.setHidden(True)
        self.layout.removeWidget(self.image_show)
        self.grade_label.setText('Total error: {}, Grade: {}'.format(total_err, grade))
        self.load_train_button.setEnabled(True)
        self.start_button.setEnabled(True)
        if self.video_player is None:
            self.video_player = VideoPlayer(self.out_path)
            self.layout.addWidget(self.video_player)
        else:
            self.video_player.reset(self.out_path)


def make_video(train_path, out_path, model, tmp_dir, processing_log, image_show):
    converted_train_path = tmp_dir + '/' + str(uuid.uuid4()) + '.mp4'
    process_videos.convert_video(train_path, converted_train_path)
    return process_videos.make_video(converted_train_path, out_path, model, processing_log, image_show)


if __name__ == '__main__':
    app = QApplication([])
    a = AppWindow()
    a.show()
    app.exec_()
