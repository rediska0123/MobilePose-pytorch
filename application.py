from PyQt5.QtWidgets import QPushButton, QApplication, QWidget, QVBoxLayout, QSlider, QStyle, \
    QHBoxLayout, QFileDialog, QLabel
from PyQt5.QtCore import QUrl, Qt, QRunnable, pyqtSignal, QObject, QThreadPool
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from process_videos import load_model, make_video
import os
import time


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


class WorkerSignals(QObject):
    result = pyqtSignal(tuple)


class VideoProcesser(QRunnable):
    def __init__(self, fn, args):
        super().__init__()
        self.fn = fn
        self.args = args
        self.signals = WorkerSignals()

    def run(self):
        res = self.fn(*self.args)
        self.signals.result.emit(res)


class AppWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(250, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.train_path_label = QLabel()
        self.train_path_label.setStyleSheet("border: 1px solid black;")
        self.load_train_button = QPushButton('Load train video')

        self.test_path_label = QLabel()
        self.test_path_label.setStyleSheet("border: 1px solid black;")
        self.load_test_button = QPushButton('Load test video')

        self.start_button = QPushButton('Compare!')

        self.video_player = None
        self.grade_label = QLabel()

        self.train_path = None
        self.test_path = None

        self.tmp_dir = os.getcwd() + '/DanceApp'
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        self.out_path = self.tmp_dir + '/out.mp4'

        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)

        self.load_train_button.clicked.connect(self.load_train_clicked)
        self.load_test_button.clicked.connect(self.load_test_clicked)
        self.start_button.clicked.connect(self.start_clicked)

        train_layout = QHBoxLayout()
        train_layout.addWidget(self.train_path_label)
        train_layout.addWidget(self.load_train_button)
        self.layout.addLayout(train_layout)

        test_layout = QHBoxLayout()
        test_layout.addWidget(self.test_path_label)
        test_layout.addWidget(self.load_test_button)
        self.layout.addLayout(test_layout)

        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.grade_label)
        self.setLayout(self.layout)

        self.model = load_model()

    def load_train_clicked(self):
        fname = QFileDialog.getOpenFileName(self, 'Load mp4 file', '~/', 'mp4')
        if fname[0] == '':
            return
        self.train_path = fname[0]
        self.load_train_button.setStyleSheet("background-color: green")
        self.train_path_label.setText(self.train_path)

    def load_test_clicked(self):
        fname = QFileDialog.getOpenFileName(self, 'Load mp4 file', '~/', 'mp4')
        if fname[0] == '':
            return
        self.test_path = fname[0]
        self.load_test_button.setStyleSheet("background-color: green")
        self.test_path_label.setText(self.test_path)

    def start_clicked(self):
        if self.train_path is None or self.test_path is None:
            return
        if self.video_player is not None:
            self.layout.removeWidget(self.video_player)
        self.load_train_button.setEnabled(False)
        self.load_test_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.grade_label.setText('Processing...')
        pr = VideoProcesser(make_video, args=(self.train_path, self.test_path, self.out_path, self.model))
        pr.signals.result.connect(self.process_result)
        self.pool.start(pr)

    def process_result(self, res):
        total_err, grade = res
        self.grade_label.setText('Total error: {}, Grade: {}'.format(total_err, grade))
        self.load_train_button.setEnabled(True)
        self.load_test_button.setEnabled(True)
        self.start_button.setEnabled(True)
        if self.video_player is None:
            self.video_player = VideoPlayer(self.out_path)
            self.layout.addWidget(self.video_player)
        else:
            self.video_player.reset(self.out_path)


if __name__ == '__main__':
    app = QApplication([])
    a = AppWindow()
    a.show()
    app.exec_()
