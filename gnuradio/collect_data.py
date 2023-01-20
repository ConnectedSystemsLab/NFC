#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Collect Data
# Author: bill
# GNU Radio version: 3.10.3.0

from packaging.version import Version as StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import blocks
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import nfc
from gnuradio import uhd
import time



from gnuradio import qtgui

class collect_data(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Collect Data", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Collect Data")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "collect_data")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.y = y = 0
        self.x = x = [[10,10],[20,20],[30,30],[20,10]]
        self.samp_rate = samp_rate = 2e6
        self.d = d = 0

        ##################################################
        # Blocks
        ##################################################
        self.variable_qtgui_msg_push_button_0 = _variable_qtgui_msg_push_button_0_toggle_button = qtgui.MsgPushButton('variable_qtgui_msg_push_button_0', 'pressed',1,"default","default")
        self.variable_qtgui_msg_push_button_0 = _variable_qtgui_msg_push_button_0_toggle_button

        self.top_grid_layout.addWidget(_variable_qtgui_msg_push_button_0_toggle_button, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=[1],
            ),
        )
        self.uhd_usrp_source_0.set_clock_source('external', 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_source_0.set_center_freq(5.95e6, 0)
        self.uhd_usrp_source_0.set_antenna('A', 0)
        self.uhd_usrp_source_0.set_gain(10, 0)
        self.qtgui_ledindicator_0 = self._qtgui_ledindicator_0_win = qtgui.GrLEDIndicator("", "green", "red", False, 40, 1, 1, 1, self)
        self.qtgui_ledindicator_0 = self._qtgui_ledindicator_0_win
        self.top_grid_layout.addWidget(self._qtgui_ledindicator_0_win, 0, 1, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            32768, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            5.95e6, #fc
            samp_rate, #bw
            "", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(False)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)
        self.nfc_series_reader_0 = nfc.series_reader(self.get_x)
        self.nfc_controlled_splitter_0 = nfc.controlled_splitter(3000000)
        self.nfc_arduino_0 = nfc.arduino('/dev/ttyACM1')
        self.blocks_msgpair_to_var_0 = blocks.msg_pair_to_var(self.set_y)
        self.blocks_message_debug_0 = blocks.message_debug(True)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, 'results/'+str(y), True)
        self.blocks_file_sink_0.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.nfc_arduino_0, 'finished'), (self.blocks_msgpair_to_var_0, 'inpair'))
        self.msg_connect((self.nfc_arduino_0, 'finished'), (self.nfc_controlled_splitter_0, 'enable'))
        self.msg_connect((self.nfc_controlled_splitter_0, 'msg_out'), (self.blocks_message_debug_0, 'print'))
        self.msg_connect((self.nfc_controlled_splitter_0, 'msg_out'), (self.nfc_series_reader_0, 'trigger'))
        self.msg_connect((self.nfc_controlled_splitter_0, 'msg_out'), (self.qtgui_ledindicator_0, 'state'))
        self.msg_connect((self.nfc_series_reader_0, 'value'), (self.blocks_message_debug_0, 'print'))
        self.msg_connect((self.nfc_series_reader_0, 'value'), (self.nfc_arduino_0, 'position'))
        self.msg_connect((self.variable_qtgui_msg_push_button_0, 'pressed'), (self.blocks_message_debug_0, 'print'))
        self.msg_connect((self.variable_qtgui_msg_push_button_0, 'pressed'), (self.nfc_controlled_splitter_0, 'enable'))
        self.connect((self.nfc_controlled_splitter_0, 1), (self.blocks_file_sink_0, 0))
        self.connect((self.nfc_controlled_splitter_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.nfc_controlled_splitter_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "collect_data")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_y(self):
        return self.y

    def set_y(self, y):
        self.y = y
        self.blocks_file_sink_0.open('results/'+str(self.y))

    def get_x(self):
        return self.x

    def set_x(self, x):
        self.x = x

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_freq_sink_x_0.set_frequency_range(5.95e6, self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_d(self):
        return self.d

    def set_d(self, d):
        self.d = d




def main(top_block_cls=collect_data, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
