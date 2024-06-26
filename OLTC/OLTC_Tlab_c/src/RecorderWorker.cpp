#include "RecorderWorker.h"
#include <QDir>
#include <QDateTime>
#include <QFile>
#include <QTextStream>
#include <alsa/asoundlib.h>
#include <iostream>

RecorderWorker::RecorderWorker(const QString &folderPath, int duration, int sampleRate, int repeatNum, QObject *parent)
    : QObject(parent), folderPath(folderPath), duration(duration), sampleRate(sampleRate), repeatNum(repeatNum), stopFlag(false) {
}

void RecorderWorker::startRecording() {
    for (int i = 0; i < repeatNum; ++i) {
        if (stopFlag)
            break;

        QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
        QString filename = QString("%1/sound_%2.wav").arg(folderPath).arg(timestamp);

        // Add ALSA recording code here
        // For example:
        snd_pcm_t *handle;
        snd_pcm_open(&handle, "default", SND_PCM_STREAM_CAPTURE, 0);
        snd_pcm_set_params(handle, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED, 1, sampleRate, 1, 500000);

        std::vector<short> buffer(sampleRate * duration);
        snd_pcm_readi(handle, buffer.data(), sampleRate * duration);
        snd_pcm_close(handle);

        QFile file(filename);
        if (file.open(QIODevice::WriteOnly)) {
            QDataStream out(&file);
            out.writeRawData(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(short));
            file.close();
        }

        emit logMessage(QString("%1 saved.").arg(filename));
        emit progress((i + 1) * 100 / repeatNum, 100);
    }

    emit finished();
}

void RecorderWorker::stopRecording() {
    stopFlag = true;
}
