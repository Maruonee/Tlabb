#ifndef RECORDERWORKER_H
#define RECORDERWORKER_H

#include <QObject>
#include <QString>
#include <atomic>

class RecorderWorker : public QObject {
    Q_OBJECT

public:
    RecorderWorker(const QString &folderPath, int duration, int sampleRate, int repeatNum, QObject *parent = nullptr);
    void startRecording();
    void stopRecording();

signals:
    void progress(int value, int maximum);
    void totalProgress(int value, int maximum);
    void logMessage(const QString &message);
    void finished();

private:
    QString folderPath;
    int duration;
    int sampleRate;
    int repeatNum;
    std::atomic<bool> stopFlag;
};

#endif // RECORDERWORKER_H
