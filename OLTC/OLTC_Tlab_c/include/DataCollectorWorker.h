#ifndef DATACOLLECTORWORKER_H
#define DATACOLLECTORWORKER_H

#include <QObject>
#include <QString>
#include <atomic>

class DataCollectorWorker : public QObject {
    Q_OBJECT

public:
    DataCollectorWorker(const QString &serialPort, int baudRate, const QString &folderPath, int duration, int repeatNum, QObject *parent = nullptr);
    void startCollection();
    void stopCollection();

signals:
    void progress(int value, int maximum);
    void totalProgress(int value, int maximum);
    void logMessage(const QString &message);
    void finished();

private:
    QString serialPort;
    int baudRate;
    QString folderPath;
    int duration;
    int repeatNum;
    std::atomic<bool> stopFlag;
};

#endif // DATACOLLECTORWORKER_H
