#include "DataCollectorWorker.h"
#include <QDir>
#include <QDateTime>
#include <QFile>
#include <QTextStream>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <iostream>

DataCollectorWorker::DataCollectorWorker(const QString &serialPort, int baudRate, const QString &folderPath, int duration, int repeatNum, QObject *parent)
    : QObject(parent), serialPort(serialPort), baudRate(baudRate), folderPath(folderPath), duration(duration), repeatNum(repeatNum), stopFlag(false) {
}

void DataCollectorWorker::startCollection() {
    int fd = open(serialPort.toStdString().c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        emit logMessage("Unable to open serial port");
        emit finished();
        return;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(fd, &tty) != 0) {
        close(fd);
        emit logMessage("Error from tcgetattr");
        emit finished();
        return;
    }

    cfsetospeed(&tty, baudRate);
    cfsetispeed(&tty, baudRate);
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 5;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        close(fd);
        emit logMessage("Error from tcsetattr");
        emit finished();
        return;
    }

    char buf[256];
    for (int i = 0; i < repeatNum; ++i) {
        if (stopFlag)
            break;

        QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
        QString filename = QString("%1/sensor_%2.txt").arg(folderPath).arg(timestamp);

        QFile file(filename);
        if (!file.open(QIODevice::WriteOnly)) {
            emit logMessage(QString("Failed to open file: %1").arg(filename));
            continue;
        }

        QTextStream out(&file);
        for (int j = 0; j < duration; ++j) {
            int n = read(fd, buf, sizeof(buf));
            if (n > 0) {
                out << QString::fromStdString(std::string(buf, n));
                emit logMessage(QString("Read %1 bytes: %2").arg(n).arg(QString::fromStdString(std::string(buf, n))));
            } else if (n < 0) {
                emit logMessage("Read error");
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        file.close();
        emit progress((i + 1) * 100 / repeatNum, 100);
    }

    close(fd);
    emit finished();
}

void DataCollectorWorker::stopCollection() {
    stopFlag = true;
}
