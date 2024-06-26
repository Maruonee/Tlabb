#ifndef LOGGER_H
#define LOGGER_H

#include <QObject>

class Logger : public QObject {
    Q_OBJECT

public:
    explicit Logger(QObject *parent = nullptr);

signals:
    void logMessage(const QString &message);

public slots:
    void log(const QString &message);
};

#endif // LOGGER_H
