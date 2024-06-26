#include "Logger.h"

Logger::Logger(QObject *parent) : QObject(parent) {
}

void Logger::log(const QString &message) {
    emit logMessage(message);
}
