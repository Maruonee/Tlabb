#include "Utils.h"
#include <QDir>

QString createFolder(const QString &baseDir, const QString &expDate, int expNum, const QString &suffix) {
    QString folderName = QString("%1_%2_%3").arg(expDate).arg(expNum).arg(suffix);
    QString fullPath = QString("%1/%2").arg(baseDir).arg(folderName);
    QDir dir;
    if (!dir.exists(fullPath)) {
        dir.mkpath(fullPath);
    }
    return fullPath;
}
