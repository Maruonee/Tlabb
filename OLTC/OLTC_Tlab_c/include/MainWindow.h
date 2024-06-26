#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include "Logger.h"
#include "RecorderWorker.h"
#include "DataCollectorWorker.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void startCollection();
    void stopCollection();
    void updateStatus();
    void onCollectionFinished();

private:
    Logger *logger;
    RecorderWorker *recorderWorker;
    DataCollectorWorker *dataCollectorWorker;
    QThread *recorderThread;
    QThread *dataCollectorThread;
    QTimer *statusTimer;
    bool statusVisible;
    int machineError;
    int tapPosition;
    int tapVoltage;
    int tapUp;
    int tapDown;
    bool stopFlag;
};

#endif // MAINWINDOW_H
