#include "MainWindow.h"
#include <QTimer>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QFileDialog>
#include "Utils.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), logger(new Logger(this)), recorderWorker(nullptr), dataCollectorWorker(nullptr),
      recorderThread(nullptr), dataCollectorThread(nullptr), statusTimer(new QTimer(this)),
      statusVisible(true), machineError(0), tapPosition(5), tapVoltage(0), tapUp(0), tapDown(0), stopFlag(false) {
    
    QWidget *centralWidget = new QWidget(this);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    QPushButton *startButton = new QPushButton("Start", this);
    QPushButton *stopButton = new QPushButton("Stop", this);
    stopButton->setEnabled(false);

    QLabel *statusLabel = new QLabel("Status: Ready", this);

    mainLayout->addWidget(startButton);
    mainLayout->addWidget(stopButton);
    mainLayout->addWidget(statusLabel);

    setCentralWidget(centralWidget);

    connect(startButton, &QPushButton::clicked, this, &MainWindow::startCollection);
    connect(stopButton, &QPushButton::clicked, this, &MainWindow::stopCollection);
    connect(statusTimer, &QTimer::timeout, this, &MainWindow::updateStatus);

    connect(logger, &Logger::logMessage, statusLabel, &QLabel::setText);
}

MainWindow::~MainWindow() {
    if (recorderThread) {
        recorderThread->quit();
        recorderThread->wait();
        delete recorderWorker;
    }
    if (dataCollectorThread) {
        dataCollectorThread->quit();
        dataCollectorThread->wait();
        delete dataCollectorWorker;
    }
}

void MainWindow::startCollection() {
    QString folderPath = QFileDialog::getExistingDirectory(this, "Select Directory");
    if (folderPath.isEmpty())
        return;

    int duration = 60;  // Example value
    int sampleRate = 44100;  // Example value
    int repeatNum = 60;  // Example value
    QString expDate = "210101";  // Example value
    int expNum = 1;  // Example value

    recorderWorker = new RecorderWorker(folderPath, duration, sampleRate, repeatNum);
    recorderThread = new QThread(this);
    recorderWorker->moveToThread(recorderThread);

    connect(recorderThread, &QThread::started, recorderWorker, &RecorderWorker::startRecording);
    connect(recorderWorker, &RecorderWorker::logMessage, logger, &Logger::log);
    connect(recorderWorker, &RecorderWorker::finished, this, &MainWindow::onCollectionFinished);

    recorderThread->start();
}

void MainWindow::stopCollection() {
    stopFlag = true;
    if (recorderWorker)
        recorderWorker->stopRecording();
    if (dataCollectorWorker)
        dataCollectorWorker->stopCollection();
}

void MainWindow::updateStatus() {
    // Update status logic here
}

void MainWindow::onCollectionFinished() {
    recorderThread->quit();
    recorderThread->wait();
    delete recorderWorker;
    recorderWorker = nullptr;
}
