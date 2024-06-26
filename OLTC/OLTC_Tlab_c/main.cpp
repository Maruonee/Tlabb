#include <QApplication>
#include "MainWindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MainWindow mainWindow;
    mainWindow.show();
    return app.exec();
}

/*
project/
│
├── main.cpp
├── CMakeLists.txt
├── include/
│   ├── MainWindow.h
│   ├── RecorderWorker.h
│   ├── DataCollectorWorker.h
│   ├── Logger.h
│   ├── Utils.h
├── src/
│   ├── MainWindow.cpp
│   ├── RecorderWorker.cpp
│   ├── DataCollectorWorker.cpp
│   ├── Logger.cpp
│   ├── Utils.cpp
*/