Edge-Vision-Streamer

基于 Epoll 与多线程的 Linux 边缘视觉流媒体并发服务器。项目实现了从底层硬件视频流采集、深度学习边缘推理，到多路非阻塞网络分发的全链路闭环。

🛠️ 技术栈
C++11 | Linux | Epoll | 多线程 (Mutex/CV) | OpenCV DNN | MobileNet-SSD**

💡 核心工程亮点

Epoll 并发推流：基于 Reactor 模式与非阻塞 Socket，实现单线程向多路 Web 端稳定推送视频流，彻底消除传统多线程阻塞带来的并发卡顿。
多线程与内存防护：利用 `std::mutex` 与条件变量构建任务池解耦图像采集与 AI 推理；引入队列阈值主动丢帧机制，宁可掉帧也坚决杜绝 OOM (内存溢出) 导致的系统死锁。
DNN 模型轻量级部署：引入 OpenCV DNN 部署 MobileNet-SSD 轻量网络，完成 Blob 预处理与高维张量解析，在纯 CPU 算力下实现 20 类目标的实时检测与 OSD 数据叠加。

📂 目录结构说明

 `linux_main.cpp`：生产环境源码。包含 Epoll 网络状态机与 V4L2 底层驱动调用的完整并发服务器代码。
 `windows_main.cpp`：开发环境源码。剥离了 Linux 网络层，用于在 Windows 宿主机上进行 OpenCV 与 DNN 模块的跨平台本地联调。
 `deploy.prototxt`：MobileNet-SSD 网络结构配置文件。
 `mobilenet_iter_73000.caffemodel`：预训练模型权重文件。