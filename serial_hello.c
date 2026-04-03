#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

// 简单的串口配置函数，只配置常用参数：波特率、8N1、无流控
static int set_serial_attr(int fd, speed_t speed)
{
    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) {
        perror("tcgetattr");
        return -1;
    }

    cfsetospeed(&tty, speed);
    cfsetispeed(&tty, speed);

    // 原始模式（不做行缓冲和特殊字符处理）
    cfmakeraw(&tty);

    // 8 数据位、无校验、1 停止位
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;

    // 本地连接，启用接收
    tty.c_cflag |= (CLOCAL | CREAD);

    // 关闭硬件/软件流控
    tty.c_cflag &= ~CRTSCTS;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);

    // 设置读超时（这里主要是写演示，用不到读）
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 10; // 1.0 秒

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        return -1;
    }
    return 0;
}

// 将整数波特率转换为 termios 使用的 speed_t
static speed_t baud_to_speed(long baud)
{
    switch (baud) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
#ifdef B230400
    case 230400: return B230400;
#endif
#ifdef B460800
    case 460800: return B460800;
#endif
#ifdef B921600
    case 921600: return B921600;
#endif
#ifdef B1000000
    case 1000000: return B1000000;
#endif
#ifdef B1152000
    case 1152000: return B1152000;
#endif
    default:
        fprintf(stderr, "不支持的波特率: %ld\n", baud);
        return (speed_t)0;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "用法: %s <串口设备> [波特率]\\n", argv[0]);
        fprintf(stderr, "示例: %s /dev/ttyUSB0 1152000\\n", argv[0]);
        return 1;
    }

    const char *dev = argv[1];
    long baud = 1152000; // 默认 1152000
    if (argc >= 3) {
        baud = strtol(argv[2], NULL, 10);
    }

    speed_t speed = baud_to_speed(baud);
    if (speed == 0) {
        fprintf(stderr, "请使用受支持的波特率，例如 115200 或 1152000(内核/驱动需支持)\\n");
        return 1;
    }

    int fd = open(dev, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        perror("open serial port");
        return 1;
    }

    if (set_serial_attr(fd, speed) != 0) {
        close(fd);
        return 1;
    }

    const char *msg = "hello world\r\n"; // 带回车换行，方便串口调试助手显示
    ssize_t len = strlen(msg);

    ssize_t n = write(fd, msg, len);
    if (n != len) {
        perror("write");
        close(fd);
        return 1;
    }

    // 确保数据发送出去
    if (tcdrain(fd) != 0) {
        perror("tcdrain");
    }

    printf("已向 %s 发送: %s", dev, msg);

    close(fd);
    return 0;
}
