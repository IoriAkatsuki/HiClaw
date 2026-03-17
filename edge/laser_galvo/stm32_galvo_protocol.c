/*
 * 激光振镜串口协议处理器
 * 用于STM32接收上位机命令并控制DAC8563
 *
 * 协议格式 (10字节):
 * [0xAA] [0x55] [CMD] [X_H] [X_L] [Y_H] [Y_L] [PARAM] [CRC_H] [CRC_L]
 */

#include "stm32f1xx_hal.h"
#include "dac8563.h"  // 您原有的DAC8563驱动
#include <string.h>

// 串口协议定义
#define FRAME_HEAD1     0xAA
#define FRAME_HEAD2     0x55
#define FRAME_LENGTH    10

// 命令定义
#define CMD_MOVE        0x01  // 移动到指定位置
#define CMD_LASER_ON    0x02  // 打开激光
#define CMD_LASER_OFF   0x03  // 关闭激光
#define CMD_DRAW_LINE   0x04  // 绘制直线
#define CMD_DRAW_BOX    0x05  // 绘制矩形

// 接收缓冲区
#define RX_BUFFER_SIZE  128
uint8_t rx_buffer[RX_BUFFER_SIZE];
uint8_t rx_index = 0;

// 帧解析缓冲区
uint8_t frame_buffer[FRAME_LENGTH];
uint8_t frame_index = 0;
uint8_t frame_state = 0;  // 0: 等待帧头1, 1: 等待帧头2, 2: 接收数据

// 外部变量（根据您的项目调整）
extern UART_HandleTypeDef huart1;  // 串口句柄
extern TIM_HandleTypeDef htim2;    // 定时器句柄（如果需要PWM控制激光）

// 激光控制引脚（根据实际硬件调整）
#define LASER_PORT      GPIOB
#define LASER_PIN       GPIO_PIN_0

/**
 * @brief CRC16校验计算
 */
uint16_t calculate_crc16(uint8_t *data, uint16_t length)
{
    uint16_t crc = 0xFFFF;

    for (uint16_t i = 0; i < length; i++) {
        crc ^= data[i];
        for (uint8_t j = 0; j < 8; j++) {
            if (crc & 0x0001) {
                crc = (crc >> 1) ^ 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }

    return crc;
}

/**
 * @brief 激光控制
 */
void laser_control(uint8_t on)
{
    if (on) {
        HAL_GPIO_WritePin(LASER_PORT, LASER_PIN, GPIO_PIN_SET);
    } else {
        HAL_GPIO_WritePin(LASER_PORT, LASER_PIN, GPIO_PIN_RESET);
    }
}

/**
 * @brief 移动振镜到指定位置
 */
void galvo_move(uint16_t x, uint16_t y)
{
    // 调用您原有的DAC8563控制函数
    DAC_OutAB(x, y);
}

/**
 * @brief 处理接收到的命令帧
 */
void process_command_frame(uint8_t *frame)
{
    // 提取数据
    uint8_t cmd = frame[2];
    uint16_t x = (frame[3] << 8) | frame[4];
    uint16_t y = (frame[5] << 8) | frame[6];
    uint8_t param = frame[7];
    uint16_t crc_recv = (frame[8] << 8) | frame[9];

    // 校验CRC
    uint16_t crc_calc = calculate_crc16(frame, 8);
    if (crc_recv != crc_calc) {
        // CRC校验失败，忽略此帧
        return;
    }

    // 执行命令
    switch (cmd) {
        case CMD_MOVE:
            galvo_move(x, y);
            break;

        case CMD_LASER_ON:
            laser_control(1);
            break;

        case CMD_LASER_OFF:
            laser_control(0);
            break;

        case CMD_DRAW_LINE:
            // 预留：可以实现更复杂的插值算法
            galvo_move(x, y);
            break;

        case CMD_DRAW_BOX:
            // 预留：可以实现矩形绘制
            galvo_move(x, y);
            break;

        default:
            // 未知命令，忽略
            break;
    }
}

/**
 * @brief 串口字节接收处理
 * @note 在串口中断或DMA接收回调中调用
 */
void uart_byte_received(uint8_t byte)
{
    switch (frame_state) {
        case 0:  // 等待帧头1
            if (byte == FRAME_HEAD1) {
                frame_buffer[0] = byte;
                frame_index = 1;
                frame_state = 1;
            }
            break;

        case 1:  // 等待帧头2
            if (byte == FRAME_HEAD2) {
                frame_buffer[1] = byte;
                frame_index = 2;
                frame_state = 2;
            } else {
                frame_state = 0;  // 帧头不匹配，重新开始
            }
            break;

        case 2:  // 接收数据
            frame_buffer[frame_index++] = byte;

            if (frame_index >= FRAME_LENGTH) {
                // 接收完整帧，处理命令
                process_command_frame(frame_buffer);

                // 重置状态
                frame_state = 0;
                frame_index = 0;
            }
            break;
    }
}

/**
 * @brief 串口中断回调函数
 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1) {
        uint8_t byte = rx_buffer[0];
        uart_byte_received(byte);

        // 重新启动接收
        HAL_UART_Receive_IT(&huart1, rx_buffer, 1);
    }
}

/**
 * @brief 初始化串口协议处理器
 */
void galvo_protocol_init(void)
{
    // 初始化状态
    frame_state = 0;
    frame_index = 0;

    // 初始化激光控制引脚
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = LASER_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(LASER_PORT, &GPIO_InitStruct);

    // 确保激光初始状态为关闭
    laser_control(0);

    // 启动串口中断接收
    HAL_UART_Receive_IT(&huart1, rx_buffer, 1);
}

/**
 * @brief 在main函数中调用示例
 */
void example_usage(void)
{
    // 在main()函数的初始化部分添加:
    // galvo_protocol_init();

    // 主循环中不需要额外代码，所有处理在中断中完成
}

/*
 * ========== 集成说明 ==========
 *
 * 1. 将此文件添加到您的STM32项目中
 *
 * 2. 确保包含以下头文件:
 *    - stm32f1xx_hal.h (根据您的芯片型号)
 *    - dac8563.h (您原有的DAC驱动)
 *
 * 3. 在main.c中调用初始化函数:
 *    galvo_protocol_init();
 *
 * 4. 确保串口配置正确:
 *    - 波特率: 115200
 *    - 数据位: 8
 *    - 停止位: 1
 *    - 校验位: None
 *
 * 5. 根据实际硬件修改:
 *    - LASER_PORT 和 LASER_PIN (激光控制引脚)
 *    - UART实例 (huart1)
 *    - DAC_OutAB() 函数名 (如果不同)
 *
 * 6. 测试流程:
 *    a. 上位机发送移动命令，观察振镜是否移动
 *    b. 上位机发送激光开关命令，观察激光是否开关
 *    c. 运行完整标定流程
 *
 * ========== 调试技巧 ==========
 *
 * 1. 添加LED指示:
 *    - 接收到有效帧时闪烁LED
 *    - CRC校验失败时点亮错误LED
 *
 * 2. 添加串口回显:
 *    - 处理完命令后发送确认消息
 *    - 格式: "OK\n" 或 "ERR\n"
 *
 * 3. 使用逻辑分析仪:
 *    - 监测串口数据线
 *    - 检查数据格式是否正确
 */
