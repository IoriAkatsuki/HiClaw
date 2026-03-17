// ========================================
// STM32固件补丁：添加 G 和 L 命令支持
// 位置: stm32f401ccu6_dac8563/Core/Src/main.c
// 在 HAL_UARTEx_RxEventCallback 函数中添加
// ========================================

void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t Size)
{
  if (huart->Instance == USART1)
  {
    // ===== 原有命令 =====
    if (uart1_rx_buf[0] == 'U')
    {
      flag_update = 1;
    }

    if (uart1_rx_buf[0] == 'C')
    {
      int i = uart1_rx_buf[1] - '0';
      int16_t x;
      int16_t y;
      uint16_t radius;
      sscanf((const char *)&uart1_rx_buf[2], "%hd,%hd,%hu", &x, &y, &radius);

      task_buf_1[i].type = CIRCLE;
      task_buf_1[i].pose.x = x;
      task_buf_1[i].pose.y = y;
      task_buf_1[i].params[0] = radius;
    }

    if (uart1_rx_buf[0] == 'R')
    {
      int i = uart1_rx_buf[1] - '0';
      int16_t x;
      int16_t y;
      uint16_t length;
      uint16_t height;
      sscanf((const char *)&uart1_rx_buf[2], "%hd,%hd,%hu,%hu", &x, &y, &length, &height);

      task_buf_1[i].type = RECTANGLE;
      task_buf_1[i].pose.x = x;
      task_buf_1[i].pose.y = y;
      task_buf_1[i].params[0] = length;
      task_buf_1[i].params[1] = height;
    }

    // ===== 新增：G 命令 - 立即移动到指定位置（用于校准）=====
    if (uart1_rx_buf[0] == 'G')
    {
      int16_t x, y;
      sscanf((const char *)&uart1_rx_buf[1], "%hd,%hd", &x, &y);
      dac8563_output_int16(x, y);
    }

    // ===== 新增：L 命令 - 激光开关（用于校准）=====
    if (uart1_rx_buf[0] == 'L')
    {
      if (uart1_rx_buf[1] == '1')
      {
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_SET);
      }
      else if (uart1_rx_buf[1] == '0')
      {
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_RESET);
      }
    }

    HAL_UARTEx_ReceiveToIdle_DMA(huart, uart1_rx_buf, BUF_SIZE);
  }

  memset(uart1_rx_buf, 0, BUF_SIZE);
}

// ========================================
// 命令格式说明：
// ========================================
// G<x>,<y>\n        - 移动到指定位置
//   例: G5000,5000  - 移动到 (5000, 5000)
//       G0,0        - 移动到中心
//       G-10000,0   - 移动到左侧
//
// L<0/1>\n          - 激光开关
//   例: L1          - 打开激光
//       L0          - 关闭激光
//
// ========================================
// 完整命令集：
// ========================================
// C<i><x>,<y>,<r>           - 圆形 (i=0-9)
// R<i><x>,<y>,<w>,<h>       - 矩形 (i=0-9)
// U                         - 更新执行
// G<x>,<y>                  - 立即移动（校准用）
// L<0/1>                    - 激光开关（校准用）
// ========================================
