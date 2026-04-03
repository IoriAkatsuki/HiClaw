/**
  ******************************************************************************
  * @file           : zzy_lib_uart_f4.c
  * @brief          : a uart lib for stm32f4xx
  ******************************************************************************
  * @attention
  *
  * Change Logs:
  * Date           Author       Notes
  * 2025-5-28      zzy          the first version
  * 2025-7-12      zzy          add configuration macro define ZZY_USE_ERROR_PRINTF ZZY_USE_AT_FUNCS ZZY_USE_PRINTF
  * 2025-7-17      zzy          add IDLE calback
  *
  ******************************************************************************
  */

#include "zzy_lib_uart_f4.h" // 头文件
#if ZZY_USE_AT_FUNCS
#include "zzy_lib_delay.h"
#endif
#include <stdarg.h> // 用于支持不定长参数
#include "stdio.h"  // 用于支持函数printf、spritnf等
#include "string.h" // 用于支持字符串处理函数strset、strcpy、memset、metcpy等

/**
 * @brief 自定义串口结构体
 */
typedef struct
{
  uint16_t usRxNum;  // 新一帧数据，接收到多少个字节数据
  uint8_t *puRxData; // 新一帧数据，数据缓存; 存放的是空闲中断后，从临时接收缓存复制过来的完整数据，并非接收过程中的不完整数据;

  uint8_t *puTxFiFoData; // 发送缓冲区，环形队列; 为了方便理解阅读，没有封装成队列函数
  uint16_t usTxFiFoData; // 环形缓冲区的队头
  uint16_t usTxFiFoTail; // 环形缓冲区的队尾

  uart_callback pIDLEcallback; // 串口空闲中断回调函数
} xUSATR_TypeDef;

//////////////////////////////////////////////////////////////   UART-1   ///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if UART1_EN

static xUSATR_TypeDef xUART1 = {0};                  // 定义 UART1 的收发结构体
static uint8_t uaUART1RxData[UART1_RX_BUF_SIZE];     // 定义 UART1 的接收缓存
static uint8_t uaUART1TxFiFoData[UART1_TX_BUF_SIZE]; // 定义 UART1 的发送缓存

/******************************************************************************
 * 函  数： UART1_Init
 * 功  能： 初始化USART2的通信引脚、协议参数、中断优先级
 *          协议：波特率-None-8-1
 *          发送：发送中断
 *          接收：接收+空闲中断
 * 返回值： 无
 ******************************************************************************/
void UART1_Init(void)
{
  // 中断优选级配置
  HAL_NVIC_SetPriority(USART1_IRQn, 1, 1); // 配置中断线的优先级
  HAL_NVIC_EnableIRQ(USART1_IRQn);         // 使能中断线
  // 串口中断设置
  USART1->CR1 &= ~(0x01 << 7); // 关闭发送中断
  USART1->CR1 |= 0x01 << 5;    // 使能接收中断: 接收缓冲区非空
  USART1->CR1 |= 0x01 << 4;    // 使能空闲中断：超过1字节时间没收到新数据
  USART1->SR = ~(0x00F0);      // 清理中断
  // 打开串口
  USART1->CR1 |= 0x01 << 13; // 使能UART开始工作
  // 关联缓冲区
  xUART1.puRxData = uaUART1RxData;         // 关联接收缓冲区的地址
  xUART1.puTxFiFoData = uaUART1TxFiFoData; // 关联发送缓冲区的地址
}

/******************************************************************************
 * 函  数： USART1_IRQHandler
 * 功  能： USART1的接收中断、空闲中断、发送中断
 * 参  数： 无
 * 返回值： 无
 * 备  注： 本函数，当产生中断事件时，由硬件调用。
 *          如果使用本文件代码，在工程文件的其它地方，要注释同名函数，否则冲突。
******************************************************************************/
void USART1_IRQHandler(void)
{
  static uint16_t cnt = 0;                  // 接收字节数累计：每一帧数据已接收到的字节数
  static uint8_t rxTemp[UART1_RX_BUF_SIZE]; // 接收数据缓存数组：每新接收１个字节，先顺序存放到这里，当一帧接收完(发生空闲中断), 再转存到全局变量：xUSART.puRxData[xx]中；

  // 发送中断：用于把环形缓冲的数据，逐字节发出
  if ((USART1->SR & 1 << 7) && (USART1->CR1 & 1 << 7)) // 检查TXE(发送数据寄存器空)、TXEIE(发送缓冲区空中断使能)
  {
    USART1->DR = xUART1.puTxFiFoData[xUART1.usTxFiFoTail++]; // 把要发送的字节，放入USART的发送寄存器
    if (xUART1.usTxFiFoTail == UART1_TX_BUF_SIZE)            // 如果数据指针到了尾部，就重新标记到0
      xUART1.usTxFiFoTail = 0;
    if (xUART1.usTxFiFoTail == xUART1.usTxFiFoData)
      USART1->CR1 &= ~(0x01 << 7); // 已发送完成，关闭发送缓冲区空置中断 TXEIE
    return;
  }

  // 接收中断：用于逐个字节接收，存放到临时缓存
  if (USART1->SR & (0x01 << 5)) // 检查RXNE(读数据寄存器非空标志位); RXNE中断清理方法：读DR时自动清理；
  {
    if ((cnt >= UART1_RX_BUF_SIZE)) //||(xUART1.ReceivedFlag==1   // 判断1: 当前帧已接收到的数据量，已满(缓存区), 为避免溢出，本包后面接收到的数据直接舍弃．
    {
      // 判断2: 如果之前接收好的数据包还没处理，就放弃新数据，即，新数据帧不能覆盖旧数据帧，直至旧数据帧被处理．缺点：数据传输过快于处理速度时会掉包；好处：机制清晰，易于调试
#if ZZY_USE_ERROR_PRINTF
      printf("警告：UART1单帧接收量，已超出接收缓存大小\n!");
#endif
      USART1->DR; // 读取数据寄存器的数据，但不保存．主要作用：读DR时自动清理接收中断标志；
      return;
    }
    rxTemp[cnt++] = USART1->DR; // 把新收到的字节数据，顺序存放到RXTemp数组中；注意：读取DR时自动清零中断位；
    return;
  }

  // 空闲中断：用于判断一帧数据结束，整理数据到外部备读
  if (USART1->SR & (0x01 << 4)) // 检查IDLE(空闲中断标志位); IDLE中断标志清理方法：序列清零，USART1 ->SR;  USART1 ->DR;
  {
    xUART1.usRxNum = 0;                                 // 把接收到的数据字节数清0
    memcpy(xUART1.puRxData, rxTemp, UART1_RX_BUF_SIZE); // 把本帧接收到的数据，存入到结构体的数组成员xUARTx.puRxData中, 等待处理; 注意：复制的是整个数组，包括0值，以方便字符串输出时尾部以0作字符串结束符
    xUART1.usRxNum = cnt;                               // 把接收到的字节数，存入到结构体变量xUARTx.usRxNum中；
    cnt = 0;                                            // 接收字节数累计器，清零; 准备下一次的接收
    memset(rxTemp, 0, UART1_RX_BUF_SIZE);               // 接收数据缓存数组，清零; 准备下一次的接收
    if (xUART1.pIDLEcallback != NULL)
    {                                                   // 判断回调函数不为空
      xUART1.pIDLEcallback(UART1_GetRxData(), UART1_GetRxNum()); // 执行空闲中断回调函数
    }
    USART1->SR;
    USART1->DR; // 清零IDLE中断标志位!! 序列清零，顺序不能错!!
    return;
  }

  return;
}

/******************************************************************************
 * 函  数： UART1_SendData
 * 功  能： UART通过中断发送数据
 *         【适合场景】本函数可发送各种数据，而不限于字符串，如int,char
 *         【不 适 合】注意h文件中所定义的发缓冲区大小、注意数据压入缓冲区的速度与串口发送速度的冲突
 * 参  数： uint8_t*  puData   需发送数据的地址
 *          uint16_t  usNum    发送的字节数 ，数量受限于h文件中设置的发送缓存区大小宏定义
 * 返回值： 无
 ******************************************************************************/
void UART1_SendData(uint8_t *puData, uint16_t usNum)
{
  for (uint16_t i = 0; i < usNum; i++) // 把数据放入环形缓冲区
  {
    xUART1.puTxFiFoData[xUART1.usTxFiFoData++] = puData[i]; // 把字节放到缓冲区最后的位置，然后指针后移
    if (xUART1.usTxFiFoData == UART1_TX_BUF_SIZE)           // 如果指针位置到达缓冲区的最大值，则归0
      xUART1.usTxFiFoData = 0;
  } // 为了方便阅读理解，这里没有把此部分封装成队列函数，可以自行封装

  if ((USART1->CR1 & 1 << 7) == 0) // 检查USART寄存器的发送缓冲区空置中断(TXEIE)是否已打开
    USART1->CR1 |= 1 << 7;         // 打开TXEIE中断
}

/******************************************************************************
 * 函  数： UART1_SendString
 * 功  能： 发送字符串
 *          用法请参考printf，及示例中的展示
 *          注意，最大发送字节数为512-1个字符，可在函数中修改上限
 * 参  数： const char *pcString, ...   (如同printf的用法)
 * 返回值： 无
 ******************************************************************************/
void UART1_SendString(const char *pcString, ...)
{
  char mBuffer[512] = {0};                             // 开辟一个缓存, 并把里面的数据置0
  va_list ap;                                          // 新建一个可变参数列表
  va_start(ap, pcString);                              // 列表指向第一个可变参数
  vsnprintf(mBuffer, 512, pcString, ap);               // 把所有参数，按格式，输出到缓存; 参数2用于限制发送的最大字节数，如果达到上限，则只发送上限值-1; 最后1字节自动置'\0'
  va_end(ap);                                          // 清空可变参数列表
  UART1_SendData((uint8_t *)mBuffer, strlen(mBuffer)); // 把字节存放环形缓冲，排队准备发送
}

/******************************************************************************
 * 函    数： UART1_SendAT
 * 功    能： 发送AT命令, 并等待返回信息
 * 参    数： char     *pcString      AT指令字符串
 *            char     *pcAckString   期待的指令返回信息字符串
 *            uint16_t  usTimeOut     发送命令后等待的时间，毫秒
 *
 * 返 回 值： 0-执行失败、1-执行正常
 ******************************************************************************/
#if ZZY_USE_AT_FUNCS
uint8_t UART1_SendAT(char *pcAT, char *pcAckString, uint16_t usTimeOutMs)
{
  UART1_ClearRx();        // 清0
  UART1_SendString(pcAT); // 发送AT指令字符串

  while (usTimeOutMs--) // 判断是否起时(这里只作简单的循环判断次数处理）
  {
    if (UART1_GetRxNum()) // 判断是否接收到数据
    {
      UART1_ClearRx();                                    // 清0接收字节数; 注意：接收到的数据内容 ，是没有被清0的
      if (strstr((char *)UART1_GetRxData(), pcAckString)) // 判断返回数据中是否有期待的字符
        return 1;                                         // 返回：0-超时没有返回、1-正常返回期待值
    }
    zjs_delay_ms(1); // 延时; 用于超时退出处理，避免死等
  }
  return 0; // 返回：0-超时、返回异常，1-正常返回期待值
}
#endif

/******************************************************************************
 * 函  数： UART1_GetRxNum
 * 功  能： 获取最新一帧数据的字节数
 * 参  数： 无
 * 返回值： 0=没有接收到数据，非0=新一帧数据的字节数
 ******************************************************************************/
uint16_t UART1_GetRxNum(void)
{
  return xUART1.usRxNum;
}

/******************************************************************************
 * 函  数： UART1_GetRxData
 * 功  能： 获取最新一帧数据 (数据的地址）
 * 参  数： 无
 * 返回值： 缓存地址(uint8_t*)
 ******************************************************************************/
uint8_t *UART1_GetRxData(void)
{
  return xUART1.puRxData;
}

/******************************************************************************
 * 函  数： UART1_ClearRx
 * 功  能： 清理最后一帧数据的缓存
 *          主要是清0字节数，因为它是用来判断接收的标准
 * 参  数： 无
 * 返回值： 无
 ******************************************************************************/
void UART1_ClearRx(void)
{
  xUART1.usRxNum = 0;
}

/******************************************************************************
 * 函  数： UART1_Register_IDLE_callback
 * 功  能： 注册串口空闲中断函数指针
 * 参  数： 回调函数指针
 * 返回值： 无
 ******************************************************************************/
void UART1_Register_IDLE_callback(uart_callback p)
{
  xUART1.pIDLEcallback = p;
}
#endif // endif UART1_EN

//////////////////////////////////////////////////////////////   UART-2   ///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if UART2_EN

static xUSATR_TypeDef xUART2 = {0};                  // 定义 UART2 的收发结构体
static uint8_t uaUART2RxData[UART2_RX_BUF_SIZE];     // 定义 UART2 的接收缓存
static uint8_t uaUART2TxFiFoData[UART2_TX_BUF_SIZE]; // 定义 UART2 的发送缓存

/******************************************************************************
 * 函  数： UART2_Init
 * 功  能： 初始化USART2的通信引脚、协议参数、中断优先级
 *          协议：波特率-None-8-1
 *          发送：发送中断
 *          接收：接收+空闲中断
 * 返回值： 无
 ******************************************************************************/
void UART2_Init(void)
{
  // 中断优选级配置
  HAL_NVIC_SetPriority(USART2_IRQn, 1, 1); // 配置中断线的优先级
  HAL_NVIC_EnableIRQ(USART2_IRQn);         // 使能中断线
  // 串口中断设置
  USART2->CR1 &= ~(0x01 << 7); // 关闭发送中断
  USART2->CR1 |= 0x01 << 5;    // 使能接收中断: 接收缓冲区非空
  USART2->CR1 |= 0x01 << 4;    // 使能空闲中断：超过1字节时间没收到新数据
  USART2->SR = ~(0x00F0);      // 清理中断
  // 开启USART2
  USART2->CR1 |= 0x01 << 13; // 使能UART开始工作
  // 关联缓冲区
  xUART2.puRxData = uaUART2RxData;         // 获取接收缓冲区的地址
  xUART2.puTxFiFoData = uaUART2TxFiFoData; // 获取发送缓冲区的地址
}

/******************************************************************************
 * 函  数： USART2_IRQHandler
 * 功  能： USART2的接收中断、空闲中断、发送中断
 * 参  数： 无
 * 返回值： 无
 * 备  注： 本函数，当产生中断事件时，由硬件调用。
 *          如果使用本文件代码，在工程文件的其它地方，要注释同名函数，否则冲突。
 ******************************************************************************/
void USART2_IRQHandler(void)
{
  static uint16_t cnt = 0;                  // 接收字节数累计：每一帧数据已接收到的字节数
  static uint8_t rxTemp[UART2_RX_BUF_SIZE]; // 接收数据缓存数组：每新接收１个字节，先顺序存放到这里，当一帧接收完(发生空闲中断), 再转存到全局变量：xUARTx.puRxData[xx]中；

  // 发送中断：用于把环形缓冲的数据，逐字节发出
  if ((USART2->SR & 1 << 7) && (USART2->CR1 & 1 << 7)) // 检查TXE(发送数据寄存器空)、TXEIE(发送缓冲区空中断使能)
  {
    USART2->DR = xUART2.puTxFiFoData[xUART2.usTxFiFoTail++]; // 把要发送的字节，放入USART的发送寄存器
    if (xUART2.usTxFiFoTail == UART2_TX_BUF_SIZE)            // 如果数据指针到了尾部，就重新标记到0
      xUART2.usTxFiFoTail = 0;
    if (xUART2.usTxFiFoTail == xUART2.usTxFiFoData)
      USART2->CR1 &= ~(1 << 7); // 已发送完成，关闭发送缓冲区空置中断 TXEIE
    return;
  }

  // 接收中断：用于逐个字节接收，存放到临时缓存
  if (USART2->SR & (1 << 5)) // 检查RXNE(读数据寄存器非空标志位); RXNE中断清理方法：读DR时自动清理；
  {
    if ((cnt >= UART2_RX_BUF_SIZE)) //||xUART2.ReceivedFlag==1   // 判断1: 当前帧已接收到的数据量，已满(缓存区), 为避免溢出，本包后面接收到的数据直接舍弃．
    {
      // 判断2: 如果之前接收好的数据包还没处理，就放弃新数据，即，新数据帧不能覆盖旧数据帧，直至旧数据帧被处理．缺点：数据传输过快于处理速度时会掉包；好处：机制清晰，易于调试
#if ZZY_USE_ERROR_PRINTF
      printf("警告：UART2单帧接收量，已超出接收缓存大小\n!");
#endif
      USART2->DR; // 读取数据寄存器的数据，但不保存．主要作用：读DR时自动清理接收中断标志；
      return;
    }
    rxTemp[cnt++] = USART2->DR; // 把新收到的字节数据，顺序存放到RXTemp数组中；注意：读取DR时自动清零中断位；
    return;
  }

  // 空闲中断：用于判断一帧数据结束，整理数据到外部备读
  if (USART2->SR & (1 << 4)) // 检查IDLE(空闲中断标志位); IDLE中断标志清理方法：序列清零，USART1 ->SR;  USART1 ->DR;
  {
    xUART2.usRxNum = 0;                                 // 把接收到的数据字节数清0
    memcpy(xUART2.puRxData, rxTemp, UART2_RX_BUF_SIZE); // 把本帧接收到的数据，存入到结构体的数组成员xUARTx.puRxData中, 等待处理; 注意：复制的是整个数组，包括0值，以方便字符串输出时尾部以0作字符串结束符
    xUART2.usRxNum = cnt;                               // 把接收到的字节数，存入到结构体变量xUARTx.usRxNum中；
    cnt = 0;                                            // 接收字节数累计器，清零; 准备下一次的接收
    memset(rxTemp, 0, UART2_RX_BUF_SIZE);               // 接收数据缓存数组，清零; 准备下一次的接收
    if (xUART2.pIDLEcallback != NULL)
    {                                                   // 判断回调函数不为空
      xUART2.pIDLEcallback(UART2_GetRxData(), UART2_GetRxNum()); // 执行空闲中断回调函数
    }
    USART2->SR;
    USART2->DR; // 清零IDLE中断标志位!! 序列清零，顺序不能错!!
    return;
  }

  return;
}

/******************************************************************************
 * 函  数： UART2_SendData
 * 功  能： UART通过中断发送数据
 *         【适合场景】本函数可发送各种数据，而不限于字符串，如int,char
 *         【不 适 合】注意h文件中所定义的发缓冲区大小、注意数据压入缓冲区的速度与串口发送速度的冲突
 * 参  数： uint8_t* puData     需发送数据的地址
 *          uint8_t  usNum      发送的字节数 ，数量受限于h文件中设置的发送缓存区大小宏定义
 * 返回值： 无
 ******************************************************************************/
void UART2_SendData(uint8_t *puData, uint16_t usNum)
{
  for (uint16_t i = 0; i < usNum; i++) // 把数据放入环形缓冲区
  {
    xUART2.puTxFiFoData[xUART2.usTxFiFoData++] = puData[i]; // 把字节放到缓冲区最后的位置，然后指针后移
    if (xUART2.usTxFiFoData == UART2_TX_BUF_SIZE)           // 如果指针位置到达缓冲区的最大值，则归0
      xUART2.usTxFiFoData = 0;
  }

  if ((USART2->CR1 & 1 << 7) == 0) // 检查USART寄存器的发送缓冲区空置中断(TXEIE)是否已打开
    USART2->CR1 |= 1 << 7;         // 打开TXEIE中断
}

/******************************************************************************
 * 函  数： UART2_SendString
 * 功  能： 发送字符串
 *          用法请参考printf，及示例中的展示
 *          注意，最大发送字节数为512-1个字符，可在函数中修改上限
 * 参  数： const char *pcString, ...   (如同printf的用法)
 * 返回值： 无
 ******************************************************************************/
void UART2_SendString(const char *pcString, ...)
{
  char mBuffer[512] = {0};
  ;                                                    // 开辟一个缓存, 并把里面的数据置0
  va_list ap;                                          // 新建一个可变参数列表
  va_start(ap, pcString);                              // 列表指向第一个可变参数
  vsnprintf(mBuffer, 512, pcString, ap);               // 把所有参数，按格式，输出到缓存; 参数2用于限制发送的最大字节数，如果达到上限，则只发送上限值-1; 最后1字节自动置'\0'
  va_end(ap);                                          // 清空可变参数列表
  UART2_SendData((uint8_t *)mBuffer, strlen(mBuffer)); // 把字节存放环形缓冲，排队准备发送
}

/******************************************************************************
 * 函    数： UART2_SendAT
 * 功    能： 发送AT命令, 并等待返回信息
 * 参    数： char     *pcString      AT指令字符串
 *            char     *pcAckString   期待的指令返回信息字符串
 *            uint16_t  usTimeOut     发送命令后等待的时间，毫秒
 *
 * 返 回 值： 0-执行失败、1-执行正常
 ******************************************************************************/
#if ZZY_USE_AT_FUNCS
uint8_t UART2_SendAT(char *pcAT, char *pcAckString, uint16_t usTimeOutMs)
{
  UART2_ClearRx();        // 清0
  UART2_SendString(pcAT); // 发送AT指令字符串

  while (usTimeOutMs--) // 判断是否起时(这里只作简单的循环判断次数处理）
  {
    if (UART2_GetRxNum()) // 判断是否接收到数据
    {
      UART2_ClearRx();                                    // 清0接收字节数; 注意：接收到的数据内容 ，是没有被清0的
      if (strstr((char *)UART2_GetRxData(), pcAckString)) // 判断返回数据中是否有期待的字符
        return 1;                                         // 返回：0-超时没有返回、1-正常返回期待值
    }
    zjs_delay_ms(1); // 延时; 用于超时退出处理，避免死等
  }
  return 0; // 返回：0-超时、返回异常，1-正常返回期待值
}
#endif

/******************************************************************************
 * 函  数： UART2_GetRxNum
 * 功  能： 获取最新一帧数据的字节数
 * 参  数： 无
 * 返回值： 0=没有接收到数据，非0=新一帧数据的字节数
 ******************************************************************************/
uint16_t UART2_GetRxNum(void)
{
  return xUART2.usRxNum;
}

/******************************************************************************
 * 函  数： UART2_GetRxData
 * 功  能： 获取最新一帧数据 (数据的地址）
 * 参  数： 无
 * 返回值： 数据的地址(uint8_t*)
 ******************************************************************************/
uint8_t *UART2_GetRxData(void)
{
  return xUART2.puRxData;
}

/******************************************************************************
 * 函  数： UART2_ClearRx
 * 功  能： 清理最后一帧数据的缓存
 *          主要是清0字节数，因为它是用来判断接收的标准
 * 参  数： 无
 * 返回值： 无
 ******************************************************************************/
void UART2_ClearRx(void)
{
  xUART2.usRxNum = 0;
}

/******************************************************************************
 * 函  数： UART2_Register_IDLE_callback
 * 功  能： 注册串口空闲中断函数指针
 * 参  数： 回调函数指针
 * 返回值： 无
 ******************************************************************************/
void UART2_Register_IDLE_callback(uart_callback p)
{
  xUART2.pIDLEcallback = p;
}
#endif // endif UART2_EN

//////////////////////////////////////////////////////////////   USART-3   //////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if UART3_EN

static xUSATR_TypeDef xUART3 = {0};                  // 定义 UART3 的收发结构体
static uint8_t uaUart3RxData[UART3_RX_BUF_SIZE];     // 定义 UART3 的接收缓存
static uint8_t uaUart3TxFiFoData[UART3_TX_BUF_SIZE]; // 定义 UART3 的发送缓存

/******************************************************************************
 * 函  数： UART3_Init
 * 功  能： 初始化USART3的通信引脚、协议参数、中断优先级
 *          协议：波特率-None-8-1
 *          发送：发送中断
 *          接收：接收+空闲中断
 * 返回值： 无
 ******************************************************************************/
void UART3_Init(void)
{
  // 中断优选级配置
  HAL_NVIC_SetPriority(USART3_IRQn, 1, 1); // 配置中断线的优先级
  HAL_NVIC_EnableIRQ(USART3_IRQn);         // 使能中断线
  // 串口中断设置
  USART3->CR1 &= ~(0x01 << 7); // 关闭发送中断
  USART3->CR1 |= 0x01 << 5;    // 使能接收中断: 接收缓冲区非空
  USART3->CR1 |= 0x01 << 4;    // 使能空闲中断：超过1字节时间没收到新数据
  USART3->SR = ~(0x00F0);      // 清理中断
  // 打开串口
  USART3->CR1 |= 0x01 << 13; // 使能UART开始工作
  // 关联缓冲区
  xUART3.puRxData = uaUart3RxData;         // 获取接收缓冲区的地址
  xUART3.puTxFiFoData = uaUart3TxFiFoData; // 获取发送缓冲区的地址
}

/******************************************************************************
 * 函  数： USART3_IRQHandler
 * 功  能： USART3的接收中断、空闲中断、发送中断
 * 参  数： 无
 * 返回值： 无
 * 备  注： 本函数，当产生中断事件时，由硬件调用。
 *          如果使用本文件代码，在工程文件的其它地方，要注释同名函数，否则冲突。
 ******************************************************************************/
void USART3_IRQHandler(void)
{
  static uint16_t cnt = 0;                  // 接收字节数累计：每一帧数据已接收到的字节数
  static uint8_t rxTemp[UART3_RX_BUF_SIZE]; // 接收数据缓存数组：每新接收１个字节，先顺序存放到这里，当一帧接收完(发生空闲中断), 再转存到全局变量：xUARTx.puRxData[xx]中；

  // 发送中断：用于把环形缓冲的数据，逐字节发出
  if ((USART3->SR & 1 << 7) && (USART3->CR1 & 1 << 7)) // 检查TXE(发送数据寄存器空)、TXEIE(发送缓冲区空中断使能)
  {
    USART3->DR = xUART3.puTxFiFoData[xUART3.usTxFiFoTail++]; // 把要发送的字节，放入USART的发送寄存器
    if (xUART3.usTxFiFoTail == UART3_TX_BUF_SIZE)            // 如果数据指针到了尾部，就重新标记到0
      xUART3.usTxFiFoTail = 0;
    if (xUART3.usTxFiFoTail == xUART3.usTxFiFoData)
      USART3->CR1 &= ~(1 << 7); // 已发送完成，关闭发送缓冲区空置中断 TXEIE
    return;
  }

  // 接收中断：用于逐个字节接收，存放到临时缓存
  if (USART3->SR & (1 << 5)) // 检查RXNE(读数据寄存器非空标志位); RXNE中断清理方法：读DR时自动清理；
  {
    if ((cnt >= UART3_RX_BUF_SIZE)) //||xUART3.ReceivedFlag==1   // 判断1: 当前帧已接收到的数据量，已满(缓存区), 为避免溢出，本包后面接收到的数据直接舍弃．
    {
      // 判断2: 如果之前接收好的数据包还没处理，就放弃新数据，即，新数据帧不能覆盖旧数据帧，直至旧数据帧被处理．缺点：数据传输过快于处理速度时会掉包；好处：机制清晰，易于调试
#if ZZY_USE_ERROR_PRINTF
      printf("警告：UART3单帧接收量，已超出接收缓存大小\n!");
#endif
      USART3->DR; // 读取数据寄存器的数据，但不保存．主要作用：读DR时自动清理接收中断标志；
      return;
    }
    rxTemp[cnt++] = USART3->DR; // 把新收到的字节数据，顺序存放到RXTemp数组中；注意：读取DR时自动清零中断位
    return;
  }

  // 空闲中断：用于判断一帧数据结束，整理数据到外部备读
  if (USART3->SR & (1 << 4)) // 检查IDLE(空闲中断标志位); IDLE中断标志清理方法：序列清零，USART1 ->SR;  USART1 ->DR;
  {
    xUART3.usRxNum = 0;                                 // 把接收到的数据字节数清0
    memcpy(xUART3.puRxData, rxTemp, UART3_RX_BUF_SIZE); // 把本帧接收到的数据，存入到结构体的数组成员xUARTx.puRxData中, 等待处理; 注意：复制的是整个数组，包括0值，以方便字符串输出时尾部以0作字符串结束符
    xUART3.usRxNum = cnt;                               // 把接收到的字节数，存入到结构体变量xUARTx.usRxNum中；
    cnt = 0;                                            // 接收字节数累计器，清零; 准备下一次的接收
    memset(rxTemp, 0, UART3_RX_BUF_SIZE);               // 接收数据缓存数组，清零; 准备下一次的接收
    if (xUART3.pIDLEcallback != NULL)
    {                                                   // 判断回调函数不为空
      xUART3.pIDLEcallback(UART3_GetRxData(), UART3_GetRxNum()); // 执行空闲中断回调函数
    }
    USART3->SR;
    USART3->DR; // 清零IDLE中断标志位!! 序列清零，顺序不能错!!
    return;
  }

  return;
}

/******************************************************************************
 * 函  数： UART3_SendData
 * 功  能： UART通过中断发送数据
 *         【适合场景】本函数可发送各种数据，而不限于字符串，如int,char
 *         【不 适 合】注意h文件中所定义的发缓冲区大小、注意数据压入缓冲区的速度与串口发送速度的冲突
 * 参  数： uint8_t* puData   需发送数据的地址
 *          uint8_t  usNum      发送的字节数 ，数量受限于h文件中设置的发送缓存区大小宏定义
 * 返回值： 无
 ******************************************************************************/
void UART3_SendData(uint8_t *puData, uint16_t usNum)
{
  for (uint16_t i = 0; i < usNum; i++) // 把数据放入环形缓冲区
  {
    xUART3.puTxFiFoData[xUART3.usTxFiFoData++] = puData[i]; // 把字节放到缓冲区最后的位置，然后指针后移
    if (xUART3.usTxFiFoData == UART3_TX_BUF_SIZE)           // 如果指针位置到达缓冲区的最大值，则归0
      xUART3.usTxFiFoData = 0;
  }

  if ((USART3->CR1 & 1 << 7) == 0) // 检查USART寄存器的发送缓冲区空置中断(TXEIE)是否已打开
    USART3->CR1 |= 1 << 7;         // 打开TXEIE中断
}

/******************************************************************************
 * 函  数： UART3_SendString
 * 功  能： 发送字符串
 *          用法请参考printf，及示例中的展示
 *          注意，最大发送字节数为512-1个字符，可在函数中修改上限
 * 参  数： const char *pcString, ...   (如同printf的用法)
 * 返回值： 无
 ******************************************************************************/
void UART3_SendString(const char *pcString, ...)
{
  char mBuffer[512] = {0};
  ;                                                    // 开辟一个缓存, 并把里面的数据置0
  va_list ap;                                          // 新建一个可变参数列表
  va_start(ap, pcString);                              // 列表指向第一个可变参数
  vsnprintf(mBuffer, 512, pcString, ap);               // 把所有参数，按格式，输出到缓存; 参数2用于限制发送的最大字节数，如果达到上限，则只发送上限值-1; 最后1字节自动置'\0'
  va_end(ap);                                          // 清空可变参数列表
  UART3_SendData((uint8_t *)mBuffer, strlen(mBuffer)); // 把字节存放环形缓冲，排队准备发送
}

/******************************************************************************
 * 函    数： UART3_SendAT
 * 功    能： 发送AT命令, 并等待返回信息
 * 参    数： char     *pcString      AT指令字符串
 *            char     *pcAckString   期待的指令返回信息字符串
 *            uint16_t  usTimeOut     发送命令后等待的时间，毫秒
 *
 * 返 回 值： 0-执行失败、1-执行正常
 ******************************************************************************/
#if ZZY_USE_AT_FUNCS
uint8_t UART3_SendAT(char *pcAT, char *pcAckString, uint16_t usTimeOutMs)
{
  UART3_ClearRx();        // 清0
  UART3_SendString(pcAT); // 发送AT指令字符串

  while (usTimeOutMs--) // 判断是否起时(这里只作简单的循环判断次数处理）
  {
    if (UART3_GetRxNum()) // 判断是否接收到数据
    {
      UART3_ClearRx();                                    // 清0接收字节数; 注意：接收到的数据内容 ，是没有被清0的
      if (strstr((char *)UART3_GetRxData(), pcAckString)) // 判断返回数据中是否有期待的字符
        return 1;                                         // 返回：0-超时没有返回、1-正常返回期待值
    }
    zjs_delay_ms(1); // 延时; 用于超时退出处理，避免死等
  }
  return 0; // 返回：0-超时、返回异常，1-正常返回期待值
}
#endif

/******************************************************************************
 * 函  数： UART3_GetRxNum
 * 功  能： 获取最新一帧数据的字节数
 * 参  数： 无
 * 返回值： 0=没有接收到数据，非0=新一帧数据的字节数
 ******************************************************************************/
uint16_t UART3_GetRxNum(void)
{
  return xUART3.usRxNum;
}

/******************************************************************************
 * 函  数： UART3_GetRxData
 * 功  能： 获取最新一帧数据 (数据的地址）
 * 参  数： 无
 * 返回值： 数据的地址(uint8_t*)
 ******************************************************************************/
uint8_t *UART3_GetRxData(void)
{
  return xUART3.puRxData;
}

/******************************************************************************
 * 函  数： UART3_ClearRx
 * 功  能： 清理最后一帧数据的缓存
 *          主要是清0字节数，因为它是用来判断接收的标准
 * 参  数： 无
 * 返回值： 无
 ******************************************************************************/
void UART3_ClearRx(void)
{
  xUART3.usRxNum = 0;
}

/******************************************************************************
 * 函  数： UART3_Register_IDLE_callback
 * 功  能： 注册串口空闲中断函数指针
 * 参  数： 回调函数指针
 * 返回值： 无
 ******************************************************************************/
void UART3_Register_IDLE_callback(uart_callback p)
{
  xUART3.pIDLEcallback = p;
}
#endif // endif UART3_EN

//////////////////////////////////////////////////////////////   UART-4   //////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if UART4_EN

static xUSATR_TypeDef xUART4 = {0};                  // 定义 UART4 的收发结构体
static uint8_t uaUart4RxData[UART4_RX_BUF_SIZE];     // 定义 UART4 的接收缓存
static uint8_t uaUart4TxFiFoData[UART4_TX_BUF_SIZE]; // 定义 UART4 的发送缓存

/******************************************************************************
 * 函  数： UART4_Init
 * 功  能： 初始化UART4的通信引脚、协议参数、中断优先级
 *          协议：波特率-None-8-1
 *          发送：发送中断
 *          接收：接收+空闲中断
 * 返回值： 无
 ******************************************************************************/
void UART4_Init(void)
{
  // 中断优选级配置
  HAL_NVIC_SetPriority(UART4_IRQn, 1, 1); // 配置中断线的优先级
  HAL_NVIC_EnableIRQ(UART4_IRQn);         // 使能中断线
  // 串口中断设置
  UART4->CR1 &= ~(0x01 << 7); // 关闭发送中断
  UART4->CR1 |= 0x01 << 5;    // 使能接收中断: 接收缓冲区非空
  UART4->CR1 |= 0x01 << 4;    // 使能空闲中断：超过1字节时间没收到新数据
  UART4->SR = ~(0x00F0);      // 清理中断
  // 打开串口
  UART4->CR1 |= 0x01 << 13; // 使能UART开始工作
  // 关联缓冲区
  xUART4.puRxData = uaUart4RxData;         // 获取接收缓冲区的地址
  xUART4.puTxFiFoData = uaUart4TxFiFoData; // 获取发送缓冲区的地址
}

/******************************************************************************
 * 函  数： UART4_IRQHandler
 * 功  能： UART4的中断处理函数
 *          接收中断、空闲中断、发送中断
 * 参  数： 无
 * 返回值： 无
 * 备  注： 本函数，当产生中断事件时，由硬件调用。
 *          如果使用本文件代码，在工程文件的其它地方，要注释同名函数，否则冲突。
 ******************************************************************************/
void UART4_IRQHandler(void)
{
  static uint16_t cnt = 0;                  // 接收字节数累计：每一帧数据已接收到的字节数
  static uint8_t rxTemp[UART4_RX_BUF_SIZE]; // 接收数据缓存数组：每新接收１个字节，先顺序存放到这里，当一帧接收完(发生空闲中断), 再转存到全局变量：xUARTx.puRxData[xx]中；

  // 发送中断：用于把环形缓冲的数据，逐字节发出
  if ((UART4->SR & 1 << 7) && (UART4->CR1 & 1 << 7)) // 检查TXE(发送数据寄存器空)、TXEIE(发送缓冲区空中断使能)
  {
    UART4->DR = xUART4.puTxFiFoData[xUART4.usTxFiFoTail++]; // 把要发送的字节，放入USART的发送寄存器
    if (xUART4.usTxFiFoTail == UART4_TX_BUF_SIZE)           // 如果数据指针到了尾部，就重新标记到0
      xUART4.usTxFiFoTail = 0;
    if (xUART4.usTxFiFoTail == xUART4.usTxFiFoData)
      UART4->CR1 &= ~(1 << 7); // 已发送完成，关闭发送缓冲区空置中断 TXEIE
    return;
  }

  // 接收中断：用于逐个字节接收，存放到临时缓存
  if (UART4->SR & (1 << 5)) // 检查RXNE(读数据寄存器非空标志位); RXNE中断清理方法：读DR时自动清理；
  {
    if ((cnt >= UART4_RX_BUF_SIZE)) //||xUART4.ReceivedFlag==1   // 判断1: 当前帧已接收到的数据量，已满(缓存区), 为避免溢出，本包后面接收到的数据直接舍弃．
    {
      // 判断2: 如果之前接收好的数据包还没处理，就放弃新数据，即，新数据帧不能覆盖旧数据帧，直至旧数据帧被处理．缺点：数据传输过快于处理速度时会掉包；好处：机制清晰，易于调试
#if ZZY_USE_ERROR_PRINTF
      printf("警告：UART4单帧接收量，已超出接收缓存大小\n!");
#endif
      UART4->DR; // 读取数据寄存器的数据，但不保存．主要作用：读DR时自动清理接收中断标志；
      return;
    }
    rxTemp[cnt++] = UART4->DR; // 把新收到的字节数据，顺序存放到RXTemp数组中；注意：读取DR时自动清零中断位
    return;
  }

  // 空闲中断：用于判断一帧数据结束，整理数据到外部备读
  if (UART4->SR & (1 << 4)) // 检查IDLE(空闲中断标志位); IDLE中断标志清理方法：序列清零，USART1 ->SR;  USART1 ->DR;
  {
    xUART4.usRxNum = 0;                                 // 把接收到的数据字节数清0
    memcpy(xUART4.puRxData, rxTemp, UART4_RX_BUF_SIZE); // 把本帧接收到的数据，存入到结构体的数组成员xUARTx.puRxData中, 等待处理; 注意：复制的是整个数组，包括0值，以方便字符串输出时尾部以0作字符串结束符
    xUART4.usRxNum = cnt;                               // 把接收到的字节数，存入到结构体变量xUARTx.usRxNum中；
    cnt = 0;                                            // 接收字节数累计器，清零; 准备下一次的接收
    memset(rxTemp, 0, UART4_RX_BUF_SIZE);               // 接收数据缓存数组，清零; 准备下一次的接收
    if (xUART4.pIDLEcallback != NULL)
    {                                                   // 判断回调函数不为空
      xUART4.pIDLEcallback(UART4_GetRxData(), UART4_GetRxNum()); // 执行空闲中断回调函数
    }
    UART4->SR;
    UART4->DR; // 清零IDLE中断标志位!! 序列清零，顺序不能错!!
    return;
  }

  return;
}

/******************************************************************************
 * 函  数： UART4_SendData
 * 功  能： UART通过中断发送数据
 *         【适合场景】本函数可发送各种数据，而不限于字符串，如int,char
 *         【不 适 合】注意h文件中所定义的发缓冲区大小、注意数据压入缓冲区的速度与串口发送速度的冲突
 * 参  数： uint8_t* puData   需发送数据的地址
 *          uint8_t  usNum    发送的字节数 ，数量受限于h文件中设置的发送缓存区大小宏定义
 * 返回值： 无
 ******************************************************************************/
void UART4_SendData(uint8_t *puData, uint16_t usNum)
{
  for (uint16_t i = 0; i < usNum; i++) // 把数据放入环形缓冲区
  {
    xUART4.puTxFiFoData[xUART4.usTxFiFoData++] = puData[i]; // 把字节放到缓冲区最后的位置，然后指针后移
    if (xUART4.usTxFiFoData == UART4_TX_BUF_SIZE)           // 如果指针位置到达缓冲区的最大值，则归0
      xUART4.usTxFiFoData = 0;
  }

  if ((UART4->CR1 & 1 << 7) == 0) // 检查USART寄存器的发送缓冲区空置中断(TXEIE)是否已打开
    UART4->CR1 |= 1 << 7;         // 打开TXEIE中断
}

/******************************************************************************
 * 函  数： UART4_SendString
 * 功  能： 发送字符串
 *          用法请参考printf，及示例中的展示
 *          注意，最大发送字节数为512-1个字符，可在函数中修改上限
 * 参  数： const char *pcString, ...   (如同printf的用法)
 * 返回值： 无
 ******************************************************************************/
void UART4_SendString(const char *pcString, ...)
{
  char mBuffer[512] = {0};
  ;                                                    // 开辟一个缓存, 并把里面的数据置0
  va_list ap;                                          // 新建一个可变参数列表
  va_start(ap, pcString);                              // 列表指向第一个可变参数
  vsnprintf(mBuffer, 512, pcString, ap);               // 把所有参数，按格式，输出到缓存; 参数2用于限制发送的最大字节数，如果达到上限，则只发送上限值-1; 最后1字节自动置'\0'
  va_end(ap);                                          // 清空可变参数列表
  UART4_SendData((uint8_t *)mBuffer, strlen(mBuffer)); // 把字节存放环形缓冲，排队准备发送
}

/******************************************************************************
 * 函    数： UART4_SendAT
 * 功    能： 发送AT命令, 并等待返回信息
 * 参    数： char     *pcString      AT指令字符串
 *            char     *pcAckString   期待的指令返回信息字符串
 *            uint16_t  usTimeOut     发送命令后等待的时间，毫秒
 *
 * 返 回 值： 0-执行失败、1-执行正常
 ******************************************************************************/
#if ZZY_USE_AT_FUNCS
uint8_t UART4_SendAT(char *pcAT, char *pcAckString, uint16_t usTimeOutMs)
{
  UART4_ClearRx();        // 清0
  UART4_SendString(pcAT); // 发送AT指令字符串

  while (usTimeOutMs--) // 判断是否起时(这里只作简单的循环判断次数处理）
  {
    if (UART4_GetRxNum()) // 判断是否接收到数据
    {
      UART4_ClearRx();                                    // 清0接收字节数; 注意：接收到的数据内容 ，是没有被清0的
      if (strstr((char *)UART4_GetRxData(), pcAckString)) // 判断返回数据中是否有期待的字符
        return 1;                                         // 返回：0-超时没有返回、1-正常返回期待值
    }
    zjs_delay_ms(1); // 延时; 用于超时退出处理，避免死等
  }
  return 0; // 返回：0-超时、返回异常，1-正常返回期待值
}
#endif

/******************************************************************************
 * 函  数： UART4_GetRxNum
 * 功  能： 获取最新一帧数据的字节数
 * 参  数： 无
 * 返回值： 0=没有接收到数据，非0=新一帧数据的字节数
 ******************************************************************************/
uint16_t UART4_GetRxNum(void)
{
  return xUART4.usRxNum;
}

/******************************************************************************
 * 函  数： UART4_GetRxData
 * 功  能： 获取最新一帧数据 (数据的地址）
 * 参  数： 无
 * 返回值： 数据的地址(uint8_t*)
 ******************************************************************************/
uint8_t *UART4_GetRxData(void)
{
  return xUART4.puRxData;
}

/******************************************************************************
 * 函  数： UART4_ClearRx
 * 功  能： 清理最后一帧数据的缓存
 *          主要是清0字节数，因为它是用来判断接收的标准
 * 参  数： 无
 * 返回值： 无
 ******************************************************************************/
void UART4_ClearRx(void)
{
  xUART4.usRxNum = 0;
}

/******************************************************************************
 * 函  数： UART4_Register_IDLE_callback
 * 功  能： 注册串口空闲中断函数指针
 * 参  数： 回调函数指针
 * 返回值： 无
 ******************************************************************************/
void UART4_Register_IDLE_callback(uart_callback p)
{
  xUART4.pIDLEcallback = p;
}
#endif // endif UART4_EN

//////////////////////////////////////////////////////////////   UART-5   //////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if UART5_EN

static xUSATR_TypeDef xUART5 = {0};                  // 定义 UART5 的收发结构体
static uint8_t uaUart5RxData[UART5_RX_BUF_SIZE];     // 定义 UART5 的接收缓存
static uint8_t uaUart5TxFiFoData[UART5_TX_BUF_SIZE]; // 定义 UART5 的发送缓存

/******************************************************************************
 * 函  数： UART5_Init
 * 功  能： 初始化UART5的通信引脚、协议参数、中断优先级
 *          协议：波特率-None-8-1
 *          发送：发送中断
 *          接收：接收+空闲中断
 * 返回值： 无
 ******************************************************************************/
void UART5_Init(void)
{
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct); // 初始化引脚工作模式
  // 中断优选级配置
  HAL_NVIC_SetPriority(UART5_IRQn, 0, 0); // 配置中断线的优先级
  HAL_NVIC_EnableIRQ(UART5_IRQn);         // 使能中断线
  // 串口中断设置
  UART5->CR1 &= ~(0x01 << 7); // 关闭发送中断
  UART5->CR1 |= 0x01 << 5;    // 使能接收中断: 接收缓冲区非空
  UART5->CR1 |= 0x01 << 4;    // 使能空闲中断：超过1字节时间没收到新数据
  UART5->SR = ~(0x00F0);      // 清理中断
  // 打开串口
  UART5->CR1 |= 0x01 << 13; // 使能UART开始工作
  // 关联缓冲区
  xUART5.puRxData = uaUart5RxData;         // 获取接收缓冲区的地址
  xUART5.puTxFiFoData = uaUart5TxFiFoData; // 获取发送缓冲区的地址
}

/******************************************************************************
 * 函  数： UART5_IRQHandler
 * 功  能： UART5的接收中断、空闲中断、发送中断
 * 参  数： 无
 * 返回值： 无
 * 备  注： 本函数，当产生中断事件时，由硬件调用。
 *          如果使用本文件代码，在工程文件的其它地方，要注释同名函数，否则冲突。
 ******************************************************************************/
void UART5_IRQHandler(void)
{
  static uint16_t cnt = 0;                  // 接收字节数累计：每一帧数据已接收到的字节数
  static uint8_t rxTemp[UART5_RX_BUF_SIZE]; // 接收数据缓存数组：每新接收１个字节，先顺序存放到这里，当一帧接收完(发生空闲中断), 再转存到全局变量：xUARTx.puRxData[xx]中；

  // 发送中断：用于把环形缓冲的数据，逐字节发出
  if ((UART5->SR & 1 << 7) && (UART5->CR1 & 1 << 7)) // 检查TXE(发送数据寄存器空)、TXEIE(发送缓冲区空中断使能)
  {
    UART5->DR = xUART5.puTxFiFoData[xUART5.usTxFiFoTail++]; // 把要发送的字节，放入USART的发送寄存器
    if (xUART5.usTxFiFoTail == UART5_TX_BUF_SIZE)           // 如果数据指针到了尾部，就重新标记到0
      xUART5.usTxFiFoTail = 0;
    if (xUART5.usTxFiFoTail == xUART5.usTxFiFoData)
      UART5->CR1 &= ~(1 << 7); // 已发送完成，关闭发送缓冲区空置中断 TXEIE
    return;
  }

  // 接收中断：用于逐个字节接收，存放到临时缓存
  if (UART5->SR & (1 << 5)) // 检查RXNE(读数据寄存器非空标志位); RXNE中断清理方法：读DR时自动清理；
  {
    if ((cnt >= UART5_RX_BUF_SIZE)) //||xUART5.ReceivedFlag==1   // 判断1: 当前帧已接收到的数据量，已满(缓存区), 为避免溢出，本包后面接收到的数据直接舍弃．
    {
      // 判断2: 如果之前接收好的数据包还没处理，就放弃新数据，即，新数据帧不能覆盖旧数据帧，直至旧数据帧被处理．缺点：数据传输过快于处理速度时会掉包；好处：机制清晰，易于调试
#if ZZY_USE_ERROR_PRINTF
      printf("警告：UART5单帧接收量，已超出接收缓存大小\n!");
#endif
      UART5->DR; // 读取数据寄存器的数据，但不保存．主要作用：读DR时自动清理接收中断标志；
      return;
    }
    rxTemp[cnt++] = UART5->DR; // 把新收到的字节数据，顺序存放到RXTemp数组中；注意：读取DR时自动清零中断位
    return;
  }

  // 空闲中断：用于判断一帧数据结束，整理数据到外部备读
  if (UART5->SR & (1 << 4)) // 检查IDLE(空闲中断标志位); IDLE中断标志清理方法：序列清零，USART1 ->SR;  USART1 ->DR;
  {
    xUART5.usRxNum = 0;                                 // 把接收到的数据字节数清0
    memcpy(xUART5.puRxData, rxTemp, UART5_RX_BUF_SIZE); // 把本帧接收到的数据，存入到结构体的数组成员xUARTx.puRxData中, 等待处理; 注意：复制的是整个数组，包括0值，以方便字符串输出时尾部以0作字符串结束符
    xUART5.usRxNum = cnt;                               // 把接收到的字节数，存入到结构体变量xUARTx.usRxNum中；
    cnt = 0;                                            // 接收字节数累计器，清零; 准备下一次的接收
    memset(rxTemp, 0, UART5_RX_BUF_SIZE);               // 接收数据缓存数组，清零; 准备下一次的接收
    if (xUART5.pIDLEcallback != NULL)
    {                                                   // 判断回调函数不为空
      xUART5.pIDLEcallback(UART5_GetRxData(), UART5_GetRxNum()); // 执行空闲中断回调函数
    }
    UART5->SR;
    UART5->DR; // 清零IDLE中断标志位!! 序列清零，顺序不能错!!
    return;
  }

  return;
}

/******************************************************************************
 * 函  数： UART5_SendData
 * 功  能： UART通过中断发送数据
 *         【适合场景】本函数可发送各种数据，而不限于字符串，如int,char
 *         【不 适 合】注意h文件中所定义的发缓冲区大小、注意数据压入缓冲区的速度与串口发送速度的冲突
 * 参  数： uint8_t* pudata     需发送数据的地址
 *          uint8_t  usNum      发送的字节数 ，数量受限于h文件中设置的发送缓存区大小宏定义
 * 返回值： 无
 ******************************************************************************/
void UART5_SendData(uint8_t *pudata, uint16_t usNum)
{
  for (uint16_t i = 0; i < usNum; i++) // 把数据放入环形缓冲区
  {
    xUART5.puTxFiFoData[xUART5.usTxFiFoData++] = pudata[i]; // 把字节放到缓冲区最后的位置，然后指针后移
    if (xUART5.usTxFiFoData == UART5_TX_BUF_SIZE)           // 如果指针位置到达缓冲区的最大值，则归0
      xUART5.usTxFiFoData = 0;
  }

  if ((UART5->CR1 & 1 << 7) == 0) // 检查USART寄存器的发送缓冲区空置中断(TXEIE)是否已打开
    UART5->CR1 |= 1 << 7;         // 打开TXEIE中断
}

/******************************************************************************
 * 函  数： UART5_SendString
 * 功  能： 发送字符串
 *          用法请参考printf，及示例中的展示
 *          注意，最大发送字节数为512-1个字符，可在函数中修改上限
 * 参  数： const char *pcString, ...   (如同printf的用法)
 * 返回值： 无
 ******************************************************************************/
void UART5_SendString(const char *pcString, ...)
{
  char mBuffer[512] = {0};
  ;                                                    // 开辟一个缓存, 并把里面的数据置0
  va_list ap;                                          // 新建一个可变参数列表
  va_start(ap, pcString);                              // 列表指向第一个可变参数
  vsnprintf(mBuffer, 512, pcString, ap);               // 把所有参数，按格式，输出到缓存; 参数2用于限制发送的最大字节数，如果达到上限，则只发送上限值-1; 最后1字节自动置'\0'
  va_end(ap);                                          // 清空可变参数列表
  UART5_SendData((uint8_t *)mBuffer, strlen(mBuffer)); // 把字节存放环形缓冲，排队准备发送
}

/******************************************************************************
 * 函    数： UART5_SendAT
 * 功    能： 发送AT命令, 并等待返回信息
 * 参    数： char     *pcString      AT指令字符串
 *            char     *pcAckString   期待的指令返回信息字符串
 *            uint16_t  usTimeOut     发送命令后等待的时间，毫秒
 *
 * 返 回 值： 0-执行失败、1-执行正常
 ******************************************************************************/
#if ZZY_USE_AT_FUNCS
uint8_t UART5_SendAT(char *pcAT, char *pcAckString, uint16_t usTimeOutMs)
{
  UART5_ClearRx();        // 清0
  UART5_SendString(pcAT); // 发送AT指令字符串

  while (usTimeOutMs--) // 判断是否起时(这里只作简单的循环判断次数处理）
  {
    if (UART5_GetRxNum()) // 判断是否接收到数据
    {
      UART5_ClearRx();                                    // 清0接收字节数; 注意：接收到的数据内容 ，是没有被清0的
      if (strstr((char *)UART5_GetRxData(), pcAckString)) // 判断返回数据中是否有期待的字符
        return 1;                                         // 返回：0-超时没有返回、1-正常返回期待值
    }
    zjs_delay_ms(1); // 延时; 用于超时退出处理，避免死等
  }
  return 0; // 返回：0-超时、返回异常，1-正常返回期待值
}
#endif

/******************************************************************************
 * 函  数： UART5_GetRxNum
 * 功  能： 获取最新一帧数据的字节数
 * 参  数： 无
 * 返回值： 0=没有接收到数据，非0=新一帧数据的字节数
 ******************************************************************************/
uint16_t UART5_GetRxNum(void)
{
  return xUART5.usRxNum;
}

/******************************************************************************
 * 函  数： UART5_GetRxData
 * 功  能： 获取最新一帧数据 (数据的地址）
 * 参  数： 无
 * 返回值： 数据的地址(uint8_t*)
 ******************************************************************************/
uint8_t *UART5_GetRxData(void)
{
  return xUART5.puRxData;
}

/******************************************************************************
 * 函  数： UART5_ClearRx
 * 功  能： 清理最后一帧数据的缓存
 *          主要是清0字节数，因为它是用来判断接收的标准
 * 参  数： 无
 * 返回值： 无
 ******************************************************************************/
void UART5_ClearRx(void)
{
  xUART5.usRxNum = 0;
}

/******************************************************************************
 * 函  数： UART5_Register_IDLE_callback
 * 功  能： 注册串口空闲中断函数指针
 * 参  数： 回调函数指针
 * 返回值： 无
 ******************************************************************************/
void UART5_Register_IDLE_callback(uart_callback p)
{
  xUART5.pIDLEcallback = p;
}
#endif // endif UART5_EN

//////////////////////////////////////////////////////////////   USART-6   //////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if UART6_EN

static xUSATR_TypeDef xUART6 = {0};                  // 定义 UART6 的收发结构体
static uint8_t uaUart6RxData[UART6_RX_BUF_SIZE];     // 定义 UART6 的接收缓存
static uint8_t uaUart6TxFiFoData[UART6_TX_BUF_SIZE]; // 定义 UART6 的发送缓存

/******************************************************************************
 * 函  数： UART6_Init
 * 功  能： 初始化USART6的通信引脚、协议参数、中断优先级
 *          协议：波特率-None-8-1
 *          发送：发送中断
 *          接收：接收+空闲中断
 * 返回值： 无
 ******************************************************************************/
void UART6_Init(void)
{
  // 中断优选级配置
  HAL_NVIC_SetPriority(USART6_IRQn, 1, 1); // 配置中断线的优先级
  HAL_NVIC_EnableIRQ(USART6_IRQn);         // 使能中断线
  // 串口中断设置
  USART6->CR1 &= ~(0x01 << 7); // 关闭发送中断
  USART6->CR1 |= 0x01 << 5;    // 使能接收中断: 接收缓冲区非空
  USART6->CR1 |= 0x01 << 4;    // 使能空闲中断：超过1字节时间没收到新数据
  USART6->SR = ~(0x00F0);      // 清理中断
  // 打开串口
  USART6->CR1 |= 0x01 << 13; // 使能UART开始工作
  // 关联缓冲区
  xUART6.puRxData = uaUart6RxData;         // 获取接收缓冲区的地址
  xUART6.puTxFiFoData = uaUart6TxFiFoData; // 获取发送缓冲区的地址
}

/******************************************************************************
 * 函  数： USART6_IRQHandler
 * 功  能： USART6的接收中断、空闲中断、发送中断
 * 参  数： 无
 * 返回值： 无
 *
******************************************************************************/
void USART6_IRQHandler(void)
{
  static uint16_t cnt = 0;                  // 接收字节数累计：每一帧数据已接收到的字节数
  static uint8_t rxTemp[UART6_RX_BUF_SIZE]; // 接收数据缓存数组：每新接收１个字节，先顺序存放到这里，当一帧接收完(发生空闲中断), 再转存到全局变量：xUARTx.puRxData[xx]中；

  // 发送中断：用于把环形缓冲的数据，逐字节发出
  if ((USART6->SR & 1 << 7) && (USART6->CR1 & 1 << 7)) // 检查TXE(发送数据寄存器空)、TXEIE(发送缓冲区空中断使能)
  {
    USART6->DR = xUART6.puTxFiFoData[xUART6.usTxFiFoTail++]; // 把要发送的字节，放入USART的发送寄存器
    if (xUART6.usTxFiFoTail == UART6_TX_BUF_SIZE)            // 如果数据指针到了尾部，就重新标记到0
      xUART6.usTxFiFoTail = 0;
    if (xUART6.usTxFiFoTail == xUART6.usTxFiFoData)
      USART6->CR1 &= ~(1 << 7); // 已发送完成，关闭发送缓冲区空置中断 TXEIE
    return;
  }

  // 接收中断：用于逐个字节接收，存放到临时缓存
  if (USART6->SR & (1 << 5)) // 检查RXNE(读数据寄存器非空标志位); RXNE中断清理方法：读DR时自动清理；
  {
    if ((cnt >= UART6_RX_BUF_SIZE)) //||(xUART1.ReceivedFlag==1   // 判断1: 当前帧已接收到的数据量，已满(缓存区), 为避免溢出，本包后面接收到的数据直接舍弃．
    {
      // 判断2: 如果之前接收好的数据包还没处理，就放弃新数据，即，新数据帧不能覆盖旧数据帧，直至旧数据帧被处理．缺点：数据传输过快于处理速度时会掉包；好处：机制清晰，易于调试
#if ZZY_USE_ERROR_PRINTF
      printf("警告：UART6单帧接收量，已超出接收缓存大小\n!");
#endif
      USART6->DR; // 读取数据寄存器的数据，但不保存．主要作用：读DR时自动清理接收中断标志；
      return;
    }
    rxTemp[cnt++] = USART6->DR; // 把新收到的字节数据，顺序存放到RXTemp数组中；注意：读取DR时自动清零中断位
    return;
  }

  // 空闲中断：用于判断一帧数据结束，整理数据到外部备读
  if (USART6->SR & (1 << 4)) // 检查IDLE(空闲中断标志位); IDLE中断标志清理方法：序列清零，USART1 ->SR;  USART1 ->DR;
  {
    xUART6.usRxNum = 0;                                 // 把接收到的数据字节数清0
    memcpy(xUART6.puRxData, rxTemp, UART6_RX_BUF_SIZE); // 把本帧接收到的数据，存入到结构体的数组成员xUARTx.puRxData中, 等待处理; 注意：复制的是整个数组，包括0值，以方便字符串输出时尾部以0作字符串结束符
    xUART6.usRxNum = cnt;                               // 把接收到的字节数，存入到结构体变量xUARTx.usRxNum中；
    cnt = 0;                                            // 接收字节数累计器，清零; 准备下一次的接收
    memset(rxTemp, 0, UART6_RX_BUF_SIZE);               // 接收数据缓存数组，清零; 准备下一次的接收
    if (xUART6.pIDLEcallback != NULL)
    {                                                   // 判断回调函数不为空
      xUART6.pIDLEcallback(UART6_GetRxData(), UART6_GetRxNum()); // 执行空闲中断回调函数
    }
    USART6->SR;
    USART6->DR; // 清零IDLE中断标志位!! 序列清零，顺序不能错!!
    return;
  }

  return;
}

/******************************************************************************
 * 函  数： UART6_SendData
 * 功  能： UART通过中断发送数据
 *         【适合场景】本函数可发送各种数据，而不限于字符串，如int,char
 *         【不 适 合】注意h文件中所定义的发缓冲区大小、注意数据压入缓冲区的速度与串口发送速度的冲突
 * 参  数： uint8_t  *puData   需发送数据的地址
 *          uint8_t   usNum    发送的字节数 ，数量受限于h文件中设置的发送缓存区大小宏定义
 * 返回值： 无
 ******************************************************************************/
void UART6_SendData(uint8_t *puData, uint16_t usNum)
{
  for (uint16_t i = 0; i < usNum; i++) // 把数据放入环形缓冲区
  {
    xUART6.puTxFiFoData[xUART6.usTxFiFoData++] = puData[i]; // 把字节放到缓冲区最后的位置，然后指针后移
    if (xUART6.usTxFiFoData == UART6_TX_BUF_SIZE)           // 如果指针位置到达缓冲区的最大值，则归0
      xUART6.usTxFiFoData = 0;
  }

  if ((USART6->CR1 & 1 << 7) == 0) // 检查USART寄存器的发送缓冲区空置中断(TXEIE)是否已打开
    USART6->CR1 |= 1 << 7;         // 打开TXEIE中断
}

/******************************************************************************
 * 函  数： UART6_SendString
 * 功  能： 发送字符串
 *          用法请参考printf，及示例中的展示
 *          注意，最大发送字节数为512-1个字符，可在函数中修改上限
 * 参  数： const char *pcString, ...   (如同printf的用法)
 * 返回值： 无
 ******************************************************************************/
void UART6_SendString(const char *pcString, ...)
{
  char mBuffer[512] = {0};
  ;                                                    // 开辟一个缓存, 并把里面的数据置0
  va_list ap;                                          // 新建一个可变参数列表
  va_start(ap, pcString);                              // 列表指向第一个可变参数
  vsnprintf(mBuffer, 512, pcString, ap);               // 把所有参数，按格式，输出到缓存; 参数2用于限制发送的最大字节数，如果达到上限，则只发送上限值-1; 最后1字节自动置'\0'
  va_end(ap);                                          // 清空可变参数列表
  UART6_SendData((uint8_t *)mBuffer, strlen(mBuffer)); // 把字节存放环形缓冲，排队准备发送
}

/******************************************************************************
 * 函    数： UART6_SendAT
 * 功    能： 发送AT命令, 并等待返回信息
 * 参    数： char     *pcString      AT指令字符串
 *            char     *pcAckString   期待的指令返回信息字符串
 *            uint16_t  usTimeOut     发送命令后等待的时间，毫秒
 *
 * 返 回 值： 0-执行失败、1-执行正常
 ******************************************************************************/
#if ZZY_USE_AT_FUNCS
uint8_t UART6_SendAT(char *pcAT, char *pcAckString, uint16_t usTimeOutMs)
{
  UART6_ClearRx();        // 清0
  UART6_SendString(pcAT); // 发送AT指令字符串

  while (usTimeOutMs--) // 判断是否起时(这里只作简单的循环判断次数处理）
  {
    if (UART6_GetRxNum()) // 判断是否接收到数据
    {
      UART6_ClearRx();                                    // 清0接收字节数; 注意：接收到的数据内容 ，是没有被清0的
      if (strstr((char *)UART6_GetRxData(), pcAckString)) // 判断返回数据中是否有期待的字符
        return 1;                                         // 返回：0-超时没有返回、1-正常返回期待值
    }
    zjs_delay_ms(1); // 延时; 用于超时退出处理，避免死等
  }
  return 0; // 返回：0-超时、返回异常，1-正常返回期待值
}
#endif

/******************************************************************************
 * 函  数： UART6_GetRxNum
 * 功  能： 获取最新一帧数据的字节数
 * 参  数： 无
 * 返回值： 0=没有接收到数据，非0=新一帧数据的字节数
 ******************************************************************************/
uint16_t UART6_GetRxNum(void)
{
  return xUART6.usRxNum;
}

/******************************************************************************
 * 函  数： UART6_GetRxData
 * 功  能： 获取最新一帧数据 (数据的地址）
 * 参  数： 无
 * 返回值： 数据的地址(uint8_t*)
 ******************************************************************************/
uint8_t *UART6_GetRxData(void)
{
  return xUART6.puRxData;
}

/******************************************************************************
 * 函  数： UART6_ClearRx
 * 功  能： 清理最后一帧数据的缓存
 *          主要是清0字节数，因为它是用来判断接收的标准
 * 参  数： 无
 * 返回值： 无
 ******************************************************************************/
void UART6_ClearRx(void)
{
  xUART6.usRxNum = 0;
}

/******************************************************************************
 * 函  数： UART6_Register_IDLE_callback
 * 功  能： 注册串口空闲中断函数指针
 * 参  数： 回调函数指针
 * 返回值： 无
 ******************************************************************************/
void UART6_Register_IDLE_callback(uart_callback p)
{
  xUART6.pIDLEcallback = p;
}
#endif // endif UART6_EN

//////////////////////////////////////////////////////////////   UART-7   ///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if UART7_EN

static xUSATR_TypeDef xUART7 = {0};                  // 定义 UART7 的收发结构体
static uint8_t uaUart6RxData[UART7_RX_BUF_SIZE];     // 定义 UART7 的接收缓存
static uint8_t uaUart6TxFiFoData[UART7_TX_BUF_SIZE]; // 定义 UART7 的发送缓存

/******************************************************************************
 * 函  数： UART7_Init
 * 功  能： 初始化UART7的通信引脚、协议参数、中断优先级
 *          协议：波特率-None-8-1
 *          发送：发送中断
 *          接收：接收+空闲中断
 * 返回值： 无
 ******************************************************************************/
void UART7_Init(void)
{
  // 中断优选级配置
  HAL_NVIC_SetPriority(UART7_IRQn, 1, 1); // 配置中断线的优先级
  HAL_NVIC_EnableIRQ(UART7_IRQn);         // 使能中断线
  // 串口中断设置
  UART7->CR1 &= ~(0x01 << 7); // 关闭发送中断
  UART7->CR1 |= 0x01 << 5;    // 使能接收中断: 接收缓冲区非空
  UART7->CR1 |= 0x01 << 4;    // 使能空闲中断：超过1字节时间没收到新数据
  UART7->SR = ~(0x00F0);      // 清理中断
  // 打开串口
  UART7->CR1 |= 0x01 << 13; // 使能UART开始工作
  // 关联缓冲区
  xUART7.puRxData = uaUart6RxData;         // 获取接收缓冲区的地址
  xUART7.puTxFiFoData = uaUart6TxFiFoData; // 获取发送缓冲区的地址
}

/******************************************************************************
 * 函  数： UART7_IRQHandler
 * 功  能： UART7的接收中断、空闲中断、发送中断
 * 参  数： 无
 * 返回值： 无
 *
******************************************************************************/
void UART7_IRQHandler(void)
{
  static uint16_t cnt = 0;                  // 接收字节数累计：每一帧数据已接收到的字节数
  static uint8_t rxTemp[UART7_RX_BUF_SIZE]; // 接收数据缓存数组：每新接收１个字节，先顺序存放到这里，当一帧接收完(发生空闲中断), 再转存到全局变量：xUARTx.puRxData[xx]中；

  // 发送中断：用于把环形缓冲的数据，逐字节发出
  if ((UART7->SR & 1 << 7) && (UART7->CR1 & 1 << 7)) // 检查TXE(发送数据寄存器空)、TXEIE(发送缓冲区空中断使能)
  {
    UART7->DR = xUART7.puTxFiFoData[xUART7.usTxFiFoTail++]; // 把要发送的字节，放入USART的发送寄存器
    if (xUART7.usTxFiFoTail == UART7_TX_BUF_SIZE)            // 如果数据指针到了尾部，就重新标记到0
      xUART7.usTxFiFoTail = 0;
    if (xUART7.usTxFiFoTail == xUART7.usTxFiFoData)
      UART7->CR1 &= ~(1 << 7); // 已发送完成，关闭发送缓冲区空置中断 TXEIE
    return;
  }

  // 接收中断：用于逐个字节接收，存放到临时缓存
  if (UART7->SR & (1 << 5)) // 检查RXNE(读数据寄存器非空标志位); RXNE中断清理方法：读DR时自动清理；
  {
    if ((cnt >= UART7_RX_BUF_SIZE)) //||(xUART1.ReceivedFlag==1   // 判断1: 当前帧已接收到的数据量，已满(缓存区), 为避免溢出，本包后面接收到的数据直接舍弃．
    {
      // 判断2: 如果之前接收好的数据包还没处理，就放弃新数据，即，新数据帧不能覆盖旧数据帧，直至旧数据帧被处理．缺点：数据传输过快于处理速度时会掉包；好处：机制清晰，易于调试
#if ZZY_USE_ERROR_PRINTF
      printf("警告：UART7单帧接收量，已超出接收缓存大小\n!");
#endif
      UART7->DR; // 读取数据寄存器的数据，但不保存．主要作用：读DR时自动清理接收中断标志；
      return;
    }
    rxTemp[cnt++] = UART7->DR; // 把新收到的字节数据，顺序存放到RXTemp数组中；注意：读取DR时自动清零中断位
    return;
  }

  // 空闲中断：用于判断一帧数据结束，整理数据到外部备读
  if (UART7->SR & (1 << 4)) // 检查IDLE(空闲中断标志位); IDLE中断标志清理方法：序列清零，USART1 ->SR;  USART1 ->DR;
  {
    xUART7.usRxNum = 0;                                 // 把接收到的数据字节数清0
    memcpy(xUART7.puRxData, rxTemp, UART7_RX_BUF_SIZE); // 把本帧接收到的数据，存入到结构体的数组成员xUARTx.puRxData中, 等待处理; 注意：复制的是整个数组，包括0值，以方便字符串输出时尾部以0作字符串结束符
    xUART7.usRxNum = cnt;                               // 把接收到的字节数，存入到结构体变量xUARTx.usRxNum中；
    cnt = 0;                                            // 接收字节数累计器，清零; 准备下一次的接收
    memset(rxTemp, 0, UART7_RX_BUF_SIZE);               // 接收数据缓存数组，清零; 准备下一次的接收
    if (xUART7.pIDLEcallback != NULL)
    {                                                   // 判断回调函数不为空
      xUART7.pIDLEcallback(UART7_GetRxData(), UART7_GetRxNum()); // 执行空闲中断回调函数
    }
    UART7->SR;
    UART7->DR; // 清零IDLE中断标志位!! 序列清零，顺序不能错!!
    return;
  }

  return;
}

/******************************************************************************
 * 函  数： UART7_SendData
 * 功  能： UART通过中断发送数据
 *         【适合场景】本函数可发送各种数据，而不限于字符串，如int,char
 *         【不 适 合】注意h文件中所定义的发缓冲区大小、注意数据压入缓冲区的速度与串口发送速度的冲突
 * 参  数： uint8_t  *puData   需发送数据的地址
 *          uint8_t   usNum    发送的字节数 ，数量受限于h文件中设置的发送缓存区大小宏定义
 * 返回值： 无
 ******************************************************************************/
void UART7_SendData(uint8_t *puData, uint16_t usNum)
{
  for (uint16_t i = 0; i < usNum; i++) // 把数据放入环形缓冲区
  {
    xUART7.puTxFiFoData[xUART7.usTxFiFoData++] = puData[i]; // 把字节放到缓冲区最后的位置，然后指针后移
    if (xUART7.usTxFiFoData == UART7_TX_BUF_SIZE)           // 如果指针位置到达缓冲区的最大值，则归0
      xUART7.usTxFiFoData = 0;
  }

  if ((UART7->CR1 & 1 << 7) == 0) // 检查USART寄存器的发送缓冲区空置中断(TXEIE)是否已打开
    UART7->CR1 |= 1 << 7;         // 打开TXEIE中断
}

/******************************************************************************
 * 函  数： UART7_SendString
 * 功  能： 发送字符串
 *          用法请参考printf，及示例中的展示
 *          注意，最大发送字节数为512-1个字符，可在函数中修改上限
 * 参  数： const char *pcString, ...   (如同printf的用法)
 * 返回值： 无
 ******************************************************************************/
void UART7_SendString(const char *pcString, ...)
{
  char mBuffer[512] = {0};
  ;                                                    // 开辟一个缓存, 并把里面的数据置0
  va_list ap;                                          // 新建一个可变参数列表
  va_start(ap, pcString);                              // 列表指向第一个可变参数
  vsnprintf(mBuffer, 512, pcString, ap);               // 把所有参数，按格式，输出到缓存; 参数2用于限制发送的最大字节数，如果达到上限，则只发送上限值-1; 最后1字节自动置'\0'
  va_end(ap);                                          // 清空可变参数列表
  UART7_SendData((uint8_t *)mBuffer, strlen(mBuffer)); // 把字节存放环形缓冲，排队准备发送
}

/******************************************************************************
 * 函    数： UART7_SendAT
 * 功    能： 发送AT命令, 并等待返回信息
 * 参    数： char     *pcString      AT指令字符串
 *            char     *pcAckString   期待的指令返回信息字符串
 *            uint16_t  usTimeOut     发送命令后等待的时间，毫秒
 *
 * 返 回 值： 0-执行失败、1-执行正常
 ******************************************************************************/
#if ZZY_USE_AT_FUNCS
uint8_t UART7_SendAT(char *pcAT, char *pcAckString, uint16_t usTimeOutMs)
{
  UART7_ClearRx();        // 清0
  UART7_SendString(pcAT); // 发送AT指令字符串

  while (usTimeOutMs--) // 判断是否起时(这里只作简单的循环判断次数处理）
  {
    if (UART7_GetRxNum()) // 判断是否接收到数据
    {
      UART7_ClearRx();                                    // 清0接收字节数; 注意：接收到的数据内容 ，是没有被清0的
      if (strstr((char *)UART7_GetRxData(), pcAckString)) // 判断返回数据中是否有期待的字符
        return 1;                                         // 返回：0-超时没有返回、1-正常返回期待值
    }
    zjs_delay_ms(1); // 延时; 用于超时退出处理，避免死等
  }
  return 0; // 返回：0-超时、返回异常，1-正常返回期待值
}
#endif

/******************************************************************************
 * 函  数： UART7_GetRxNum
 * 功  能： 获取最新一帧数据的字节数
 * 参  数： 无
 * 返回值： 0=没有接收到数据，非0=新一帧数据的字节数
 ******************************************************************************/
uint16_t UART7_GetRxNum(void)
{
  return xUART7.usRxNum;
}

/******************************************************************************
 * 函  数： UART7_GetRxData
 * 功  能： 获取最新一帧数据 (数据的地址）
 * 参  数： 无
 * 返回值： 数据的地址(uint8_t*)
 ******************************************************************************/
uint8_t *UART7_GetRxData(void)
{
  return xUART7.puRxData;
}

/******************************************************************************
 * 函  数： UART7_ClearRx
 * 功  能： 清理最后一帧数据的缓存
 *          主要是清0字节数，因为它是用来判断接收的标准
 * 参  数： 无
 * 返回值： 无
 ******************************************************************************/
void UART7_ClearRx(void)
{
  xUART7.usRxNum = 0;
}

/******************************************************************************
 * 函  数： UART7_Register_IDLE_callback
 * 功  能： 注册串口空闲中断函数指针
 * 参  数： 回调函数指针
 * 返回值： 无
 ******************************************************************************/
void UART7_Register_IDLE_callback(uart_callback p)
{
  xUART7.pIDLEcallback = p;
}
#endif // endif UART7_EN

//////////////////////////////////////////////////////////////   UART-8   ///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if UART8_EN

static xUSATR_TypeDef xUART8 = {0};                  // 定义 UART8 的收发结构体
static uint8_t uaUart6RxData[UART8_RX_BUF_SIZE];     // 定义 UART8 的接收缓存
static uint8_t uaUart6TxFiFoData[UART8_TX_BUF_SIZE]; // 定义 UART8 的发送缓存

/******************************************************************************
 * 函  数： UART8_Init
 * 功  能： 初始化UART8的通信引脚、协议参数、中断优先级
 *          协议：波特率-None-8-1
 *          发送：发送中断
 *          接收：接收+空闲中断
 * 返回值： 无
 ******************************************************************************/
void UART8_Init(void)
{
  // 中断优选级配置
  HAL_NVIC_SetPriority(UART8_IRQn, 1, 1); // 配置中断线的优先级
  HAL_NVIC_EnableIRQ(UART8_IRQn);         // 使能中断线
  // 串口中断设置
  UART8->CR1 &= ~(0x01 << 7); // 关闭发送中断
  UART8->CR1 |= 0x01 << 5;    // 使能接收中断: 接收缓冲区非空
  UART8->CR1 |= 0x01 << 4;    // 使能空闲中断：超过1字节时间没收到新数据
  UART8->SR = ~(0x00F0);      // 清理中断
  // 打开串口
  UART8->CR1 |= 0x01 << 13; // 使能UART开始工作
  // 关联缓冲区
  xUART8.puRxData = uaUart6RxData;         // 获取接收缓冲区的地址
  xUART8.puTxFiFoData = uaUart6TxFiFoData; // 获取发送缓冲区的地址
}

/******************************************************************************
 * 函  数： UART8_IRQHandler
 * 功  能： UART8的接收中断、空闲中断、发送中断
 * 参  数： 无
 * 返回值： 无
 *
******************************************************************************/
void UART8_IRQHandler(void)
{
  static uint16_t cnt = 0;                  // 接收字节数累计：每一帧数据已接收到的字节数
  static uint8_t rxTemp[UART8_RX_BUF_SIZE]; // 接收数据缓存数组：每新接收１个字节，先顺序存放到这里，当一帧接收完(发生空闲中断), 再转存到全局变量：xUARTx.puRxData[xx]中；

  // 发送中断：用于把环形缓冲的数据，逐字节发出
  if ((UART8->SR & 1 << 7) && (UART8->CR1 & 1 << 7)) // 检查TXE(发送数据寄存器空)、TXEIE(发送缓冲区空中断使能)
  {
    UART8->DR = xUART8.puTxFiFoData[xUART8.usTxFiFoTail++]; // 把要发送的字节，放入USART的发送寄存器
    if (xUART8.usTxFiFoTail == UART8_TX_BUF_SIZE)            // 如果数据指针到了尾部，就重新标记到0
      xUART8.usTxFiFoTail = 0;
    if (xUART8.usTxFiFoTail == xUART8.usTxFiFoData)
      UART8->CR1 &= ~(1 << 7); // 已发送完成，关闭发送缓冲区空置中断 TXEIE
    return;
  }

  // 接收中断：用于逐个字节接收，存放到临时缓存
  if (UART8->SR & (1 << 5)) // 检查RXNE(读数据寄存器非空标志位); RXNE中断清理方法：读DR时自动清理；
  {
    if ((cnt >= UART8_RX_BUF_SIZE)) //||(xUART1.ReceivedFlag==1   // 判断1: 当前帧已接收到的数据量，已满(缓存区), 为避免溢出，本包后面接收到的数据直接舍弃．
    {
      // 判断2: 如果之前接收好的数据包还没处理，就放弃新数据，即，新数据帧不能覆盖旧数据帧，直至旧数据帧被处理．缺点：数据传输过快于处理速度时会掉包；好处：机制清晰，易于调试
#if ZZY_USE_ERROR_PRINTF
      printf("警告：UART8单帧接收量，已超出接收缓存大小\n!");
#endif
      UART8->DR; // 读取数据寄存器的数据，但不保存．主要作用：读DR时自动清理接收中断标志；
      return;
    }
    rxTemp[cnt++] = UART8->DR; // 把新收到的字节数据，顺序存放到RXTemp数组中；注意：读取DR时自动清零中断位
    return;
  }

  // 空闲中断：用于判断一帧数据结束，整理数据到外部备读
  if (UART8->SR & (1 << 4)) // 检查IDLE(空闲中断标志位); IDLE中断标志清理方法：序列清零，USART1 ->SR;  USART1 ->DR;
  {
    xUART8.usRxNum = 0;                                 // 把接收到的数据字节数清0
    memcpy(xUART8.puRxData, rxTemp, UART8_RX_BUF_SIZE); // 把本帧接收到的数据，存入到结构体的数组成员xUARTx.puRxData中, 等待处理; 注意：复制的是整个数组，包括0值，以方便字符串输出时尾部以0作字符串结束符
    xUART8.usRxNum = cnt;                               // 把接收到的字节数，存入到结构体变量xUARTx.usRxNum中；
    cnt = 0;                                            // 接收字节数累计器，清零; 准备下一次的接收
    memset(rxTemp, 0, UART8_RX_BUF_SIZE);               // 接收数据缓存数组，清零; 准备下一次的接收
    if (xUART8.pIDLEcallback != NULL)
    {                                                   // 判断回调函数不为空
      xUART8.pIDLEcallback(UART8_GetRxData(), UART8_GetRxNum()); // 执行空闲中断回调函数
    }
    UART8->SR;
    UART8->DR; // 清零IDLE中断标志位!! 序列清零，顺序不能错!!
    return;
  }

  return;
}

/******************************************************************************
 * 函  数： UART8_SendData
 * 功  能： UART通过中断发送数据
 *         【适合场景】本函数可发送各种数据，而不限于字符串，如int,char
 *         【不 适 合】注意h文件中所定义的发缓冲区大小、注意数据压入缓冲区的速度与串口发送速度的冲突
 * 参  数： uint8_t  *puData   需发送数据的地址
 *          uint8_t   usNum    发送的字节数 ，数量受限于h文件中设置的发送缓存区大小宏定义
 * 返回值： 无
 ******************************************************************************/
void UART8_SendData(uint8_t *puData, uint16_t usNum)
{
  for (uint16_t i = 0; i < usNum; i++) // 把数据放入环形缓冲区
  {
    xUART8.puTxFiFoData[xUART8.usTxFiFoData++] = puData[i]; // 把字节放到缓冲区最后的位置，然后指针后移
    if (xUART8.usTxFiFoData == UART8_TX_BUF_SIZE)           // 如果指针位置到达缓冲区的最大值，则归0
      xUART8.usTxFiFoData = 0;
  }

  if ((UART8->CR1 & 1 << 7) == 0) // 检查USART寄存器的发送缓冲区空置中断(TXEIE)是否已打开
    UART8->CR1 |= 1 << 7;         // 打开TXEIE中断
}

/******************************************************************************
 * 函  数： UART8_SendString
 * 功  能： 发送字符串
 *          用法请参考printf，及示例中的展示
 *          注意，最大发送字节数为512-1个字符，可在函数中修改上限
 * 参  数： const char *pcString, ...   (如同printf的用法)
 * 返回值： 无
 ******************************************************************************/
void UART8_SendString(const char *pcString, ...)
{
  char mBuffer[512] = {0};
  ;                                                    // 开辟一个缓存, 并把里面的数据置0
  va_list ap;                                          // 新建一个可变参数列表
  va_start(ap, pcString);                              // 列表指向第一个可变参数
  vsnprintf(mBuffer, 512, pcString, ap);               // 把所有参数，按格式，输出到缓存; 参数2用于限制发送的最大字节数，如果达到上限，则只发送上限值-1; 最后1字节自动置'\0'
  va_end(ap);                                          // 清空可变参数列表
  UART8_SendData((uint8_t *)mBuffer, strlen(mBuffer)); // 把字节存放环形缓冲，排队准备发送
}

/******************************************************************************
 * 函    数： UART8_SendAT
 * 功    能： 发送AT命令, 并等待返回信息
 * 参    数： char     *pcString      AT指令字符串
 *            char     *pcAckString   期待的指令返回信息字符串
 *            uint16_t  usTimeOut     发送命令后等待的时间，毫秒
 *
 * 返 回 值： 0-执行失败、1-执行正常
 ******************************************************************************/
#if ZZY_USE_AT_FUNCS
uint8_t UART8_SendAT(char *pcAT, char *pcAckString, uint16_t usTimeOutMs)
{
  UART8_ClearRx();        // 清0
  UART8_SendString(pcAT); // 发送AT指令字符串

  while (usTimeOutMs--) // 判断是否起时(这里只作简单的循环判断次数处理）
  {
    if (UART8_GetRxNum()) // 判断是否接收到数据
    {
      UART8_ClearRx();                                    // 清0接收字节数; 注意：接收到的数据内容 ，是没有被清0的
      if (strstr((char *)UART8_GetRxData(), pcAckString)) // 判断返回数据中是否有期待的字符
        return 1;                                         // 返回：0-超时没有返回、1-正常返回期待值
    }
    zjs_delay_ms(1); // 延时; 用于超时退出处理，避免死等
  }
  return 0; // 返回：0-超时、返回异常，1-正常返回期待值
}
#endif

/******************************************************************************
 * 函  数： UART8_GetRxNum
 * 功  能： 获取最新一帧数据的字节数
 * 参  数： 无
 * 返回值： 0=没有接收到数据，非0=新一帧数据的字节数
 ******************************************************************************/
uint16_t UART8_GetRxNum(void)
{
  return xUART8.usRxNum;
}

/******************************************************************************
 * 函  数： UART8_GetRxData
 * 功  能： 获取最新一帧数据 (数据的地址）
 * 参  数： 无
 * 返回值： 数据的地址(uint8_t*)
 ******************************************************************************/
uint8_t *UART8_GetRxData(void)
{
  return xUART8.puRxData;
}

/******************************************************************************
 * 函  数： UART8_ClearRx
 * 功  能： 清理最后一帧数据的缓存
 *          主要是清0字节数，因为它是用来判断接收的标准
 * 参  数： 无
 * 返回值： 无
 ******************************************************************************/
void UART8_ClearRx(void)
{
  xUART8.usRxNum = 0;
}

/******************************************************************************
 * 函  数： UART8_Register_IDLE_callback
 * 功  能： 注册串口空闲中断函数指针
 * 参  数： 回调函数指针
 * 返回值： 无
 ******************************************************************************/
void UART8_Register_IDLE_callback(uart_callback p)
{
  xUART8.pIDLEcallback = p;
}
#endif // endif UART8_EN

/**
 * @brief 初始化串口
 * @param my_usart 串口号
 */
void UART_Init(USART_TypeDef *my_usart)
{
#if UART1_EN
if (my_usart == USART1)
    UART1_Init();
#endif
#if UART2_EN
  if (my_usart == USART2)
    UART2_Init();
#endif
#if UART3_EN
  if (my_usart == USART3)
    UART3_Init();
#endif
#if UART4_EN
  if (my_usart == UART4)
    UART4_Init();
#endif
#if UART5_EN
  if (my_usart == UART5)
    UART5_Init();
#endif
#if UART6_EN
  if (my_usart == USART6)
    UART6_Init();
#endif
#if UART7_EN
  if (my_usart == UART7)
    UART7_Init();
#endif
#if UART8_EN
  if (my_usart == UART8)
    UART8_Init();
#endif
}

/**
 * @brief 发送指定数据
 * @param my_usart 串口号
 * @param puData 数据地址
 * @param usNum 字节数
 */
void UART_SendData(USART_TypeDef *my_usart, uint8_t *puData, uint16_t usNum)
{
#if UART1_EN
  if (my_usart == USART1)
    UART1_SendData(puData, usNum);
#endif
#if UART2_EN
  if (my_usart == USART2)
    UART2_SendData(puData, usNum);
#endif
#if UART3_EN
  if (my_usart == USART3)
    UART3_SendData(puData, usNum);
#endif
#if UART4_EN
  if (my_usart == UART4)
    UART4_SendData(puData, usNum);
#endif
#if UART5_EN
  if (my_usart == UART5)
    UART5_SendData(puData, usNum);
#endif
#if UART6_EN
  if (my_usart == USART6)
    UART6_SendData(puData, usNum);
#endif
#if UART7_EN
  if (my_usart == UART7)
    UART7_SendData(puData, usNum);
#endif
#if UART8_EN
  if (my_usart == UART8)
    UART8_SendData(puData, usNum);
#endif
}

/**
 * @brief 发送字符串，使用方法如同printf
 * @param my_usart 串口号
 * @param pcString 字符串地址
 */
void UART_SendString(USART_TypeDef *my_usart, const char *pcString, ...)
{
  char mBuffer[512] = {0};
  va_list ap;
  va_start(ap, pcString);
  vsnprintf(mBuffer, 512, pcString, ap);
  va_end(ap);
  UART_SendData(my_usart, (uint8_t *)mBuffer, strlen(mBuffer));
}

/**
 * @brief 本函数，针对ESP8266、蓝牙模块等AT固件，用于等待返回期待的信息
 * @param my_usart 串口号
 * @param pcAT AT指令字符串
 * @param pcAckString 期待返回信息字符串
 * @param usTimeOutMs 等待超时
 * @return 0-执行失败、1-执行成功
 */
#if ZZY_USE_AT_FUNCS
uint8_t UART_SendAT(USART_TypeDef *my_usart, char *pcAT, char *pcAckString, uint16_t usTimeOutMs)
{
#if UART1_EN
  if (my_usart == USART1)
    return UART1_SendAT(pcAT, pcAckString, usTimeOutMs);
#endif
#if UART2_EN
  if (my_usart == USART2)
    return UART2_SendAT(pcAT, pcAckString, usTimeOutMs);
#endif
#if UART3_EN
  if (my_usart == USART3)
    return UART3_SendAT(pcAT, pcAckString, usTimeOutMs);
#endif
#if UART4_EN
  if (my_usart == UART4)
    return UART4_SendAT(pcAT, pcAckString, usTimeOutMs);
#endif
#if UART5_EN
  if (my_usart == UART5)
    return UART5_SendAT(pcAT, pcAckString, usTimeOutMs);
#endif
#if UART6_EN
  if (my_usart == USART6)
    return UART6_SendAT(pcAT, pcAckString, usTimeOutMs);
#endif
#if UART7_EN
  if (my_usart == UART7)
    return UART7_SendAT(pcAT, pcAckString, usTimeOutMs);
#endif
#if UART8_EN
  if (my_usart == UART8)
    return UART8_SendAT(pcAT, pcAckString, usTimeOutMs);
#endif
  return 0;
}
#endif

/**
 * @brief 获取接收到的最新一帧字节数
 * @param my_usart 串口号
 * @return 最新一帧字节数
 */
uint16_t UART_GetRxNum(USART_TypeDef *my_usart)
{
#if UART1_EN
  if (my_usart == USART1)
    return UART1_GetRxNum();
#endif
#if UART2_EN
  if (my_usart == USART2)
    return UART2_GetRxNum();
#endif
#if UART3_EN
  if (my_usart == USART3)
    return UART3_GetRxNum();
#endif
#if UART4_EN
  if (my_usart == UART4)
    return UART4_GetRxNum();
#endif
#if UART5_EN
  if (my_usart == UART5)
    return UART5_GetRxNum();
#endif
#if UART6_EN
  if (my_usart == USART6)
    return UART6_GetRxNum();
#endif
#if UART7_EN
  if (my_usart == UART7)
    return UART7_GetRxNum();
#endif
#if UART8_EN
  if (my_usart == UART8)
    return UART8_GetRxNum();
#endif
  return 0;
}

/**
 * @brief 获取接收到的最新一帧数据地址
 * @param my_usart 串口号
 * @return 最新接收到的数据地址
 */
uint8_t *UART_GetRxData(USART_TypeDef *my_usart)
{
#if UART1_EN
  if (my_usart == USART1)
    return UART1_GetRxData();
#endif
#if UART2_EN
  if (my_usart == USART2)
    return UART2_GetRxData();
#endif
#if UART3_EN
  if (my_usart == USART3)
    return UART3_GetRxData();
#endif
#if UART4_EN
  if (my_usart == UART4)
    return UART4_GetRxData();
#endif
#if UART5_EN
  if (my_usart == UART5)
    return UART5_GetRxData();
#endif
#if UART6_EN
  if (my_usart == USART6)
    return UART6_GetRxData();
#endif
#if UART7_EN
  if (my_usart == UART7)
    return UART7_GetRxData();
#endif
#if UART8_EN
  if (my_usart == UART8)
    return UART7_GetRxData();
#endif
  return NULL;
}

/**
 * @brief 清理接收到的数据 (清理最后一帧字节数，因为它是判断接收的标志)
 * @param my_usart 串口号
 * @return 最新接收到的数据地址
 */
void UART_ClearRx(USART_TypeDef *my_usart)
{
#if UART1_EN
  if (my_usart == USART1)
    UART1_ClearRx();
#endif
#if UART2_EN
  if (my_usart == USART2)
    UART2_ClearRx();
#endif
#if UART3_EN
  if (my_usart == USART3)
    UART3_ClearRx();
#endif
#if UART4_EN
  if (my_usart == UART4)
    UART4_ClearRx();
#endif
#if UART5_EN
  if (my_usart == UART5)
    UART5_ClearRx();
#endif
#if UART6_EN
  if (my_usart == USART6)
    UART6_ClearRx();
#endif
#if UART7_EN
  if (my_usart == UART7)
    UART7_ClearRx();
#endif
#if UART8_EN
  if (my_usart == UART8)
    UART8_ClearRx();
#endif
}

/**
 * @brief 清理接收到的数据 (清理最后一帧字节数，因为它是判断接收的标志)
 * @param my_usart 串口号
 * @param p 空闲中断回调函数指针
 * @return 最新接收到的数据地址
 */
void UART_Register_IDLE_callback(USART_TypeDef *my_usart, uart_callback p)
{
  #if UART1_EN
  if (my_usart == USART1)
    UART1_Register_IDLE_callback(p);
#endif
#if UART2_EN
  if (my_usart == USART2)
    UART2_Register_IDLE_callback(p);
#endif
#if UART3_EN
  if (my_usart == USART3)
    UART3_Register_IDLE_callback(p);
#endif
#if UART4_EN
  if (my_usart == UART4)
    UART4_Register_IDLE_callback(p);
#endif
#if UART5_EN
  if (my_usart == UART5)
    UART5_Register_IDLE_callback(p);
#endif
#if UART6_EN
  if (my_usart == USART6)
    UART6_Register_IDLE_callback(p);
#endif
#if UART7_EN
  if (my_usart == UART7)
    UART7_Register_IDLE_callback(p);
#endif
#if UART8_EN
  if (my_usart == UART8)
    UART8_Register_IDLE_callback(p);
#endif
}

/////////////////////////////////////////////////////////////   辅助函数   ///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief 调试辅助函数
 * @param puRxData 字符串
 * @param usRxNum 长度
 */
void showData(uint8_t *puRxData, uint16_t usRxNum)
{
  printf("字节数： %d \n", usRxNum);                // 显示字节数
  printf("ASCII 显示数据: %s\n", (char *)puRxData); // 显示数据，以ASCII方式显示，即以字符串的方式显示
  printf("16进制显示数据: ");                       // 显示数据，以16进制方式，显示每一个字节的值
  while (usRxNum--)                                 // 逐个字节判断，只要不为'\0', 就继续
    printf("0x%X ", *puRxData++);                   // 格式化
  printf("\n\n");                                   // 显示换行
}

/////////////////////////////////////////////////////////////   串口重定向   /////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if ZZY_USE_PRINTF
 /**
 * @brief printf函数支持代码
 * @warning 加入以下代码, 使用printf函数时, 不再需要打勾use MicroLIB
 */

#pragma import(__use_no_semihosting)

struct __FILE
{
  int handle;
}; // 标准库需要的支持函数

FILE __stdout; // FILE 在stdio.h文件
void _sys_exit(int x)
{
  x = x; // 定义_sys_exit()以避免使用半主机模式
}

void _ttywrch(int ch)
{
  ch = ch;
}

int fputc(int ch, FILE *f) // 重定向fputc函数，使printf的输出，由fputc输出到UART
{
  UART1_SendData((uint8_t *)&ch, 1); // 使用队列+中断方式发送数据; 无需像方式1那样等待耗时，但要借助已写好的函数、环形缓冲
  return ch;
}

#endif
