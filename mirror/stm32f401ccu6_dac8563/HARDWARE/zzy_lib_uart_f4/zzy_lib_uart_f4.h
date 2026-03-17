#ifndef __ZZY_LIB_UART_F4_H
#define __ZZY_LIB_UART_F4_H
/**
  ******************************************************************************
  * @file           : zzy_lib_uart_f4.h
  * @brief          : a uart lib for stm32h7xx
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

#include "zzy_global_head.h"

/**
 * @brief 移植配置修改区
 */
// 函数开关
#define ZZY_USE_ERROR_PRINTF           1          // 错误日志打印开关
#define ZZY_USE_AT_FUNCS               0          // 是否使用at指令函数
#define ZZY_USE_PRINTF                 1          // 是否使用串口重定向
// 串口开关
#define UART1_EN                       0          // 串口1，0=关、1=启用;  倘若没用到UART1, 置0，就不会开辟UART1发送缓存、接收缓存，省一点资源;
#define UART2_EN                       0          // 串口2，0=关、1=启用;  同上;
#define UART3_EN                       0          // 串口3，0=关、1=启用;  同上;
#define UART4_EN                       0          // 串口4，0=关、1=启用;  同上;
#define UART5_EN                       0          // 串口5，0=关、1=启用;  同上;
#define UART6_EN                       0          // 串口6，0=关、1=启用;  同上;
#define UART7_EN                       0          // 串口7，0=关、1=启用;  同上;
#define UART8_EN                       0          // 串口8，0=关、1=启用;  同上;
// 发送缓冲区大小
#define UART1_TX_BUF_SIZE           2048          // 配置每组UART发送环形缓冲区数组的大小，单位：字节数; 
#define UART2_TX_BUF_SIZE            512          // -- 只有在前面串口开关在打开状态，才会定义具体的缓冲区数组
#define UART3_TX_BUF_SIZE            512          // -- 默认值：512，此值已能适配大部场景的通信; 如果与ESP8266之类的设备通信，可适当增大此值。
#define UART4_TX_BUF_SIZE            512          // -- 值范围：1~65535; 注意初始化后，不要超过芯片最大RAM值。
#define UART5_TX_BUF_SIZE            512          // -- 注意此值是一个环形缓冲区大小，决定每一帧或多帧数据进入发送前的总缓存字节数，先进先出。
#define UART6_TX_BUF_SIZE            512          //
#define UART7_TX_BUF_SIZE            512          //
#define UART8_TX_BUF_SIZE            512          //
// 接收缓冲区大小
#define UART1_RX_BUF_SIZE            512          // 配置每组UART接收缓冲区的大小，单位：字节; 此值影响范围：中断里的接收缓存大小，接收后数据缓存的大小
#define UART2_RX_BUF_SIZE            512          // --- 当接收到的一帧数据字节数，小于此值时，数据正常；
#define UART3_RX_BUF_SIZE            512          // --- 当接收到的一帧数据字节数，超过此值时，超出部分，将在中断中直接弃舍，直到此帧接收结束(发生空闲中断); 
#define UART4_RX_BUF_SIZE            512          // 
#define UART5_RX_BUF_SIZE            512          // 
#define UART6_RX_BUF_SIZE            512          //
#define UART7_RX_BUF_SIZE            512          //
#define UART8_RX_BUF_SIZE            512          //
// 结束-配置修改

/**
 * @brief 自定义回调函数类型
 */
typedef void (*uart_callback)(uint8_t *, uint16_t);

/**
 * @brief 初始化串口
 * @param my_usart 串口号
 */
void UART_Init(USART_TypeDef *my_usart);

/**
 * @brief 发送指定数据
 * @param my_usart 串口号
 * @param puData 数据地址
 * @param usNum 字节数
 */
void UART_SendData(USART_TypeDef *my_usart, uint8_t *puData, uint16_t usNum);

/**
 * @brief 发送字符串，使用方法如同printf
 * @param my_usart 串口号
 * @param pcString 字符串地址
 */
void UART_SendString(USART_TypeDef *my_usart, const char *pcString, ...);

/**
 * @brief 获取接收到的最新一帧字节数
 * @param my_usart 串口号
 * @return 最新一帧字节数
 */
uint16_t UART_GetRxNum(USART_TypeDef *my_usart);

/**
 * @brief 获取接收到的最新一帧数据地址
 * @param my_usart 串口号
 * @return 最新接收到的数据地址
 */
uint8_t * UART_GetRxData(USART_TypeDef *my_usart);

/**
 * @brief 清理接收到的数据 (清理最后一帧字节数，因为它是判断接收的标志)
 * @param my_usart 串口号
 */
void UART_ClearRx(USART_TypeDef *my_usart);

/**
 * @brief 清理接收到的数据 (清理最后一帧字节数，因为它是判断接收的标志)
 * @param my_usart 串口号
 * @param p 空闲中断回调函数指针
 * @return 最新接收到的数据地址
 */
void UART_Register_IDLE_callback(USART_TypeDef *my_usart, uart_callback p);

#if ZZY_USE_AT_FUNCS
/**
 * @brief 本函数，针对ESP8266、蓝牙模块等AT固件，用于等待返回期待的信息
 * @param my_usart 串口号
 * @param pcAT AT指令字符串
 * @param pcAckString 期待返回信息字符串
 * @param usTimeOutMs 等待超时
 * @return 0-执行失败、1-执行成功
 */
uint8_t UART_SendAT(USART_TypeDef *my_usart, char *pcAT, char *pcAckString, uint16_t usTimeOutMs)
#endif

#if UART1_EN
// UART1
void      UART1_Init(void);                                                     // 初始化串口
void      UART1_SendData(uint8_t *puData, uint16_t usNum);                      // 发送指定数据; 参数：数据地址、字节数
void      UART1_SendString(const char *pcString, ...);                          // 发送字符串;   参数：字符串地址; 使用方法如同printf
uint16_t  UART1_GetRxNum(void);                                                 // 获取接收到的最新一帧字节数
uint8_t * UART1_GetRxData(void);                                                // 获取接收到的数据 (缓存的地址)
void      UART1_ClearRx(void);                                                  // 清理接收到的数据 (清理最后一帧字节数，因为它是判断接收的标志)
void      UART1_Register_IDLE_callback(uart_callback p);                        // 注册串口空闲中断函数指针
#if ZZY_USE_AT_FUNCS
uint8_t   UART1_SendAT(char *pcAT, char *pcAckString, uint16_t usTimeOutMs);    // 本函数，针对ESP8266、蓝牙模块等AT固件，用于等待返回期待的信息; 参数：AT指令字符串、期待返回信息字符串、ms等待超时; 返回：0-执行失败、1-执行成功 
#endif
#endif
#if UART2_EN
// UART2
void      UART2_Init(void);                                                     // 初始化串口
void      UART2_SendData(uint8_t *puData, uint16_t usNum);                      // 发送指定数据; 参数：数据地址、字节数
void      UART2_SendString(const char *pcString, ...);                          // 发送字符串;   参数：字符串地址; 使用方法如同printf
uint16_t  UART2_GetRxNum(void);                                                 // 获取接收到的最新一帧字节数
uint8_t * UART2_GetRxData(void);                                                // 获取接收到的数据 (缓存的地址)
void      UART2_ClearRx(void);                                                  // 清理接收到的数据 (清理最后一帧字节数，因为它是判断接收的标志)
void      UART2_Register_IDLE_callback(uart_callback p);                        // 注册串口空闲中断函数指针
#if ZZY_USE_AT_FUNCS
uint8_t   UART2_SendAT(char *pcAT, char *pcAckString, uint16_t usTimeOutMs);    // 本函数，针对ESP8266、蓝牙模块等AT固件，用于等待返回期待的信息; 参数：AT指令字符串、期待返回信息字符串、ms等待超时; 返回：0-执行失败、1-执行成功 
#endif
#endif
#if UART3_EN
// UART3
void      UART3_Init(void);                                                     // 初始化串口
void      UART3_SendData(uint8_t *puData, uint16_t usNum);                      // 发送指定数据; 参数：数据地址、字节数
void      UART3_SendString(const char *pcString, ...);                          // 发送字符串;   参数：字符串地址; 使用方法如同printf
uint16_t  UART3_GetRxNum(void);                                                 // 获取接收到的最新一帧字节数
uint8_t * UART3_GetRxData(void);                                                // 获取接收到的数据 (缓存的地址)
void      UART3_ClearRx(void);                                                  // 清理接收到的数据 (清理最后一帧字节数，因为它是判断接收的标志)
void      UART3_Register_IDLE_callback(uart_callback p);                        // 注册串口空闲中断函数指针
#if ZZY_USE_AT_FUNCS
uint8_t   UART3_SendAT(char *pcAT, char *pcAckString, uint16_t usTimeOutMs);    // 本函数，针对ESP8266、蓝牙模块等AT固件，用于等待返回期待的信息; 参数：AT指令字符串、期待返回信息字符串、ms等待超时; 返回：0-执行失败、1-执行成功 
#endif
#endif
#if UART4_EN
// UART4
void      UART4_Init(void);                                                     // 初始化串口
void      UART4_SendData(uint8_t *puData, uint16_t usNum);                      // 发送指定数据; 参数：数据地址、字节数
void      UART4_SendString(const char *pcString, ...);                          // 发送字符串;   参数：字符串地址; 使用方法如同printf
uint16_t  UART4_GetRxNum(void);                                                 // 获取接收到的最新一帧字节数
uint8_t * UART4_GetRxData(void);                                                // 获取接收到的数据 (缓存的地址)
void      UART4_ClearRx(void);                                                  // 清理接收到的数据 (清理最后一帧字节数，因为它是判断接收的标志)
void      UART4_Register_IDLE_callback(uart_callback p);                        // 注册串口空闲中断函数指针
#if ZZY_USE_AT_FUNCS
uint8_t   UART4_SendAT(char *pcAT, char *pcAckString, uint16_t usTimeOutMs);    // 本函数，针对ESP8266、蓝牙模块等AT固件，用于等待返回期待的信息; 参数：AT指令字符串、期待返回信息字符串、ms等待超时; 返回：0-执行失败、1-执行成功 
#endif
#endif
#if UART5_EN
// UART5
void      UART5_Init(void);                                                     // 初始化串口
void      UART5_SendData(uint8_t *puData, uint16_t usNum);                      // 发送指定数据; 参数：数据地址、字节数
void      UART5_SendString(const char *pcString, ...);                          // 发送字符串;   参数：字符串地址; 使用方法如同printf
uint16_t  UART5_GetRxNum(void);                                                 // 获取接收到的最新一帧字节数
uint8_t * UART5_GetRxData(void);                                                // 获取接收到的数据 (缓存的地址)
void      UART5_ClearRx(void);                                                  // 清理接收到的数据 (清理最后一帧字节数，因为它是判断接收的标志)
void      UART5_Register_IDLE_callback(uart_callback p);                        // 注册串口空闲中断函数指针
#if ZZY_USE_AT_FUNCS
uint8_t   UART5_SendAT(char *pcAT, char *pcAckString, uint16_t usTimeOutMs);    // 本函数，针对ESP8266、蓝牙模块等AT固件，用于等待返回期待的信息; 参数：AT指令字符串、期待返回信息字符串、ms等待超时; 返回：0-执行失败、1-执行成功 
#endif
#endif
#if UART6_EN
// UART6
void      UART6_Init(void);                                                     // 初始化串口
void      UART6_SendData(uint8_t *puData, uint16_t usNum);                      // 发送指定数据; 参数：数据地址、字节数
void      UART6_SendString(const char *pcString, ...);                          // 发送字符串;   参数：字符串地址; 使用方法如同printf
uint16_t  UART6_GetRxNum(void);                                                 // 获取接收到的最新一帧字节数
uint8_t * UART6_GetRxData(void);                                                // 获取接收到的数据 (缓存的地址)
void      UART6_ClearRx(void);                                                  // 清理接收到的数据 (清理最后一帧字节数，因为它是判断接收的标志)
void      UART6_Register_IDLE_callback(uart_callback p);                        // 注册串口空闲中断函数指针
#if ZZY_USE_AT_FUNCS
uint8_t   UART6_SendAT(char *pcAT, char *pcAckString, uint16_t usTimeOutMs);    // 本函数，针对ESP8266、蓝牙模块等AT固件，用于等待返回期待的信息; 参数：AT指令字符串、期待返回信息字符串、ms等待超时; 返回：0-执行失败、1-执行成功 
#endif
#endif
#if UART7_EN
// UART7
void      UART7_Init(void);                                                     // 初始化串口
void      UART7_SendData(uint8_t *puData, uint16_t usNum);                      // 发送指定数据; 参数：数据地址、字节数
void      UART7_SendString(const char *pcString, ...);                          // 发送字符串;   参数：字符串地址; 使用方法如同printf
uint16_t  UART7_GetRxNum(void);                                                 // 获取接收到的最新一帧字节数
uint8_t * UART7_GetRxData(void);                                                // 获取接收到的数据 (缓存的地址)
void      UART7_ClearRx(void);                                                  // 清理接收到的数据 (清理最后一帧字节数，因为它是判断接收的标志)
void      UART7_Register_IDLE_callback(uart_callback p);                        // 注册串口空闲中断函数指针
#if ZZY_USE_AT_FUNCS
uint8_t   UART7_SendAT(char *pcAT, char *pcAckString, uint16_t usTimeOutMs);    // 本函数，针对ESP8266、蓝牙模块等AT固件，用于等待返回期待的信息; 参数：AT指令字符串、期待返回信息字符串、ms等待超时; 返回：0-执行失败、1-执行成功 
#endif
#endif
#if UART8_EN
// UART8
void      UART8_Init(void);                                                     // 初始化串口
void      UART8_SendData(uint8_t *puData, uint16_t usNum);                      // 发送指定数据; 参数：数据地址、字节数
void      UART8_SendString(const char *pcString, ...);                          // 发送字符串;   参数：字符串地址; 使用方法如同printf
uint16_t  UART8_GetRxNum(void);                                                 // 获取接收到的最新一帧字节数
uint8_t * UART8_GetRxData(void);                                                // 获取接收到的数据 (缓存的地址)
void      UART8_ClearRx(void);                                                  // 清理接收到的数据 (清理最后一帧字节数，因为它是判断接收的标志)
void      UART8_Register_IDLE_callback(uart_callback p);                        // 注册串口空闲中断函数指针
#if ZZY_USE_AT_FUNCS
uint8_t   UART8_SendAT(char *pcAT, char *pcAckString, uint16_t usTimeOutMs);    // 本函数，针对ESP8266、蓝牙模块等AT固件，用于等待返回期待的信息; 参数：AT指令字符串、期待返回信息字符串、ms等待超时; 返回：0-执行失败、1-执行成功 
#endif
#endif
  
/**
 * @brief 调试辅助函数
 * @param puRxData 字符串
 * @param usRxNum 长度
 */
void      showData(uint8_t *puRxData, uint16_t usRxNum);                        // 经printf，发送到串口助手上，方便观察


#endif

