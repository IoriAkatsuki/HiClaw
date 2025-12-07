#ifndef __ZZY_LIB_DELAY_H
#define __ZZY_LIB_DELAY_H
/**
  ******************************************************************************
  * @file           : zzy_lib_dealy.h
  * @author         : zjs
  * @brief          : a blocking delay lib
  ******************************************************************************
  * @attention
  *
  * Change Logs:
  * Date           Author       Notes
  * 2025-7-11      zzy          the first version
  *
  ******************************************************************************
  */

#include "zzy_global_head.h"

/**
 * @brief CPU主频配置修改
 */
#define ZJS_CPU_FREQUENCY_MHZ 84 // 当前的时钟频率下，1us对应的tick数

/**
 * @brief 微秒延时
 * @param vu32Us 需要延时的微秒数
 */
void zjs_delay_us(volatile uint32_t vu32Us);

/**
 * @brief 毫秒延时
 * @param u32Ms  需要延时的毫秒数
 */
void zjs_delay_ms(uint32_t u32Ms);

#endif
