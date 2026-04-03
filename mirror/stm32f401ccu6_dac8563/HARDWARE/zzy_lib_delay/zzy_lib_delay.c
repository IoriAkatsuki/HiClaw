/**
  ******************************************************************************
  * @file           : zzy_lib_dealy.c
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

#include "zzy_lib_delay.h"

/**
 * @brief 微秒延时
 * @param vu32Us 需要延时的微秒数
 */
void zjs_delay_us(volatile uint32_t vu32Us)
{
  static int32_t s32Start = 0;
  static int32_t s32Goal = 0;
  static int32_t s32Val = 0;
  uint32_t u32Step = 0;

  while (0 != vu32Us)
  {
    u32Step = vu32Us > 900 ? 900 : vu32Us; //900为一步，大于900则分多步执行（因为系统tick为1ms，这个数不能超过1000）
    s32Start = (uint32_t)(SysTick->VAL);
    s32Goal = s32Start - ZJS_CPU_FREQUENCY_MHZ * u32Step;

    if (s32Goal >= 0)
    {
      do
      {
        s32Val = (uint32_t)(SysTick->VAL); // 向下计数
      } while ((s32Val <= s32Start) && (s32Val >= s32Goal));
    }
    else
    {
      s32Goal += ZJS_CPU_FREQUENCY_MHZ * 1000; // 系统初始化的tick是1ms（1us的tick数*1000）

      do
      {
        s32Val = (uint32_t)(SysTick->VAL); // 向下计数
      } while ((s32Val <= s32Start) || (s32Val >= s32Goal));
    }

    vu32Us -= u32Step; //延时数大于900，则分步执行
  }
}

/**
 * @brief 毫秒延时
 * @param vu32Us 需要延时的毫秒数
 */
void zjs_delay_ms(uint32_t u32Ms)
{
  while (u32Ms--)
  {
    //ms--;
    zjs_delay_us(1000);
  };
}
