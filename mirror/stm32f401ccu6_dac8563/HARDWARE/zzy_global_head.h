#ifndef __ZZY_GLOBAL_HEAD_H
#define __ZZY_GLOBAL_HEAD_H

#include "main.h"

#define GPIO_FAST_SETBIT(port, bit)  GPIO##port##->BSRR |= 0x00000001 << bit
#define GPIO_FAST_RESETBIT(port, bit)  GPIO##port##->BSRR |= 0x00000001 << (bit + 16)

extern SPI_HandleTypeDef hspi1;

#endif

