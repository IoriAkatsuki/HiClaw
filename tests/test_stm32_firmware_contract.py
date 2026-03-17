#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STM32 振镜固件协议契约测试。"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_C = REPO_ROOT / "mirror" / "stm32f401ccu6_dac8563" / "Core" / "Src" / "main.c"


def test_main_firmware_supports_g_and_l_commands():
    source = MAIN_C.read_text(encoding="utf-8", errors="ignore")

    assert "uart1_rx_buf[0] == 'G'" in source
    assert "uart1_rx_buf[0] == 'L'" in source
    assert "dac8563_output_int16(x, y);" in source
    assert "HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_SET);" in source
    assert "HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_RESET);" in source
