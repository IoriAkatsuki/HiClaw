#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STM32 振镜固件协议契约测试。"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_C = REPO_ROOT / "mirror" / "stm32f401ccu6_dac8563" / "Core" / "Src" / "main.c"


def test_main_firmware_supports_g_and_l_commands():
    source = MAIN_C.read_text(encoding="utf-8", errors="ignore")

    assert "void uart6_IDLE_callback" in source
    assert "token[0] >= '0' && token[0] <= '9'" in source
    assert "token[1] == 'C'" in source
    assert "token[1] == 'R'" in source
    assert "flag_update = 1;" in source
    assert "laser_on(void)" in source
    assert "laser_off(void)" in source


def test_main_firmware_starts_in_safe_idle_state():
    source = MAIN_C.read_text(encoding="utf-8", errors="ignore")

    assert "laser_off();" in source
    assert "task_buf[0].type = CIRCLE;" not in source
    assert "task_buf[0].pose.x = 5000;" not in source
    assert "task_buf[0].params[0] = 10000;" not in source


def test_main_firmware_appends_uart6_packets_instead_of_overwriting_pending_buffer():
    source = MAIN_C.read_text(encoding="utf-8", errors="ignore")

    assert "available = (UART6_RX_BUF_SIZE - 1) - uart6_packet_len;" in source
    assert "memcpy(&uart6_pending_buf[uart6_packet_len], data, copy_len);" in source
    assert "memcpy(uart6_pending_buf, data, copy_len);" not in source
