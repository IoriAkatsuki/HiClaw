#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STM32 振镜固件协议契约测试。"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_C = REPO_ROOT / "mirror" / "stm32f401ccu6_dac8563" / "Core" / "Src" / "main.c"


def test_main_firmware_supports_g_and_l_commands():
    source = MAIN_C.read_text(encoding="utf-8", errors="ignore")

    assert "void uart6_IDLE_callback" in source
    assert "strtok((char *)buf, \";\")" in source
    assert "token[0] >= '0' && token[0] <= '9'" in source
    assert "token[1] == 'C'" in source
    assert "token[1] == 'R'" in source
    assert "flag_update = 1;" in source
    assert "laser_on(void)" in source
    assert "laser_off(void)" in source
