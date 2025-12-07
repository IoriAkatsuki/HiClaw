#!/usr/bin/env python3
# 用于在板卡上修改 transformers.pytorch_utils.isin_mps_friendly，避免 mindtorch 下 CPU 不支持 torch.isin 报错。

from pathlib import Path


def main() -> None:
    path = Path("/home/HwHiAiUser/.local/lib/python3.9/site-packages/transformers/pytorch_utils.py")
    print(f"[*] 准备修改文件: {path}")
    if not path.exists():
        raise SystemExit(f"[!] 文件不存在: {path}")

    code = path.read_text(encoding="utf-8")

    old = (
        "    else:\n"
        "        # Note: don't use named arguments in `torch.isin`, see https://github.com/pytorch/pytorch/issues/126045\n"
        "        return torch.isin(elements, test_elements)\n"
    )

    if old not in code:
        raise SystemExit("[!] 未找到原始 isin_mps_friendly 代码片段，可能 transformers 版本不匹配。")

    new = """    else:
        # Note: 在 Ascend + mindtorch 上，CPU 场景下 torch.isin 未实现，这里改用 numpy 实现以避免报错。
        try:
            import numpy as np
            import torch

            elements_np = elements.detach().cpu().numpy()
            test_tensor = test_elements
            if not torch.is_tensor(test_tensor):
                test_tensor = torch.tensor(test_tensor)
            test_np = test_tensor.detach().cpu().numpy()
            mask_np = np.isin(elements_np, test_np)
            return torch.from_numpy(mask_np).to(elements.device)
        except Exception:
            # 兜底：逐元素比较，保证功能正确
            import torch

            test_tensor = test_elements
            if not torch.is_tensor(test_tensor):
                test_tensor = torch.tensor(test_tensor, device=elements.device)
            else:
                test_tensor = test_tensor.to(elements.device)
            result = torch.zeros_like(elements, dtype=torch.bool)
            for v in test_tensor.view(-1):
                result |= elements == v
            return result
"""

    patched = code.replace(old, new)
    path.write_text(patched, encoding="utf-8")
    print("[*] 已成功替换 isin_mps_friendly 的实现。")


if __name__ == "__main__":
    main()

