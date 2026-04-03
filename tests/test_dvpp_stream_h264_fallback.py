import importlib.util
import sys
import types
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DVPP_TARGET = ROOT / "edge" / "unified_app" / "dvpp_stream.py"
MONITOR_TARGET = ROOT / "edge" / "unified_app" / "unified_monitor_mp.py"


def load_module(module_name: str, target: Path):
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop(module_name, None)
    spec.loader.exec_module(module)
    return module


class DvppStreamH264FallbackTest(unittest.TestCase):
    def test_build_ffmpeg_h264_command_targets_libx264_annexb_pipe(self):
        module = load_module("dvpp_stream_h264_fallback_test", DVPP_TARGET)

        cmd = module.build_ffmpeg_h264_command("/usr/bin/ffmpeg", 1280, 720, 25)

        self.assertEqual(cmd[0], "/usr/bin/ffmpeg")
        self.assertIn("libx264", cmd)
        self.assertIn("zerolatency", cmd)
        self.assertIn("h264", cmd)
        self.assertEqual(cmd[-1], "pipe:1")

    def test_init_h264_stream_encoder_prefers_venc(self):
        module = load_module("unified_monitor_h264_prefers_venc_test", MONITOR_TARGET)

        class DummyVenc:
            def __init__(self, width, height):
                self.size = (width, height)

        class DummyFfmpeg:
            def __init__(self, *_args, **_kwargs):
                raise AssertionError("不应走到 ffmpeg fallback")

        stream_module = types.SimpleNamespace(
            VencStreamEncoder=DummyVenc,
            FfmpegH264Encoder=DummyFfmpeg,
        )

        encoder, backend = module.init_h264_stream_encoder(640, 480, 25, stream_module=stream_module)

        self.assertIsInstance(encoder, DummyVenc)
        self.assertEqual(encoder.size, (640, 480))
        self.assertEqual(backend, "venc")

    def test_init_h264_stream_encoder_defaults_to_mjpeg_when_venc_unavailable(self):
        module = load_module("unified_monitor_h264_ffmpeg_fallback_test", MONITOR_TARGET)

        class FailingVenc:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("venc create failed")

        class DummyFfmpeg:
            def __init__(self, width, height, fps):
                self.args = (width, height, fps)

        stream_module = types.SimpleNamespace(
            VencStreamEncoder=FailingVenc,
            FfmpegH264Encoder=DummyFfmpeg,
        )

        with mock.patch.dict(module.os.environ, {}, clear=False):
            encoder, backend = module.init_h264_stream_encoder(640, 480, 25, stream_module=stream_module)

        self.assertIsNone(encoder)
        self.assertEqual(backend, "mjpeg")

    def test_init_h264_stream_encoder_uses_ffmpeg_when_explicitly_enabled(self):
        module = load_module("unified_monitor_h264_ffmpeg_enabled_test", MONITOR_TARGET)

        class FailingVenc:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("venc create failed")

        class DummyFfmpeg:
            def __init__(self, width, height, fps):
                self.args = (width, height, fps)

        stream_module = types.SimpleNamespace(
            VencStreamEncoder=FailingVenc,
            FfmpegH264Encoder=DummyFfmpeg,
        )

        with mock.patch.dict(module.os.environ, {"ICT_ENABLE_FFMPEG_H264_FALLBACK": "1"}, clear=False):
            encoder, backend = module.init_h264_stream_encoder(640, 480, 25, stream_module=stream_module)

        self.assertIsInstance(encoder, DummyFfmpeg)
        self.assertEqual(encoder.args, (640, 480, 25))
        self.assertEqual(backend, "ffmpeg")

    def test_init_h264_stream_encoder_returns_mjpeg_when_all_paths_fail(self):
        module = load_module("unified_monitor_h264_mjpeg_fallback_test", MONITOR_TARGET)

        class FailingVenc:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("venc create failed")

        class FailingFfmpeg:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("ffmpeg create failed")

        stream_module = types.SimpleNamespace(
            VencStreamEncoder=FailingVenc,
            FfmpegH264Encoder=FailingFfmpeg,
        )

        encoder, backend = module.init_h264_stream_encoder(640, 480, 25, stream_module=stream_module)

        self.assertIsNone(encoder)
        self.assertEqual(backend, "mjpeg")


if __name__ == "__main__":
    unittest.main()
