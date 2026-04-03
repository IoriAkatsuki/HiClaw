import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

import smoke_qwen35_vl as smoke


class SmokeQwen35VlHelpersTest(unittest.TestCase):
    def test_resize_keeps_bounds(self):
        image = Image.new('RGB', (1600, 900), color='white')
        resized = smoke.resize_image_if_needed(image, 512)
        self.assertLessEqual(max(resized.size), 512)
        self.assertEqual(resized.mode, 'RGB')

    def test_build_messages_uses_single_image_and_text(self):
        messages = smoke.build_messages('请描述图片内容')
        self.assertEqual(messages[0]['role'], 'user')
        self.assertEqual(messages[0]['content'][0]['type'], 'image')
        self.assertEqual(messages[0]['content'][1]['type'], 'text')
        self.assertEqual(messages[0]['content'][1]['text'], '请描述图片内容')

    def test_build_messages_text_only_skips_image_part(self):
        messages = smoke.build_messages('只测试文本', include_image=False)
        self.assertEqual(messages[0]['role'], 'user')
        self.assertEqual(len(messages[0]['content']), 1)
        self.assertEqual(messages[0]['content'][0]['type'], 'text')
        self.assertEqual(messages[0]['content'][0]['text'], '只测试文本')

    def test_classify_runtime_issue(self):
        self.assertEqual(smoke.classify_runtime_issue(RuntimeError('OOM on device')), 'oom')
        self.assertEqual(smoke.classify_runtime_issue(RuntimeError('unsupported op Foo')), 'unsupported_op')
        self.assertEqual(smoke.classify_runtime_issue(RuntimeError('dtype mismatch: bfloat16')), 'dtype')
        self.assertEqual(
            smoke.classify_runtime_issue(RuntimeError('Input type (float) and bias type (c10::Half) should be the same')),
            'dtype',
        )
        self.assertEqual(smoke.classify_runtime_issue(RuntimeError('processor mismatch in tokenizer')), 'processor_mismatch')
        self.assertEqual(smoke.classify_runtime_issue(RuntimeError('other issue')), 'unknown')

    def test_load_processor_falls_back_when_torchvision_missing(self):
        error = ImportError(
            'AutoVideoProcessor requires the Torchvision library but it was not found in your environment.'
        )
        with mock.patch.object(smoke.AutoProcessor, 'from_pretrained', side_effect=error), \
             mock.patch.object(smoke, 'build_image_only_processor_fallback', return_value='fallback-processor') as fallback:
            processor = smoke.load_processor('/tmp/model-dir')

        self.assertEqual(processor, 'fallback-processor')
        fallback.assert_called_once_with('/tmp/model-dir')

    def test_load_processor_falls_back_when_qwen3_vl_video_processor_hits_none_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / 'preprocessor_config.json').write_text(
                '{"processor_class": "Qwen3VLProcessor", "image_processor_type": "Qwen2VLImageProcessorFast"}',
                encoding='utf-8',
            )
            error = TypeError("argument of type 'NoneType' is not iterable")
            with mock.patch.object(smoke.AutoProcessor, 'from_pretrained', side_effect=error), \
                 mock.patch.object(smoke, 'build_image_only_processor_fallback', return_value='fallback-processor') as fallback:
                processor = smoke.load_processor(str(model_dir))

        self.assertEqual(processor, 'fallback-processor')
        fallback.assert_called_once_with(str(model_dir))

    def test_build_image_only_processor_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / 'chat_template.jinja').write_text('chat-template-body', encoding='utf-8')

            fake_image_processor = object()
            fake_tokenizer = object()
            captured = {}

            def fake_qwen3_processor(**kwargs):
                captured.update(kwargs)
                return 'processor-instance'

            with mock.patch.object(smoke.AutoImageProcessor, 'from_pretrained', return_value=fake_image_processor) as image_loader, \
                 mock.patch.object(smoke.AutoTokenizer, 'from_pretrained', return_value=fake_tokenizer) as tokenizer_loader, \
                 mock.patch.object(smoke, 'Qwen3VLProcessor', side_effect=fake_qwen3_processor):
                processor = smoke.build_image_only_processor_fallback(str(model_dir))

        self.assertEqual(processor, 'processor-instance')
        image_loader.assert_called_once_with(str(model_dir), trust_remote_code=True, use_fast=False)
        tokenizer_loader.assert_called_once_with(str(model_dir), trust_remote_code=True)
        self.assertIs(captured['image_processor'], fake_image_processor)
        self.assertIs(captured['tokenizer'], fake_tokenizer)
        self.assertIsInstance(captured['video_processor'], smoke.PlaceholderVideoProcessor)
        self.assertEqual(captured['chat_template'], 'chat-template-body')

    def test_select_model_loader_falls_back_without_auto_model_for_vision2seq(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / 'config.json').write_text('{"architectures": ["OtherVisionModel"]}', encoding='utf-8')
            with mock.patch.object(smoke, 'AutoModelForVision2Seq', None):
                loader = smoke.select_model_loader(str(model_dir))

        self.assertIs(loader, smoke.AutoModelForImageTextToText)

    def test_get_patch_embed_proj_prefers_nested_visual_module(self):
        proj = object()
        model = SimpleNamespace(
            model=SimpleNamespace(
                visual=SimpleNamespace(
                    patch_embed=SimpleNamespace(proj=proj)
                )
            )
        )

        self.assertIs(smoke.get_patch_embed_proj(model), proj)

    def test_normalize_patch_embed_proj_dtype_only_aligns_bias(self):
        proj = SimpleNamespace(
            weight=smoke.torch.nn.Parameter(smoke.torch.ones((1, 1, 1, 1, 1), dtype=smoke.torch.float32)),
            bias=smoke.torch.nn.Parameter(smoke.torch.ones((1,), dtype=smoke.torch.float16)),
        )
        model = SimpleNamespace(
            model=SimpleNamespace(
                visual=SimpleNamespace(
                    patch_embed=SimpleNamespace(proj=proj)
                )
            )
        )

        probe = smoke.normalize_patch_embed_proj_dtype(model)

        self.assertEqual(proj.weight.dtype, smoke.torch.float32)
        self.assertEqual(proj.bias.dtype, smoke.torch.float32)
        self.assertEqual(probe['before']['weight_dtype'], 'torch.float32')
        self.assertEqual(probe['before']['bias_dtype'], 'torch.float16')
        self.assertEqual(probe['after']['weight_dtype'], 'torch.float32')
        self.assertEqual(probe['after']['bias_dtype'], 'torch.float32')

    def test_force_patch_embed_proj_fp16_after_to_device_aligns_weight_and_bias(self):
        proj = SimpleNamespace(
            weight=smoke.torch.nn.Parameter(smoke.torch.ones((1, 1, 1, 1, 1), dtype=smoke.torch.float32)),
            bias=smoke.torch.nn.Parameter(smoke.torch.ones((1,), dtype=smoke.torch.float32)),
        )
        model = SimpleNamespace(
            model=SimpleNamespace(
                visual=SimpleNamespace(
                    patch_embed=SimpleNamespace(proj=proj)
                )
            )
        )

        probe = smoke.force_patch_embed_proj_fp16_after_to_device(model)

        self.assertEqual(proj.weight.dtype, smoke.torch.float16)
        self.assertEqual(proj.bias.dtype, smoke.torch.float16)
        self.assertEqual(probe['before']['weight_dtype'], 'torch.float32')
        self.assertEqual(probe['before']['bias_dtype'], 'torch.float32')
        self.assertEqual(probe['after']['weight_dtype'], 'torch.float16')
        self.assertEqual(probe['after']['bias_dtype'], 'torch.float16')

    def test_move_inputs_to_device_and_dtype_by_module_only_casts_floating_tensors(self):
        module = SimpleNamespace(
            weight=smoke.torch.nn.Parameter(smoke.torch.ones((1, 1, 1, 1, 1), dtype=smoke.torch.float32))
        )
        inputs = {
            'pixel_values': smoke.torch.ones((1, 2), dtype=smoke.torch.float16),
            'input_ids': smoke.torch.ones((1, 2), dtype=smoke.torch.int64),
            'attention_mask': smoke.torch.ones((1, 2), dtype=smoke.torch.bool),
            'image_grid_thw': smoke.torch.ones((1, 3), dtype=smoke.torch.int32),
            '_prompt_seq_len': 12,
        }

        moved = smoke.move_inputs_to_device_and_dtype_by_module(inputs, module, smoke.torch.device('cpu'))

        self.assertEqual(moved['pixel_values'].dtype, smoke.torch.float32)
        self.assertEqual(moved['input_ids'].dtype, smoke.torch.int64)
        self.assertEqual(moved['attention_mask'].dtype, smoke.torch.bool)
        self.assertEqual(moved['image_grid_thw'].dtype, smoke.torch.int32)
        self.assertEqual(moved['_prompt_seq_len'], 12)

    def test_collect_tensor_debug_info_reports_dtype_shape_and_layout(self):
        base = smoke.torch.arange(10, dtype=smoke.torch.int64)
        sliced = base[2:6]
        info = smoke.collect_tensor_debug_info({
            'pixel_values': smoke.torch.zeros((2, 3), dtype=smoke.torch.float16),
            'input_ids': sliced.view(1, 4),
            '_prompt_seq_len': 8,
            'non_tensor': 'skip',
        })

        self.assertEqual(info['pixel_values']['dtype'], 'torch.float16')
        self.assertEqual(info['pixel_values']['shape'], [2, 3])
        self.assertEqual(info['pixel_values']['stride'], [3, 1])
        self.assertTrue(info['pixel_values']['is_contiguous'])
        self.assertEqual(info['pixel_values']['storage_offset'], 0)
        self.assertEqual(info['input_ids']['dtype'], 'torch.int64')
        self.assertEqual(info['input_ids']['shape'], [1, 4])
        self.assertEqual(info['input_ids']['storage_offset'], 2)
        self.assertNotIn('_prompt_seq_len', info)
        self.assertNotIn('non_tensor', info)

    def test_build_decode_attention_mask_appends_one_step(self):
        attention_mask = smoke.torch.tensor([[1, 1, 0]], dtype=smoke.torch.int64)

        decode_attention_mask = smoke.build_decode_attention_mask(attention_mask)

        self.assertEqual(tuple(decode_attention_mask.shape), (1, 4))
        self.assertTrue(smoke.torch.equal(decode_attention_mask, smoke.torch.tensor([[1, 1, 0, 1]], dtype=smoke.torch.int64)))

    def test_format_probe_stage_marker_embeds_json_payload(self):
        marker = smoke.format_probe_stage_marker('mm_decode1_prefill_done', {'has_past_key_values': True})

        self.assertIn('[probe-stage] mm_decode1_prefill_done', marker)
        self.assertIn('"has_past_key_values": true', marker)

    def test_extract_first_kv_debug_info_reads_dtype_shape_and_device(self):
        kv = ((
            smoke.torch.zeros((1, 2, 3), dtype=smoke.torch.float16),
            smoke.torch.zeros((1, 2, 3), dtype=smoke.torch.float16),
        ),)

        info = smoke.extract_first_kv_debug_info(kv)

        self.assertEqual(info['dtype'], 'torch.float16')
        self.assertEqual(info['shape'], [1, 2, 3])
        self.assertEqual(info['device'], 'cpu')

    def test_build_child_probe_command_appends_internal_flag(self):
        args = SimpleNamespace(
            model_path='/tmp/model',
            image_path='/tmp/image.jpg',
            question='测试问题',
            max_new_tokens=1,
            max_context_len=256,
            max_image_edge=128,
            failure_report='failure.md',
            metrics_json='metrics.json',
            temperature=0.0,
            top_p=1.0,
            probe='mm_prefill',
            probe_timeout=180,
            disable_cache=False,
            child_probe=False,
        )

        command = smoke.build_child_probe_command(args)

        self.assertIn('--child-probe', command)
        self.assertIn('--probe', command)
        self.assertIn('mm_prefill', command)
        self.assertIn('--max-new-tokens', command)
        self.assertIn('1', command)
        self.assertNotIn('--disable-cache', command)

    def test_build_child_probe_command_forwards_disable_cache(self):
        args = SimpleNamespace(
            model_path='/tmp/model',
            image_path='/tmp/image.jpg',
            question='测试问题',
            max_new_tokens=1,
            max_context_len=256,
            max_image_edge=128,
            failure_report='failure.md',
            metrics_json='metrics.json',
            temperature=0.0,
            top_p=1.0,
            probe='text_prefill',
            probe_timeout=180,
            disable_cache=True,
            child_probe=False,
        )

        command = smoke.build_child_probe_command(args)

        self.assertIn('--disable-cache', command)


if __name__ == '__main__':
    unittest.main()
