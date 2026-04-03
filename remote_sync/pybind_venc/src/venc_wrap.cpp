#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <atomic>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/dvpp/hi_dvpp_venc.h"
#include "acl/dvpp/hi_media_common.h"
#include "acl/dvpp/hi_mpi_sys.h"

namespace py = pybind11;

static std::once_flag s_sys_init_flag;
static std::atomic<int> s_next_chn{0};

static void ensure_sys_init() {
    std::call_once(s_sys_init_flag, []() {
        hi_s32 ret = hi_mpi_sys_init();
        if (ret != HI_SUCCESS) {
            throw std::runtime_error("hi_mpi_sys_init failed: " + std::to_string(ret));
        }
    });
}

static inline uint32_t align_up(uint32_t x, uint32_t a) {
    return ((x + a - 1) / a) * a;
}

class PyVencSession {
public:
    PyVencSession(int width, int height, std::string codec = "H264_MAIN",
                  int max_width = 0, int max_height = 0,
                  int bitrate_kbps = 2000, int fps = 20) {
        ensure_sys_init();

        width_ = width;
        height_ = height;
        chn_ = s_next_chn.fetch_add(1);

        // Stride aligned to DEFAULT_ALIGN (32), height to 2
        stride_ = align_up((uint32_t)width_, 32);
        uint32_t h_aligned = align_up((uint32_t)height_, 2);
        y_size_ = stride_ * h_aligned;
        buf_size_ = y_size_ + y_size_ / 2;

        // Channel attributes
        hi_venc_chn_attr chn_attr;
        memset(&chn_attr, 0, sizeof(chn_attr));

        chn_attr.venc_attr.type = HI_PT_H264;
        chn_attr.venc_attr.max_pic_width  = (max_width  > 0) ? (uint32_t)max_width  : (uint32_t)width_;
        chn_attr.venc_attr.max_pic_height = (max_height > 0) ? (uint32_t)max_height : (uint32_t)height_;
        chn_attr.venc_attr.pic_width      = (uint32_t)width_;
        chn_attr.venc_attr.pic_height     = (uint32_t)height_;
        chn_attr.venc_attr.buf_size       = 4 * 1024 * 1024;
        chn_attr.venc_attr.profile        = 1; // Main Profile
        chn_attr.venc_attr.is_by_frame    = HI_TRUE;
        chn_attr.venc_attr.h264_attr.rcn_ref_share_buf_en = HI_FALSE;

        chn_attr.rc_attr.rc_mode               = HI_VENC_RC_MODE_H264_CBR;
        chn_attr.rc_attr.h264_cbr.gop          = (uint32_t)(fps * 2);
        chn_attr.rc_attr.h264_cbr.stats_time   = 1;
        chn_attr.rc_attr.h264_cbr.src_frame_rate = (uint32_t)fps;
        chn_attr.rc_attr.h264_cbr.dst_frame_rate = (uint32_t)fps;
        chn_attr.rc_attr.h264_cbr.bit_rate       = (uint32_t)bitrate_kbps;

        chn_attr.gop_attr.gop_mode           = HI_VENC_GOP_MODE_NORMAL_P;
        chn_attr.gop_attr.normal_p.ip_qp_delta = 3;

        hi_s32 ret = hi_mpi_venc_create_chn(chn_, &chn_attr);
        if (ret != HI_SUCCESS) {
            throw std::runtime_error("hi_mpi_venc_create_chn failed: " + std::to_string(ret));
        }

        hi_venc_start_param recv_param;
        recv_param.recv_pic_num = -1;
        ret = hi_mpi_venc_start_chn(chn_, &recv_param);
        if (ret != HI_SUCCESS) {
            hi_mpi_venc_destroy_chn(chn_);
            throw std::runtime_error("hi_mpi_venc_start_chn failed: " + std::to_string(ret));
        }

        // Allocate DVPP device buffer for NV12 frames
        void* ptr = nullptr;
        ret = hi_mpi_dvpp_malloc(0, &ptr, buf_size_);
        if (ret != HI_SUCCESS || !ptr) {
            hi_mpi_venc_stop_chn(chn_);
            hi_mpi_venc_destroy_chn(chn_);
            throw std::runtime_error("hi_mpi_dvpp_malloc failed: " + std::to_string(ret));
        }
        dev_buf_ = ptr;

        // Pre-fill static fields of frame_info (updated per-frame in encode_nv12)
        memset(&frame_info_, 0, sizeof(frame_info_));
        frame_info_.pool_id                        = 0;
        frame_info_.mod_id                         = HI_ID_VENC;
        frame_info_.v_frame.width                  = (uint32_t)width_;
        frame_info_.v_frame.height                 = (uint32_t)height_;
        frame_info_.v_frame.pixel_format           = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
        frame_info_.v_frame.video_format           = HI_VIDEO_FORMAT_LINEAR;
        frame_info_.v_frame.compress_mode          = HI_COMPRESS_MODE_NONE;
        frame_info_.v_frame.dynamic_range          = HI_DYNAMIC_RANGE_SDR8;
        frame_info_.v_frame.color_gamut            = HI_COLOR_GAMUT_BT709;
        frame_info_.v_frame.field                  = HI_VIDEO_FIELD_FRAME;
        frame_info_.v_frame.width_stride[0]        = stride_;
        frame_info_.v_frame.width_stride[1]        = stride_;
        frame_info_.v_frame.virt_addr[0]           = dev_buf_;
        frame_info_.v_frame.virt_addr[1]           = (void*)((uintptr_t)dev_buf_ + y_size_);
    }

    py::bytes encode_nv12(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> nv12) {
        auto info = nv12.request();
        size_t expect = (size_t)width_ * height_ * 3 / 2;
        if ((size_t)info.size < expect) {
            throw std::runtime_error("nv12 too small: got " + std::to_string(info.size)
                                     + " need " + std::to_string(expect));
        }

        const uint8_t* src = static_cast<const uint8_t*>(info.ptr);
        uint8_t* dst_y  = static_cast<uint8_t*>(dev_buf_);
        uint8_t* dst_uv = static_cast<uint8_t*>(dev_buf_) + y_size_;

        if (stride_ == (uint32_t)width_) {
            memcpy(dst_y,  src,                    (size_t)width_ * height_);
            memcpy(dst_uv, src + width_ * height_, (size_t)width_ * height_ / 2);
        } else {
            for (int r = 0; r < height_; r++)
                memcpy(dst_y  + (size_t)r * stride_, src + (size_t)r * width_, (size_t)width_);
            const uint8_t* src_uv = src + (size_t)width_ * height_;
            for (int r = 0; r < height_ / 2; r++)
                memcpy(dst_uv + (size_t)r * stride_, src_uv + (size_t)r * width_, (size_t)width_);
        }

        hi_s32 ret = hi_mpi_venc_send_frame(chn_, &frame_info_, 0);
        if (ret != HI_SUCCESS) {
            throw std::runtime_error("hi_mpi_venc_send_frame failed: " + std::to_string(ret));
        }

        hi_venc_chn_status status;
        memset(&status, 0, sizeof(status));
        ret = hi_mpi_venc_query_status(chn_, &status);
        uint32_t pack_cnt = (ret == HI_SUCCESS && status.cur_packs > 0) ? status.cur_packs : 1;

        hi_venc_stream stream;
        memset(&stream, 0, sizeof(stream));
        stream.pack_cnt = pack_cnt;
        stream.pack = new hi_venc_pack[pack_cnt]();

        ret = hi_mpi_venc_get_stream(chn_, &stream, 2000);
        if (ret != HI_SUCCESS) {
            delete[] stream.pack;
            throw std::runtime_error("hi_mpi_venc_get_stream failed: " + std::to_string(ret));
        }

        std::vector<uint8_t> out;
        for (uint32_t i = 0; i < stream.pack_cnt; i++) {
            uint32_t offset = stream.pack[i].offset;
            uint32_t len    = stream.pack[i].len;
            if (len > offset)
                out.insert(out.end(), stream.pack[i].addr + offset,
                                      stream.pack[i].addr + len);
        }

        hi_mpi_venc_release_stream(chn_, &stream);
        delete[] stream.pack;

        return py::bytes(reinterpret_cast<char*>(out.data()), out.size());
    }

    void close() {
        if (closed_) return;
        closed_ = true;
        if (dev_buf_) {
            hi_mpi_dvpp_free(dev_buf_);
            dev_buf_ = nullptr;
        }
        hi_mpi_venc_stop_chn(chn_);
        hi_mpi_venc_destroy_chn(chn_);
    }

    ~PyVencSession() { close(); }

private:
    int      chn_      = 0;
    int      width_    = 0;
    int      height_   = 0;
    uint32_t stride_   = 0;
    uint32_t y_size_   = 0;
    uint32_t buf_size_ = 0;
    void*    dev_buf_  = nullptr;
    hi_video_frame_info frame_info_{};
    bool     closed_   = false;
};

PYBIND11_MODULE(venc_wrapper, m) {
    py::class_<PyVencSession>(m, "VencSession")
        .def(py::init<int, int, std::string, int, int, int, int>(),
             py::arg("width"), py::arg("height"),
             py::arg("codec")        = "H264_MAIN",
             py::arg("max_width")    = 0,
             py::arg("max_height")   = 0,
             py::arg("bitrate_kbps") = 2000,
             py::arg("fps")          = 20)
        .def("encode_nv12", &PyVencSession::encode_nv12)
        .def("close", &PyVencSession::close);
}
