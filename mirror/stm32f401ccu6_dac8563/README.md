# stm32f401ccu6_dac8563 — 快速上手说明

## 项目概述
本工程驱动 DAC8563（双通道 16-bit DAC），并提供通过 SPI 输出坐标点的绘图功能（如画圆、画矩形）。工程基于 STM32F401，使用 HAL 库及部分自定义库（`zzy_lib_delay`、`zzy_lib_uart_f4` 等）。

主要源码位置：
- Core/Src/main.c — 系统初始化、任务调度、绘图逻辑
- HARDWARE/dac8563/dac8563.c/.h — DAC 驱动实现
- HARDWARE/zzy_lib_* — 延时、串口等工具

硬件约定（代码中宏定义）
- SYNC — PA4
- LDAC — PA3
- CLR — PA6
- 若使用软件 SPI（宏关闭 HW），SCLK/DIN 定义为 PA5/PA7
- 控笔/索引控制信号示例：PB10（代码中用作绘制使能）

---

## 快速使用（初始化与主流程）
在 `main()` 中典型初始化顺序：
1. HAL_Init();
2. SystemClock_Config();
3. MX_GPIO_Init(); MX_DMA_Init(); MX_USART1_UART_Init(); MX_SPI1_Init();
4. dac8563_init();

随后通过 `dac8563_output*` 系列函数发送坐标到 DAC，或使用工程提供的 `draw_circle` / `draw_rectangle` / `move` 接口执行绘图任务。

---

## API 参考（主要函数与参数说明）
> 📐 **坐标系统说明**：所有坐标与半径/尺寸均以 int16_t 表示（有效范围：-32768 ~ +32767）。坐标原点 (0,0) 对应 DAC 输出中点（内部通过 +32768 偏移转换为 uint16）。

### 1. DAC 底层驱动函数

#### `void dac8563_init(void)`
**功能**：初始化 DAC8563 芯片
- 使能 SPI 接口
- 复位所有 DAC 寄存器
- 上电 DAC A 和 B 通道
- 使能内部参考电压
- 设置增益为 2（输出范围 0-5V）

**调用时机**：系统初始化阶段调用一次

**示例**：
```c
dac8563_init();  // 在 main() 的 USER CODE BEGIN 2 区域调用
```

---

#### `void dac8563_write(uint8_t cmd, uint16_t data)`
**功能**：向 DAC 写入命令和数据（底层函数）

**参数**：
- `cmd`：8-bit 命令字节（如 `CMD_SETA_UPDATEA`、`CMD_SETB_UPDATEB` 等，详见 `dac8563.h`）
- `data`：16-bit 数据或参数

**调用场景**：一般由上层函数调用，除非需要直接控制 DAC 寄存器

---

#### `void dac8563_output(uint16_t data_a, uint16_t data_b)`
**功能**：直接输出两个通道的无符号 16-bit 值

**参数**：
- `data_a`：DAC A 通道输出值（0~65535）
- `data_b`：DAC B 通道输出值（0~65535）

**用途**：当需要直接控制 DAC 原始输出时使用

**示例**：
```c
dac8563_output(32768, 32768);  // 两个通道输出中点电压
dac8563_output(0, 65535);      // A 通道最小值，B 通道最大值
```

---

#### `void dac8563_output_int16(int16_t data_a, int16_t data_b)` ⭐
**功能**：输出有符号坐标（最常用接口）

**参数**：
- `data_a`：X 坐标（-32768~32767，0 对应中点）
- `data_b`：Y 坐标（-32768~32767，0 对应中点）

**转换规则**：内部自动将 int16 转为 uint16（加 32768 偏移）

**示例**：
```c
dac8563_output_int16(0, 0);        // 输出到原点（中心）
dac8563_output_int16(1000, -500);  // 输出到 (1000, -500)
dac8563_output_int16(-10000, 5000); // 输出到 (-10000, 5000)
```

---

#### `void dac8563_output_float(float data_a, float data_b)`
**功能**：以浮点数形式输出（用于电压映射）

**参数**：
- `data_a`、`data_b`：浮点值

**映射公式**：`output = (data / 20.0f) * 65535`

**示例**：
```c
dac8563_output_float(0.0f, 0.0f);    // 输出 0V
dac8563_output_float(10.0f, -5.0f);  // 根据公式映射电压
```

---

### 2. 绘图功能函数

#### `void draw_circle(int16_t x, int16_t y, uint16_t radius)` ⭐
**功能**：绘制圆形

**参数**：
- `x`：圆心 X 坐标（int16）
- `y`：圆心 Y 坐标（int16）
- `radius`：半径（uint16）

**算法**：使用 90 个点参数化圆形轨迹（通过三角函数计算）

**性能参数**：
- 点数：90（可在函数内修改循环条件提高平滑度）
- 单点延时：20μs（可调整 `zjs_delay_us(20)` 改变速度）

**示例**：
```c
draw_circle(0, 0, 1000);       // 在原点画半径 1000 的圆
draw_circle(5000, 5000, 2000); // 在 (5000,5000) 画半径 2000 的圆
draw_circle(-3000, 1000, 500); // 在 (-3000,1000) 画半径 500 的圆
```

**优化建议**：
- 增加点数（如 180、360）可提升平滑度：`for (i = 0; i < 360; ++i)`
- 调整延时改变速度：`zjs_delay_us(10)` 加快，`zjs_delay_us(50)` 减慢

---

#### `void draw_rectangle(int16_t x, int16_t y, uint16_t length, uint16_t height)`
**功能**：绘制矩形边框

**参数**：
- `x`：矩形中心 X 坐标（int16）
- `y`：矩形中心 Y 坐标（int16）
- `length`：水平边长（uint16）
- `height`：竖直边长（uint16）

**算法**：按步长 100 依次绘制四条边

**注意**：当前实现为采样点绘制，非连续线条。如需连续线条，减小步长（如改为 10）

**示例**：
```c
draw_rectangle(0, 0, 2000, 1000);  // 在原点画 2000×1000 的矩形
draw_rectangle(3000, -2000, 1500, 800); // 在 (3000,-2000) 画矩形
```

---

#### `void move(int16_t x, int16_t y, uint16_t delay_us)`
**功能**：移动到指定位置（带抬笔/落笔控制）

**参数**：
- `x`：目标 X 坐标（int16）
- `y`：目标 Y 坐标（int16）
- `delay_us`：到位后稳定延时（微秒）

**行为流程**：
1. 延时 400μs（准备）
2. 拉低 PB10（落笔/使能）
3. 输出坐标到 DAC
4. 延时 delay_us（等待机械稳定）
5. 拉高 PB10（抬笔/禁止）

**用途**：在绘制图形前先移动到起点位置，避免绘制移动轨迹

**示例**：
```c
move(5000, 5000, 1500);  // 移动到 (5000,5000) 并稳定 1.5ms
move(0, 0, 1000);         // 移动到原点并稳定 1ms
```

---

### 3. 任务管理函数

#### `pose_t get_next_pose(task_t *task)`
**功能**：计算任务的起始位置（用于 move 函数）

**参数**：
- `task`：指向任务结构的指针

**返回值**：调整后的起始坐标（pose_t 结构）

**逻辑**：
- 对于 CIRCLE：返回圆心右侧点（x + radius, y）
- 对于 RECTANGLE：返回矩形右上角（x + length/2, y + height/2）

**示例**：
```c
task_t my_task;
my_task.type = CIRCLE;
my_task.pose.x = 1000;
my_task.pose.y = 2000;
my_task.params[0] = 500;

pose_t start_pos = get_next_pose(&my_task);  // 返回 (1500, 2000)
move(start_pos.x, start_pos.y, 1000);
```

---

#### `void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t Size)`
**功能**：串口 DMA 接收完成回调（支持动态任务更新）

**命令格式**：

| 命令 | 格式 | 说明 | 示例 |
|------|------|------|------|
| 更新 | `U` | 触发任务缓冲更新 | `U` |
| 圆形 | `C<idx><x>,<y>,<radius>` | 设置圆形任务 | `C0 1000,2000,500` |
| 矩形 | `R<idx><x>,<y>,<length>,<height>` | 设置矩形任务 | `R1-500,1000,2000,1500` |

**参数说明**：
- `idx`：任务索引（单字节 ASCII，'0'~'9'）
- `x,y`：中心坐标（int16，支持负数）
- `radius`/`length`/`height`：尺寸参数（uint16）

**工作流程**：
1. 接收串口命令并解析到 `task_buf_1`
2. 收到 'U' 命令后设置 `flag_update = 1`
3. 主循环检测到标志后，将 `task_buf_1` 复制到 `task_buf`
4. 清空 `task_buf_1` 和更新标志

**使用示例（通过串口发送）**：
```
C01000,2000,500    → 设置索引0为圆：圆心(1000,2000)，半径500
R1-500,0,3000,1500  → 设置索引1为矩形：中心(-500,0)，尺寸3000×1500
U                   → 触发更新，使新任务生效
```

---

### 4. 数据结构定义

#### `type_t` 枚举（任务类型）
```c
typedef enum {
    NONE,       // 空任务（用于标记任务列表结束）
    CIRCLE,     // 圆形
    RECTANGLE   // 矩形
} type_t;
```

#### `pose_t` 结构（位置坐标）
```c
typedef struct {
    int16_t x;  // X 坐标
    int16_t y;  // Y 坐标
} pose_t;
```

#### `task_t` 结构（任务描述）
```c
typedef struct {
    type_t type;         // 任务类型
    pose_t pose;         // 中心坐标
    uint16_t params[4];  // 参数数组
} task_t;
```

**参数数组用途**：
- `CIRCLE`：`params[0]` = 半径
- `RECTANGLE`：`params[0]` = 长度，`params[1]` = 高度
- `params[2]`、`params[3]`：预留扩展

---

## 使用示例

### 示例 1：直接绘制单个图形
```c
int main(void) {
    // ...初始化代码...
    dac8563_init();
    
    // 绘制一个圆
    draw_circle(0, 0, 1000);  // 在原点画半径1000的圆
    
    // 绘制一个矩形
    draw_rectangle(5000, 3000, 2000, 1500);  // 在(5000,3000)画2000×1500的矩形
    
    while (1) {
        // 主循环
    }
}
```

### 示例 2：使用任务缓冲执行多个图形
```c
int main(void) {
    // ...初始化代码...
    dac8563_init();
    
    // 配置任务列表
    task_buf[0].type = CIRCLE;
    task_buf[0].pose.x = 5000;
    task_buf[0].pose.y = 5000;
    task_buf[0].params[0] = 1000;  // 半径1000
    
    task_buf[1].type = CIRCLE;
    task_buf[1].pose.x = 0;
    task_buf[1].pose.y = 5000;
    task_buf[1].params[0] = 1000;
    
    task_buf[2].type = RECTANGLE;
    task_buf[2].pose.x = 0;
    task_buf[2].pose.y = 0;
    task_buf[2].params[0] = 2000;  // 长度
    task_buf[2].params[1] = 1000;  // 高度
    
    task_buf[3].type = NONE;  // 结束标记
    
    while (1) {
        // 主循环会自动遍历任务列表并执行
        task_t *current_task;
        for (current_task = &task_buf[0]; current_task->type != NONE; ++current_task) {
            pose_t next_pose = get_next_pose(current_task);
            
            switch (current_task->type) {
                case CIRCLE:
                    move(next_pose.x, next_pose.y, 1500);
                    draw_circle(current_task->pose.x, current_task->pose.y, 
                               current_task->params[0]);
                    break;
                    
                case RECTANGLE:
                    move(next_pose.x, next_pose.y, 1500);
                    draw_rectangle(current_task->pose.x, current_task->pose.y,
                                  current_task->params[0], current_task->params[1]);
                    break;
            }
        }
    }
}
```

### 示例 3：通过串口动态更新任务

**Python 发送脚本示例**：
```python
import serial

ser = serial.Serial('COM3', 115200, timeout=1)  # 根据实际端口修改

# 设置索引0为圆形任务
ser.write(b'C01000,2000,500\n')  # 圆心(1000,2000)，半径500

# 设置索引1为矩形任务
ser.write(b'R1-500,0,3000,1500\n')  # 中心(-500,0)，尺寸3000×1500

# 设置索引2为另一个圆
ser.write(b'C2-3000,-2000,800\n')

# 触发更新，使新任务生效
ser.write(b'U\n')

ser.close()
```

**串口调试助手发送示例**：
```
C01000,2000,500      # 设置任务0：圆形
R10,0,2000,1000      # 设置任务1：矩形
C2-5000,5000,1500    # 设置任务2：圆形
U                    # 触发更新
```

### 示例 4：单点坐标输出
```c
// 直接控制 DAC 输出到指定坐标
dac8563_output_int16(0, 0);          // 输出到原点
dac8563_output_int16(10000, -5000);  // 输出到(10000, -5000)

// 或使用无符号值（原始 DAC 值）
dac8563_output(32768, 32768);  // 中点
dac8563_output(0, 65535);      // 左下角到右上角
```

---

## 常见注意事项与优化建议

### ⚙️ 硬件配置
- **SPI 模式**：默认使用硬件 SPI（`DAC8563_USE_SPI_HW = 1`），确保 STM32CubeMX 已正确配置 SPI1
  - 如需使用软件 SPI，在 `dac8563.h` 中修改宏定义为 `#define DAC8563_USE_SPI_HW 0`
- **引脚连接**：
  ```
  STM32          DAC8563
  PA4 (SYNC)  -> SYNC
  PA3 (LDAC)  -> LDAC
  PA6 (CLR)   -> CLR
  PA5 (SCK)   -> SCLK
  PA7 (MOSI)  -> DIN
  ```
- **控制信号**：PB10 用作绘制使能（可根据实际硬件修改）

### 📐 坐标系统
- **坐标范围**：-32768 ~ +32767（int16_t）
- **原点位置**：(0, 0) 对应 DAC 输出中点（约 2.5V，假设 Vref=5V，增益=2）
- **转换公式**：`DAC_value = coordinate + 32768`
- **电压输出**：`Vout = (DAC_value / 65535) × Vref × Gain`

### 🎨 绘图质量优化
1. **提升圆的平滑度**（修改 `draw_circle` 函数）：
   ```c
   for (int i = 0; i < 360; ++i)  // 从90改为360
   {
       float theta = (2.0f * PI * i) / 360;  // 同步修改
       // ...
   }
   ```

2. **矩形连续线条**（修改 `draw_rectangle` 函数）：
   ```c
   for (i = 0; i < length; i += 10)  // 从100改为10，步长更小
   {
       // ...
       zjs_delay_us(10);  // 减小延时以保持速度
   }
   ```

3. **速度调整**：
   - 加快：减小 `zjs_delay_us()` 参数值
   - 减慢：增大 `zjs_delay_us()` 参数值
   - 实时控制：可通过串口下发速度参数

### ⚡ 性能优化
- **减少延时开销**：使用 DMA + 定时器代替软件延时实现恒速插补
- **批量坐标输出**：预计算轨迹点存入数组，批量发送
- **浮点运算优化**：将三角函数查表（参考已有的 `SineWave_Value` 数组）

### 🐛 调试建议
1. **验证坐标输出**：
   ```c
   // 输出四个角点测试
   dac8563_output_int16(-32768, -32768);  // 左下
   HAL_Delay(1000);
   dac8563_output_int16(32767, -32768);   // 右下
   HAL_Delay(1000);
   dac8563_output_int16(32767, 32767);    // 右上
   HAL_Delay(1000);
   dac8563_output_int16(-32768, 32767);   // 左上
   ```

2. **串口回显**：在 `HAL_UARTEx_RxEventCallback` 中添加确认消息
3. **LED 指示**：在绘图函数中切换 GPIO 指示运行状态

### 📊 任务管理最佳实践
- **任务数量**：`task_buf` 数组大小为 10，最多支持 9 个有效任务（最后一个必须是 `NONE`）
- **双缓冲机制**：避免在执行过程中直接修改 `task_buf`，应先写入 `task_buf_1`，然后通过 'U' 命令触发更新
- **清空任务**：通过串口发送多个 `NONE` 类型任务覆盖旧数据

### 🔧 扩展功能建议
1. **添加新图形**：
   - 定义新的 `type_t` 枚举值（如 `ELLIPSE`、`LINE`）
   - 实现对应的绘图函数
   - 在主循环 switch 中添加 case 分支

2. **参数化控制**：
   - 添加速度参数：`task_buf[i].params[2] = speed`
   - 添加重复次数：`task_buf[i].params[3] = repeat_count`

3. **路径规划**：
   - 实现最短路径算法优化移动顺序
   - 添加避障逻辑

---

## 📞 技术支持
如需进一步帮助，请参考：
- DAC8563 数据手册：Texas Instruments 官网
- STM32F401 参考手册：STMicroelectronics 官网
- HAL 库文档：STM32CubeF4 包中的 Documentation 文件夹

---

## 📝 版本历史
- **v1.0** (2025-12-04)：初始版本，支持圆形和矩形绘制，串口动态任务更新
