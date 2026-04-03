/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "spi.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "zzy_lib_delay.h"
#include "zzy_lib_uart_f4.h"
#include "dac8563.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef enum
{
  NONE,
  CIRCLE,
  RECTANGLE
} type_t;

typedef struct
{
  uint8_t active;
  uint8_t n;
  int16_t v[16][2];
} PolygonSlot;

typedef struct
{
  int16_t x, y;
  uint8_t blank;
} DisplayPoint;

typedef struct
{
  int16_t x;
  int16_t y;
} pose_t;


typedef struct
{
  type_t type;
  pose_t pose;
  uint16_t params[4];
} task_t;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define SPEEED 2.0f
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
const float PI = 3.14159265358979323846f;

task_t task_buf[10];
task_t task_buf_1[10];
volatile uint8_t flag_update = 0;
volatile uint8_t uart6_packet_ready = 0;
volatile uint16_t uart6_packet_len = 0;
uint8_t uart6_pending_buf[UART6_RX_BUF_SIZE];

int16_t current_x = 0;
int16_t current_y = 0;
uint8_t current_laser_state = 2; // 0=off, 1=on, 2=unknown

PolygonSlot polygon_slots[10];
PolygonSlot polygon_slots_1[10];

#define MAX_DL_POINTS 512
DisplayPoint display_list[MAX_DL_POINTS];
volatile uint16_t dl_count = 0;
volatile uint16_t dl_expected = 0;
volatile uint8_t dl_mode = 0; // 0=normal, 1=uploading, 2=scanning
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
void laser_on(void)
{
  if (current_laser_state == 1) return;
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_SET);
  current_laser_state = 1;
}

void laser_off(void)
{
  if (current_laser_state == 0) return;
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_RESET);
  current_laser_state = 0;
}

void move(int16_t x, int16_t y)
{
  if (x == current_x && y == current_y) return;

  float dx = x - current_x;
  float dy = y - current_y;
  float distance = sqrtf(dx * dx + dy * dy);
  int steps = (int)(distance / 100.0f) + 1;
  float step_dist = distance / steps;
  uint32_t delay = (uint32_t)(step_dist * SPEEED);
  if (delay == 0) delay = 1;

  int16_t start_x = current_x;
  int16_t start_y = current_y;

  for (int i = 1; i <= steps; ++i)
  {
    int16_t px = start_x + (int16_t)(dx * i / steps);
    int16_t py = start_y + (int16_t)(dy * i / steps);
    dac8563_output_int16(px, py);
    zjs_delay_us(delay);
  }
  current_x = x;
  current_y = y;
}

void draw_circle(int16_t x, int16_t y, uint16_t radius)
{
  laser_off();
  move(x + radius, y);
  laser_on();

  for (int i = 1; i <= 36; ++i)
  {
    float theta = (2.0f * PI * i) / 36.0f;
    int16_t target_x = (int16_t)(radius * cosf(theta) + x);
    int16_t target_y = (int16_t)(radius * sinf(theta) + y);
    
    move(target_x, target_y);
  }
  laser_off();
}

void draw_rectangle(int16_t x, int16_t y, uint16_t length, uint16_t height)
{
  int16_t p1_x = x + length / 2;
  int16_t p1_y = y + height / 2;
  
  int16_t p2_x = x - length / 2;
  int16_t p2_y = p1_y;
  
  int16_t p3_x = p2_x;
  int16_t p3_y = y - height / 2;
  
  int16_t p4_x = p1_x;
  int16_t p4_y = p3_y;
  
  laser_off();
  move(p1_x, p1_y);
  laser_on();

  move(p2_x, p2_y);
  move(p3_x, p3_y);
  move(p4_x, p4_y);
  move(p1_x, p1_y);

  laser_off();
}

void draw_polygon_slot(uint8_t slot)
{
  PolygonSlot *ps = &polygon_slots[slot];
  if (!ps->active || ps->n < 3) return;
  laser_off();
  move(ps->v[0][0], ps->v[0][1]);
  laser_on();
  for (uint8_t i = 1; i < ps->n; i++)
    move(ps->v[i][0], ps->v[i][1]);
  move(ps->v[0][0], ps->v[0][1]);
  laser_off();
}

void uart1_IDLE_callback(uint8_t *data, uint16_t num)
{
  if (num == 0) return;
  if (num >= UART1_RX_BUF_SIZE) num = UART1_RX_BUF_SIZE - 1;
  data[num] = '\0';
  char *buf = (char *)data;

  if (buf[0] == 'U')
  {
    flag_update = 1;
  }
  else if (buf[0] == 'G')
  {
    int16_t x, y;
    if (sscanf(&buf[1], "%hd,%hd", &x, &y) == 2)
    {
      dac8563_output_int16(x, y);
      current_x = x;
      current_y = y;
    }
  }
  else if (buf[0] == 'L')
  {
    if (buf[1] == '1') laser_on();
    else if (buf[1] == '0') laser_off();
  }
  else if (buf[0] == 'R')
  {
    int slot = buf[1] - '0';
    if (slot >= 0 && slot <= 9)
    {
      int16_t x, y;
      uint16_t length, height;
      if (sscanf(&buf[2], " %hd,%hd,%hu,%hu", &x, &y, &length, &height) == 4)
      {
        task_buf_1[slot].type = RECTANGLE;
        task_buf_1[slot].pose.x = x;
        task_buf_1[slot].pose.y = y;
        task_buf_1[slot].params[0] = length;
        task_buf_1[slot].params[1] = height;
      }
    }
  }
  else if (buf[0] == 'C')
  {
    int slot = buf[1] - '0';
    if (slot >= 0 && slot <= 9)
    {
      int16_t x, y;
      uint16_t radius;
      if (sscanf(&buf[2], " %hd,%hd,%hu", &x, &y, &radius) == 3)
      {
        task_buf_1[slot].type = CIRCLE;
        task_buf_1[slot].pose.x = x;
        task_buf_1[slot].pose.y = y;
        task_buf_1[slot].params[0] = radius;
      }
    }
  }
  else if (buf[0] == 'P')
  {
    int slot = buf[1] - '0';
    if (slot < 0 || slot > 9) return;
    int n = 0;
    int offset = 0;
    sscanf(&buf[2], " %d %n", &n, &offset);
    if (n == 0)
    {
      polygon_slots_1[slot].active = 0;
      polygon_slots_1[slot].n = 0;
      return;
    }
    if (n < 3 || n > 16) return;
    char *p = &buf[2] + offset;
    for (int k = 0; k < n; k++)
    {
      int16_t vx, vy;
      int consumed = 0;
      if (sscanf(p, "%hd,%hd%n", &vx, &vy, &consumed) < 2) break;
      polygon_slots_1[slot].v[k][0] = vx;
      polygon_slots_1[slot].v[k][1] = vy;
      p += consumed;
      if (*p == ',') p++;
    }
    polygon_slots_1[slot].n = (uint8_t)n;
    polygon_slots_1[slot].active = 1;
  }
  else if (buf[0] == 'D')
  {
    if (buf[1] == 'S')
    {
      dl_mode = 2;
    }
    else if (buf[1] == 'X')
    {
      dl_mode = 0;
      laser_off();
    }
    else if (buf[1] == '+')
    {
      if (dl_mode == 1 && dl_count < MAX_DL_POINTS)
      {
        int16_t x, y;
        int b;
        if (sscanf(&buf[2], " %hd,%hd,%d", &x, &y, &b) == 3)
        {
          display_list[dl_count].x = x;
          display_list[dl_count].y = y;
          display_list[dl_count].blank = (uint8_t)(b != 0);
          dl_count++;
        }
      }
    }
    else if (buf[1] >= '0' && buf[1] <= '9')
    {
      uint16_t n = 0;
      sscanf(&buf[1], "%hu", &n);
      if (n > MAX_DL_POINTS) n = MAX_DL_POINTS;
      dl_expected = n;
      dl_count = 0;
      dl_mode = 1;
    }
  }
}

pose_t get_next_pose(task_t *task)
{
  pose_t pose = task->pose;
  switch (task->type)
  {
  case CIRCLE:
    pose.x += task->params[0];
    break;

  case RECTANGLE:
    pose.x += task->params[0] / 2;
    pose.y += task->params[1] / 2;
    break;
  
  default:
    break;
  }

  return pose;
}

void apply_uart6_token(char *token)
{
  if (token == NULL || token[0] == '\0')
  {
    return;
  }

  if (token[0] == 'U')
  {
    flag_update = 1;
  }
  else if (token[0] >= '0' && token[0] <= '9')
  {
    int i = token[0] - '0';

    if (token[1] == 'C')
    {
      int16_t x;
      int16_t y;
      uint16_t radius;
      if (sscanf(&token[2], ",%hd,%hd,%hu", &x, &y, &radius) == 3)
      {
        task_buf_1[i].type = CIRCLE;
        task_buf_1[i].pose.x = x;
        task_buf_1[i].pose.y = y;
        task_buf_1[i].params[0] = radius;
      }
    }
    else if (token[1] == 'R')
    {
      int16_t x;
      int16_t y;
      uint16_t length;
      uint16_t height;
      if (sscanf(&token[2], ",%hd,%hd,%hu,%hu", &x, &y, &length, &height) == 4)
      {
        task_buf_1[i].type = RECTANGLE;
        task_buf_1[i].pose.x = x;
        task_buf_1[i].pose.y = y;
        task_buf_1[i].params[0] = length;
        task_buf_1[i].params[1] = height;
      }
    }
  }
}

void process_uart6_packet(void)
{
  uint8_t local_buf[UART6_RX_BUF_SIZE];
  uint16_t local_len = 0;
  char *token = NULL;

  __disable_irq();
  if (!uart6_packet_ready)
  {
    __enable_irq();
    return;
  }
  local_len = uart6_packet_len;
  memcpy(local_buf, uart6_pending_buf, local_len + 1);
  uart6_packet_ready = 0;
  uart6_packet_len = 0;
  __enable_irq();

  token = strtok((char *)local_buf, ";");
  while (token != NULL)
  {
    apply_uart6_token(token);
    token = strtok(NULL, ";");
  }
}

void uart6_IDLE_callback(uint8_t *data, uint16_t num)
{
  uint16_t available = 0;
  uint16_t copy_len = num;

  __disable_irq();
  available = (UART6_RX_BUF_SIZE - 1) - uart6_packet_len;
  __enable_irq();

  if (copy_len > available)
  {
    copy_len = available;
  }
  if (copy_len >= UART6_RX_BUF_SIZE)
  {
    copy_len = UART6_RX_BUF_SIZE - 1;
  }

  __disable_irq();
  memcpy(&uart6_pending_buf[uart6_packet_len], data, copy_len);
  uart6_packet_len += copy_len;
  uart6_pending_buf[uart6_packet_len] = '\0';
  if (copy_len > 0)
  {
    uart6_packet_ready = 1;
  }
  __enable_irq();

  UART6_ClearRx();
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART1_UART_Init();
  MX_SPI1_Init();
  MX_USART6_UART_Init();
  /* USER CODE BEGIN 2 */
  dac8563_init();
  memset((uint8_t *)task_buf, 0, sizeof(task_buf));
  memset((uint8_t *)task_buf_1, 0, sizeof(task_buf_1));
  memset((uint8_t *)uart6_pending_buf, 0, sizeof(uart6_pending_buf));

  UART1_Init();
  UART6_Init();
  UART1_Register_IDLE_callback(uart1_IDLE_callback);
  UART6_Register_IDLE_callback(uart6_IDLE_callback);
  memset((uint8_t *)polygon_slots,   0, sizeof(polygon_slots));
  memset((uint8_t *)polygon_slots_1, 0, sizeof(polygon_slots_1));
  laser_off();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    process_uart6_packet();

    if (dl_mode == 2)
    {
      for (uint16_t i = 0; i < dl_count && dl_mode == 2; i++)
      {
        if (display_list[i].blank) laser_off();
        else laser_on();
        move(display_list[i].x, display_list[i].y);
      }
    }
    else
    {
      task_t *current_task;
      for (current_task = &task_buf[0]; current_task->type != NONE; ++current_task)
      {
        switch (current_task->type)
        {
        case CIRCLE:
          GPIO_FAST_SETBIT(A, 0);
          draw_circle(current_task->pose.x, current_task->pose.y, current_task->params[0]);
          GPIO_FAST_RESETBIT(A, 0);
          break;

        case RECTANGLE:
          draw_rectangle(current_task->pose.x, current_task->pose.y, current_task->params[0], current_task->params[1]);
          break;

        default:
          break;
        }
      }

      for (uint8_t s = 0; s < 10; s++)
        draw_polygon_slot(s);
    }

    if (flag_update)
    {
      __disable_irq();
      memcpy((uint8_t *)task_buf,         (uint8_t *)task_buf_1,        sizeof(task_buf));
      memcpy((uint8_t *)polygon_slots,    (uint8_t *)polygon_slots_1,   sizeof(polygon_slots));
      memset((uint8_t *)task_buf_1,       0, sizeof(task_buf_1));
      memset((uint8_t *)polygon_slots_1,  0, sizeof(polygon_slots_1));
      flag_update = 0;
      __enable_irq();
    }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE2);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 25;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
