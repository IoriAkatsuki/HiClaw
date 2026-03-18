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

int16_t current_x = 0;
int16_t current_y = 0;
uint8_t current_laser_state = 2; // 0=off, 1=on, 2=unknown
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

void uart6_IDLE_callback(uint8_t *data, uint16_t num)
{
  uint8_t *buf = UART6_GetRxData();

  char *token = strtok((char *)buf, ";");
  while (token != NULL)
  {
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
    
    token = strtok(NULL, ";");
  }

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

  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_SET);

  UART1_Init();
  UART6_Init();
  UART6_Register_IDLE_callback(uart6_IDLE_callback);

  // while (1)
  // {
  //   draw_rectangle(0, 0, 10000, 10000);
  // }

  task_buf[0].type = CIRCLE;
  task_buf[0].pose.x = 5000;
  task_buf[0].pose.y = 5000;
  task_buf[0].params[0] = 10000;

  // task_buf[1].type = CIRCLE;
  // task_buf[1].pose.x = 0;
  // task_buf[1].pose.y = 5000;
  // task_buf[1].params[0] = 1000;

  // task_buf[2].type = CIRCLE;
  // task_buf[2].pose.x = -5000;
  // task_buf[2].pose.y = 5000;
  // task_buf[2].params[0] = 1000;

  // task_buf[3].type = RECTANGLE;
  // task_buf[3].pose.x = 0;
  // task_buf[3].pose.y = 0;
  // task_buf[3].params[0] = 2000;
  // task_buf[3].params[1] = 1000;
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    task_t *current_task;
    for (current_task = &task_buf[0]; current_task->type != NONE; ++current_task)
    {
      pose_t next_pose = get_next_pose(current_task);
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

    
    if (flag_update)
    {
      memcpy((uint8_t *)task_buf, (uint8_t *)task_buf_1, sizeof(task_buf));
      memset((uint8_t *)task_buf_1, 0, sizeof(task_buf_1));
      flag_update = 0;
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
