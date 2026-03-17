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
#include "dma.h"
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
#define BUF_SIZE 1024
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
const float PI = 3.14159265358979323846f;

uint8_t uart1_rx_buf[BUF_SIZE];

task_t task_buf[10];
task_t task_buf_1[10];
volatile uint8_t flag_update = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
void draw_circle(int16_t x, int16_t y, uint16_t radius)
{
  for (int i = 0; i < 90; ++i)
  {
    float theta = (2.0f * PI * i) / 90;
    int16_t target_x = (int16_t)(radius * cosf(theta) + x);
    int16_t target_y = (int16_t)(radius * sinf(theta) + y);
    dac8563_output_int16(target_x, target_y);
    zjs_delay_us(20); // 根据需要调整速度
  }
}

void draw_rectangle(int16_t x, int16_t y, uint16_t length, uint16_t height)
{
  int i = 0;
  for (i = 0; i < length; i += 100)
  {
    dac8563_output_int16(x - i + length / 2, y + height / 2);
    zjs_delay_us(120);
  }
  for (i = 0; i < height; i += 100)
  {
    dac8563_output_int16(x - length / 2, y - i + height / 2);
    zjs_delay_us(120);
  }
  for (i = 0; i < length; i += 100)
  {
    dac8563_output_int16(x + i - length / 2, y - height / 2);
    zjs_delay_us(120);
  }
  for (i = 0; i < height; i += 100)
  {
    dac8563_output_int16(x + length / 2, y + i - height / 2);
    zjs_delay_us(120);
  }
}

void move(int16_t x, int16_t y, uint16_t delay_us)
{
  zjs_delay_us(400);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_RESET);
  dac8563_output_int16(x, y);
  zjs_delay_us(delay_us);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_SET);
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

void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t Size)
{
  if (huart->Instance == USART1)
  {
    if (uart1_rx_buf[0] == 'U')
    {
      flag_update = 1;
    }

    if (uart1_rx_buf[0] == 'C')
    {
      int i = uart1_rx_buf[1] - '0';
      int16_t x;
      int16_t y;
      uint16_t radius;
      sscanf((const char *)&uart1_rx_buf[2], "%hd,%hd,%hu", &x, &y, &radius);

      task_buf_1[i].type = CIRCLE;
      task_buf_1[i].pose.x = x;
      task_buf_1[i].pose.y = y;
      task_buf_1[i].params[0] = radius;
    }

    if (uart1_rx_buf[0] == 'R')
    {
      int i = uart1_rx_buf[1] - '0';
      int16_t x;
      int16_t y;
      uint16_t length;
      uint16_t height;
      sscanf((const char *)&uart1_rx_buf[2], "%hd,%hd,%hu,%hu", &x, &y, &length, &height);

      task_buf_1[i].type = RECTANGLE;
      task_buf_1[i].pose.x = x;
      task_buf_1[i].pose.y = y;
      task_buf_1[i].params[0] = length;
      task_buf_1[i].params[1] = height;
    }

    // G 命令：立即移动到指定位置（标定模式使用）
    if (uart1_rx_buf[0] == 'G')
    {
      int16_t x;
      int16_t y;
      sscanf((const char *)&uart1_rx_buf[1], "%hd,%hd", &x, &y);
      dac8563_output_int16(x, y);
    }

    // L 命令：激光开关（标定模式使用）
    if (uart1_rx_buf[0] == 'L')
    {
      if (uart1_rx_buf[1] == '1')
      {
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_SET);
      }
      else if (uart1_rx_buf[1] == '0')
      {
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_RESET);
      }
    }

    HAL_UARTEx_ReceiveToIdle_DMA(huart, uart1_rx_buf, BUF_SIZE);
  }

  memset(uart1_rx_buf, 0, BUF_SIZE);
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
  MX_DMA_Init();
  MX_USART1_UART_Init();
  MX_SPI1_Init();
  /* USER CODE BEGIN 2 */
  dac8563_init();

  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_SET);

  HAL_UARTEx_ReceiveToIdle_DMA(&huart1, uart1_rx_buf, BUF_SIZE);

  task_buf[0].type = CIRCLE;
  task_buf[0].pose.x = 5000;
  task_buf[0].pose.y = 5000;
  task_buf[0].params[0] = 1000;

  task_buf[1].type = CIRCLE;
  task_buf[1].pose.x = 0;
  task_buf[1].pose.y = 5000;
  task_buf[1].params[0] = 1000;

  task_buf[2].type = CIRCLE;
  task_buf[2].pose.x = -5000;
  task_buf[2].pose.y = 5000;
  task_buf[2].params[0] = 1000;

  task_buf[3].type = RECTANGLE;
  task_buf[3].pose.x = 0;
  task_buf[3].pose.y = 0;
  task_buf[3].params[0] = 2000;
  task_buf[3].params[1] = 1000;
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
        move(next_pose.x, next_pose.y, 1500);
        draw_circle(current_task->pose.x, current_task->pose.y, current_task->params[0]);
        GPIO_FAST_RESETBIT(A, 0);
        break;

      case RECTANGLE:
        move(next_pose.x, next_pose.y, 1500);
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
