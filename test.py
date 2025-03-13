import random
import copy
import numpy as np
import math
from collections import defaultdict



import pandas as pd

# 从Excel文件读取教室信息
classrooms_df = pd.read_excel('教室信息.xlsx', engine='openpyxl')
classrooms = classrooms_df[['教室编号', '教室类型', '最大上课容纳人数']].rename(
    columns={
        '教室编号': 'id',
        '教室类型': 'type',
        '最大上课容纳人数': 'capacity'
    }
).to_dict('records')


valid_classrooms = defaultdict(list)  # 使用defaultdict自动初始化空列表
classrooms_capacity = {}

# 确保数据格式正确
for room in classrooms:
    room['capacity'] = int(room['capacity'])
    valid_classrooms[room['type']].append(room['id'])  # 将教室ID添加到对应类型的列表
    classrooms_capacity[room['id']] = room['capacity']

# 验证每个教室类型都有可用教室
for room_type, ids in valid_classrooms.items():
    if not ids:
        raise ValueError(f"教室类型'{room_type}'没有对应的教室，请检查教室信息表")
    if not all(key in room for key in ('id', 'type', 'capacity')):
        raise ValueError("教室信息.xlsx 文件格式错误，必须包含：教室编号、教室类型、最大上课容纳人数")


# 从Excel文件读取教师信息
teachers_df = pd.read_excel('教师信息.xlsx', engine='openpyxl', dtype={'工号': str})
teachers = teachers_df[['工号', '姓名']].rename(
    columns={
        '工号': 'id',
        '姓名': 'name'
    }
).to_dict('records')

# 从排课任务.xlsx读取任务数据
tasks_df = pd.read_excel('排课任务.xlsx', engine='openpyxl', dtype={'课程编号': str, '教师工号': str, '教学班编号': str, '连排节次': int, '排课优先级': int})

# 分割周次和总节次
def parse_schedule(schedule_str, duration_sections):
    """解析多时间段开课周次格式（示例：'5-8:6,13-16:2'）
    返回元组列表[(周起始，周结束，单次节次), ...]"""
    try:
        result = []
        # 分割多个时间段
        for part in schedule_str.split(','):
            # 分割周次和总节次
            week_part, total_sec = part.split(':')
            total_sections = int(total_sec)
            
            # 验证总节次是否匹配连排节次
            if total_sections % duration_sections != 0:
                raise ValueError(f"总节次{total_sections}无法被{duration_sections}整除")
            
            # 计算需要拆分的次数
            times = total_sections // duration_sections
            
            # 解析周范围
            week_start, week_end = map(int, week_part.split('-'))
            
            # 生成对应数量的记录
            result.extend([(week_start, week_end, duration_sections)] * times)
            
        return result
    except Exception as e:
        raise ValueError(f"无效的开课周次格式: {schedule_str}") from e

tasks = []
for _, row in tasks_df.iterrows():
    if pd.isna(row['教师工号']):
        continue
    #print(f'有效任务数量：{len(tasks)}')
    schedule_configs = parse_schedule(row['开课周次学时'], row['连排节次'])
    for config in schedule_configs:
        week_start, week_end, duration_sections = config
        tasks.append({
            'course_code': row['课程编号'],
            'teacher_id': row['教师工号'],
            'class_code': row['教学班编号'],
            'priority': row['排课优先级'],
            'class_size': 50,
            'week_start': week_start,
            'week_end': week_end,
            'duration_sections': row['连排节次'],  # 根据需求文档使用连排节次字段
            'classroom_type': row['指定教室类型'],
            'classroom_code': None,
            'weekday': None,
            'start_section': None
        })

"""
# 根据连排节次从大到小排序任务列表
tasks = sorted(tasks, key=lambda x: x['duration_sections'], reverse=True)
"""


import requests
import unittest
import subprocess
import time
import json

class TestScheduleAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server_process = subprocess.Popen(['python', 'main.py'])
        time.sleep(2)  # 等待服务器启动

    @classmethod
    def tearDownClass(cls):
        cls.server_process.terminate()

    def test_arrange_endpoint(self):
        # 构造测试数据
        test_data = {
            "tasks": tasks,
            "classrooms": classrooms,
            "teachers": teachers
        }
        #with open('test.txt', 'w', encoding='utf-8') as f:
            #f.write(json.dumps(test_data, indent=4, ensure_ascii=False))
        
        # 发送POST请求
        response = requests.post('http://localhost:5000/arrange', json=test_data)
        #print("原始响应内容：", response.text)
        
        # 验证响应状态码
        self.assertEqual(response.status_code, 200)
        
        # 验证返回数据结构
        result = response.json()
        self.assertIn('schedule', result)
        self.assertIn('conflicts', result)
        
        # 验证排课结果非空
        self.assertTrue(len(result['schedule']) > 0)
        """
        # 输出排课详情
        print('\n排课结果详情：')
        for item in result['schedule']:
            print(f"课程{item['course_code']} 教师{item['teacher_id']} 教室{item['classroom']} 周{item['week_start']}-{item['week_end']} 星期{item['weekday']} 第{item['start_section']}节")
        """
        # 验证冲突统计存在
        self.assertIsInstance(result['conflicts']['time_conflict'], int)
        self.assertIsInstance(result['conflicts']['room_conflict'], int)
        self.assertIsInstance(result['conflicts']['soft_conflict'], int)

if __name__ == '__main__':
    unittest.main()
