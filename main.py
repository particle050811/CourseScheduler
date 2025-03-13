import random
import copy
import numpy as np
import math
from collections import defaultdict

class Gene:
    """表示排课基因的染色体单元"""
    def __init__(self,
                 course_code: str,
                 teacher_id: str,
                 class_code: str,
                 priority: int,
                 class_size: int,  # 新增班级人数参数
                 week_start: int,
                 week_end: int,
                 duration_sections: int,
                 classroom_type: str,
                 classroom_code: str = None,
                 weekday: int = None,
                 start_section: int = None):

        if classroom_code is None:
            classroom_code = random.choice(valid_classrooms[classroom_type])
            
        if weekday is None:
            weekday = random.randint(1, 7)  # 随机星期1-7
            
        if start_section is None:
            if duration_sections == 2:
                start_section = random.choice([1,3,5,7])
            if duration_sections == 4:
                start_section = random.choice([1,3,5])
        # 随机选择开始节次
            

        # 初始化基因参数
        self.course_code = course_code
        self.teacher_id = teacher_id
        self.class_code = class_code
        self.priority = priority
        self.week_start = week_start
        self.week_end = week_end
        self.duration_sections = duration_sections
        self.classroom_type = classroom_type
        self.class_size = class_size  # 新增班级人数属性
        self.classroom_code = classroom_code
        self.weekday = weekday  # 星期几（1-7）
        self.start_section = start_section  # 开始节次
        
    def mutate(self):
        """随机改变基因的教室、星期和开始节次"""
        self.classroom_code = random.choice(valid_classrooms[self.classroom_type])

        self.weekday = random.randint(1, 7)

        if self.duration_sections == 2:
            self.start_section = random.choice([1,3,5,7])
        if self.duration_sections == 4:
            self.start_section = random.choice([1,3,5])
        
class Calendar:
    def __init__(self):
        self.calendar = np.zeros((25,7,4), dtype=int)
    def add(self, gene:Gene):
        conflict = 0
        weekday = gene.weekday - 1
        start = gene.start_section // 2
        end = (gene.start_section+gene.duration_sections) // 2
        for week in range(gene.week_start,gene.week_end+1):
            for section in range(start,end):
                if self.calendar[week, weekday, section] >= 1:
                    conflict += 1
                self.calendar[week, weekday, section] += 1
        return conflict
    
    def sub(self, gene:Gene):
        conflict = 0
        weekday = gene.weekday - 1
        start = gene.start_section // 2
        end = (gene.start_section+gene.duration_sections) // 2
        for week in range(gene.week_start,gene.week_end+1):
            for section in range(start,end):
                if self.calendar[week, weekday, section] > 1:
                    conflict += -1
                self.calendar[week, weekday, section] += -1
        return conflict

    def calculate(self, gene:Gene):
        conflict = 0
        weekday = gene.weekday - 1
        start = gene.start_section // 2
        end = (gene.start_section+gene.duration_sections) // 2
        for week in range(gene.week_start,gene.week_end+1):
            for section in range(start,end):
                conflict += self.calendar[week, weekday, section]
        return conflict
        
class DNA:
    def __init__(self, tasks):
        self.genes = [Gene(**task) for task in tasks]

        # 初始化三个日历字典
        self.class_calendar = defaultdict(Calendar)
        self.teacher_calendar = defaultdict(Calendar)
        self.classroom_calendar = defaultdict(Calendar)
        self.time_conflict = 0
        self.average_conflict = 1
        for gene in self.genes:
            self.add(gene)

    def add(self, gene:Gene):
        self.time_conflict += self.class_calendar[gene.class_code].add(gene)
        self.time_conflict += self.teacher_calendar[gene.teacher_id].add(gene)
        self.time_conflict += self.classroom_calendar[gene.classroom_code].add(gene)
    
    def sub(self, gene:Gene):
        self.time_conflict += self.class_calendar[gene.class_code].sub(gene)
        self.time_conflict += self.teacher_calendar[gene.teacher_id].sub(gene)
        self.time_conflict += self.classroom_calendar[gene.classroom_code].sub(gene)        

    def calculate(self, gene:Gene):
        conflict = 0
        conflict += self.class_calendar[gene.class_code].calculate(gene)
        conflict += self.teacher_calendar[gene.teacher_id].calculate(gene)
        conflict += self.classroom_calendar[gene.classroom_code].calculate(gene)

        if classrooms_capacity[gene.classroom_code] < gene.class_size:
            conflict += 10  # 容量不足视为硬约束冲突
        if gene.weekday >= 6:  
            conflict += 1  # 检查周末排课情况（周六=6，周日=7）

        return conflict        

    def mutate(self, time):
        for i in range(len(self.genes)):
            gene = self.genes[i]
            self.sub(gene)
            gene_mutated = copy.deepcopy(gene)
            gene_best = copy.deepcopy(gene)

            conflict = self.calculate(gene)

            for j in range(time):
                if conflict <= self.average_conflict*math.log10(j+7):
                    break

                gene_mutated.mutate()
                conflict_mutated = self.calculate(gene_mutated)

                if conflict > conflict_mutated:
                    gene_best = copy.deepcopy(gene_mutated)
                    conflict = conflict_mutated
            
            self.add(gene_best)    
            self.genes[i] = copy.deepcopy(gene_best)

    def calculate_conflict(self):
        self.room_conflict = 0
        self.soft_conflict = 0

        for gene in self.genes:
            if classrooms_capacity[gene.classroom_code] < gene.class_size:
                self.room_conflict += 1  # 容量不足视为硬约束冲突
            if gene.weekday >= 6:  
                self.soft_conflict += 1  # 检查周末排课情况（周六=6，周日=7）

        self.conflict = self.time_conflict + self.room_conflict*10 + self.soft_conflict
        self.average_conflict = self.conflict / len(self.genes)

        return self.conflict

    def print(self,cnt):
        print(  f"第{cnt}次, 时间约束冲突={self.time_conflict}, "
                f"教室约束冲突={self.room_conflict}, 软约束冲突={self.soft_conflict}")

    def test(self):
        pass

    




class Population:
    """遗传算法种群类"""
    
    def __init__(self, tasks, population_size=1, max_generations=10,
                 selection_rate=0.7, mutation_rate=0.1):
        """
        参数:
        tasks: 初始任务列表
        population_size: 种群规模
        max_generations: 最大迭代次数
        selection_rate: 选择率（前N%的个体保留）
        mutation_rate: 变异概率
        """
        self.population = [DNA(tasks) for _ in range(population_size)]
        self.population_size = population_size
        self.max_generations = max_generations
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.best_dna = None
        
    
    def evolve(self):
        """执行进化流程"""
        time=1000
        for generation in range(self.max_generations):

            for dna in self.population:
                dna.mutate(time)

            self.population = sorted(self.population, key=lambda dna: dna.calculate_conflict())

            self.population[0].print(generation+1)

        return self.population[0]



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
teachers_df = pd.read_excel('教师信息.xlsx', engine='openpyxl')
teachers = teachers_df[['工号', '姓名']].rename(
    columns={
        '工号': 'id',
        '姓名': 'name'
    }
).to_dict('records')

# 从排课任务.xlsx读取任务数据
tasks_df = pd.read_excel('排课任务.xlsx', engine='openpyxl', dtype={'教师工号': str})

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


if __name__ == '__main__':
    # 初始化种群
    population = Population(tasks)

    # 执行进化流程
    best_dna = population.evolve()

    # 输出最佳排课结果

    print("Best Schedule:")
    for gene in best_dna.genes:
        print(f"Course: {gene.course_code}, Teacher: {gene.teacher_id}, Class: {gene.class_code}, "
              f"Classroom: {gene.classroom_code}, Weekday: {gene.weekday}, Section: {gene.start_section}")


