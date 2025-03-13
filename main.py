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
    
    def __init__(self, tasks, population_size=1, max_generations=5,
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


from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/arrange', methods=['POST'])
def arrange_schedule():
    try:
        data = request.get_json()
        
        # 验证必要字段
        required_fields = ['tasks', 'classrooms', 'teachers']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        # 处理教室数据
        global valid_classrooms, classrooms_capacity
        valid_classrooms = defaultdict(list)
        classrooms_capacity = {}
        for room in data['classrooms']:
            if not all(key in room for key in ['id', 'type', 'capacity']):
                return jsonify({'error': 'Invalid classroom format'}), 400
            valid_classrooms[room['type']].append(room['id'])
            classrooms_capacity[room['id']] = int(room['capacity'])

        # 处理任务数据
        tasks = []
        for task in data['tasks']:
            required_task_fields = ['course_code', 'teacher_id', 'class_code', 'priority', 
                                   'class_size', 'week_start', 'week_end', 'duration_sections', 'classroom_type']
            if not all(field in task for field in required_task_fields):
                return jsonify({'error': 'Invalid task format'}), 400
            
            tasks.append({
                'course_code': task['course_code'],
                'teacher_id': task['teacher_id'],
                'class_code': task['class_code'],
                'priority': task['priority'],
                'class_size': task['class_size'],
                'week_start': task['week_start'],
                'week_end': task['week_end'],
                'duration_sections': task['duration_sections'],
                'classroom_type': task['classroom_type'],
                'classroom_code': task.get('classroom_code'),
                'weekday': task.get('weekday'),
                'start_section': task.get('start_section')
            })

        # 运行遗传算法
        population = Population(tasks)
        best_dna = population.evolve()

        # 格式化结果
        result = [{
            'course_code': gene.course_code,
            'teacher_id': gene.teacher_id,
            'class_code': gene.class_code,
            'priority': gene.priority,
            'class_size': gene.class_size,
            'week_start': gene.week_start,
            'week_end': gene.week_end,
            'duration_sections': gene.duration_sections,
            'classroom_type': gene.classroom_type,
            'classroom': gene.classroom_code,
            'weekday': gene.weekday,
            'start_section': gene.start_section
        } for gene in best_dna.genes]

        return jsonify({
            'schedule': result,
            'conflicts': {
                'time_conflict': best_dna.time_conflict,
                'room_conflict': best_dna.room_conflict,
                'soft_conflict': best_dna.soft_conflict
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


