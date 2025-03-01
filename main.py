import random
import copy
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

        # 为None的参数生成随机默认值（需满足容量约束）
        if classroom_code is None:
            # 筛选符合类型且容量足够的教室
            valid_classrooms = [c for c in classrooms
                              if c['type'] == classroom_type
                              and c['capacity'] >= class_size]
            if not valid_classrooms:
                raise ValueError(f"No available {classroom_type} classroom with capacity ≥ {class_size}")
            classroom_code = random.choice(valid_classrooms)['id']
            
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
        """随机改变基因的教室、星期和开始节次（需满足容量约束）"""
        valid_classrooms = [c for c in classrooms
                          if c['type'] == self.classroom_type
                          and c['capacity'] >= self.class_size]
        if not valid_classrooms:
            raise ValueError(f"No available {self.classroom_type} classroom with capacity ≥ {self.class_size}")
        self.classroom_code = random.choice(valid_classrooms)['id']
        self.weekday = random.randint(1, 7)
        if self.duration_sections == 2:
            self.start_section = random.choice([1,3,5,7])
        if self.duration_sections == 4:
            self.start_section = random.choice([1,3,5])
            
class DNA:
    def __init__(self, tasks):
        self.genes = [Gene(**task) for task in tasks]
        
    def mutate(self):
        """随机选择一个基因进行变异"""
        random.choice(self.genes).mutate()
        
    def crossover(self, other):
        """
        与另一个DNA对象进行交叉
        随机选择一个位置，将该位置的基因替换为另一个DNA对象的基因
        """
        if len(self.genes) != len(other.genes):
            raise ValueError("两个DNA对象的基因长度必须相同")
            
        if self.genes:
            crossover_point = random.randint(0, len(self.genes) - 1)
            self.genes[crossover_point] = other.genes[crossover_point]
    def check_conflict(self):
        """计算时间冲突数量"""
        self.conflict_count = 0
        
        # 初始化三个日历字典
        class_calendar = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
        teacher_calendar = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
        classroom_calendar = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

        # 遍历所有基因
        for gene in self.genes:
            # 获取基因的时间信息
            week_range = range(gene.week_start, gene.week_end + 1)
            sections = range(gene.start_section, gene.start_section + gene.duration_sections)
            
            # 更新三个日历
            for week in week_range:
                for section in sections:
                    # 更新班级日历
                    class_calendar[gene.class_code][week][gene.weekday][section] += 1
                    # 更新教师日历
                    teacher_calendar[gene.teacher_id][week][gene.weekday][section] += 1
                    # 更新教室日历
                    classroom_calendar[gene.classroom_code][week][gene.weekday][section] += 1

        # 统计所有冲突（包括容量违规）
        total_conflicts = 0
        
        # 检查教室容量约束
        for gene in self.genes:
            classroom = next(c for c in classrooms if c['id'] == gene.classroom_code)
            if classroom['capacity'] < gene.class_size:
                total_conflicts += 1  # 容量不足视为硬约束冲突
        
        # 遍历班级日历
        for class_schedules in class_calendar.values():
            for week_schedules in class_schedules.values():
                for day_schedules in week_schedules.values():
                    for count in day_schedules.values():
                        if count > 1:
                            total_conflicts += (count - 1)
                            
        # 遍历教师日历
        for teacher_schedules in teacher_calendar.values():
            for week_schedules in teacher_schedules.values():
                for day_schedules in week_schedules.values():
                    for count in day_schedules.values():
                        if count > 1:
                            total_conflicts += (count - 1)
                            
        # 遍历教室日历
        for room_schedules in classroom_calendar.values():
            for week_schedules in room_schedules.values():
                for day_schedules in week_schedules.values():
                    for count in day_schedules.values():
                        if count > 1:
                            total_conflicts += (count - 1)

        self.conflict_count = total_conflicts
    def check_soft_constraints(self):
        total_conflicts = 0
        # 检查周末排课情况（周六=6，周日=7）
        for gene in self.genes:
            if gene.weekday >= 6:  
                total_conflicts += 1

        self.soft_constraint_count = total_conflicts

    def calculate_fitness(self):
        """计算并存储适应度值"""
        self.check_conflict()
        self.check_soft_constraints()
        self.fitness = -(self.conflict_count * 1000 + self.soft_constraint_count)
    def print(self,cnt):
        print(f"第{cnt}次, 硬约束冲突={self.conflict_count}, 软约束冲突={self.soft_constraint_count}, weekday={self.genes[0].weekday}")
    




class Population:
    """遗传算法种群类"""
    
    def __init__(self, tasks, population_size=100, max_generations=100,
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
        self.max_generations = max_generations
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.best_dna = None
        
    
    def evolve(self):
        """执行进化流程"""
        for generation in range(self.max_generations):
            # 评估种群
            for dna in self.population:
                dna.calculate_fitness()

            
            # 选择前N%的个体并保留最优（深拷贝）
            select_size = int(len(self.population) * self.selection_rate)
            selected = [copy.deepcopy(dna) for dna in
                       sorted(self.population, key=lambda dna: dna.fitness, reverse=True)[:select_size]]
            # 更新最优个体为深拷贝
            if self.best_dna is None or selected[0].fitness > self.best_dna.fitness:
                self.best_dna = copy.deepcopy(selected[0])


            # 生成下一代
            children = []
            while len(children) < len(self.population) - select_size:
                # 选择父母并进行单子代交叉
                parent1, parent2 = random.sample(selected, 2)

                child = copy.deepcopy(parent1)
                child.crossover(parent2)  # 使用DNA类自身的交叉方法
                children.append(child)

            
            # 合并新旧种群
            self.population = copy.deepcopy(selected + children[:len(self.population)-select_size])

            # 变异
            for dna in self.population:
                if random.random() < self.mutation_rate:
                    dna.mutate()
                    
            # 输出当前最优
            self.best_dna.print(generation+1)

            if self.best_dna.fitness == 0:
                break
        
        return self.best_dna



classrooms = [
    {'id': 'JXL5#517', 'type': '多媒体教室', 'capacity': 100},
    {'id': 'JYYS2#203', 'type': '虚拟仿真实训室', 'capacity': 80},
    {'id': 'QCGCZX3-305', 'type': '多媒体教室', 'capacity': 50},
    {'id': 'JDGCZX105', 'type': '多媒体教室', 'capacity': 50},
    {'id': 'XXJSZX304', 'type': '机房', 'capacity': 50},
    {'id': 'ZHL3-302', 'type': '多媒体教室', 'capacity': 50},
    {'id': 'JYYS2#205', 'type': '玩具实训室', 'capacity': 50},
    {'id': 'QCGCZX3-309', 'type': '智能网联实训室', 'capacity': 50},
    {'id': 'JXL4#415', 'type': '乐理实训室', 'capacity': 50},
    {'id': 'JYYS1#101', 'type': '舞蹈教室', 'capacity': 50}
]
#capacity字段表示教室可以容纳的学生数量

teachers = [
    {'id': '0130', 'name': '曹立汶'},
    {'id': '0681', 'name': '郭科伟'},
    {'id': '0372', 'name': '赵亚丽'},
    {'id': '0381', 'name': '王晶晶'},
    {'id': '0761', 'name': '孟凯凯'},
    {'id': '0481', 'name': '孟怀洲'},
    {'id': '0586', 'name': '张丰'},
    {'id': '0611', 'name': '李来徽'},
    {'id': '0582', 'name': '彭廷海'},
    {'id': '0166', 'name': '赵银芳'}
]

tasks = [
    {
        'course_code': '570102KBOB03',
        'teacher_id': '0130',
        'class_code': '570102KBOB032024202511017',
        'priority': 6,
        'class_size': 31,
        'week_start': 10,
        'week_end': 18,
        'duration_sections': 2,
        'classroom_type': '多媒体教室',
        'classroom_code': None,
        'weekday': None,
        'start_section': None
    }
]

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
