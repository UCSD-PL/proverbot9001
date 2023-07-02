from collections import defaultdict
import json
import random
from search_worker import ReportJob, Worker, get_predictor

class Taskhandler() :
    def __init__(self) :
        self.something = True

    def configure(self,config_dict = {}) :
        if "curriculum" in config_dict :
            self.curriculum = config_dict["curriculum"]
        else :
            self.curriculum = False

        if "max_target_len" in config_dict :
            self.max_target_len = config_dict["max_target_len"]
        else :
            self.max_target_len = 3

        if "excluded_lemmas" in config_dict :
            self.excluded_lemmas = config_dict["excluded_lemmas"]
        else :
            self.excluded_lemmas = []


    def get_jobs(self, tasks_file) :
        print("Creating jobs with the configuration")
        print("Curriculum : ", self.curriculum)
        print("Excluded Lemmas : ", self.excluded_lemmas)
        print("Max Target Length : ", self.max_target_len)
        jobs_dict = defaultdict(list)
        with open(tasks_file, 'r') as f:
            for line in f:
                task = json.loads(line) 
                if task["target_length"] <= self.max_target_len and not task['proof_statement'] in self.excluded_lemmas:
                    task_job = ReportJob(project_dir=".", filename=task['src_file'], module_prefix=task['module_prefix'], 
                            lemma_statement=task['proof_statement'])
                    jobs_dict[task["target_length"]].append((task_job, task['tactic_prefix']))
        f.close()

        jobs = []
        keys = sorted(jobs_dict.keys())
        for target_length in keys :
            jobs += jobs_dict[target_length]

        if not self.curriculum :
            random.shuffle(jobs)

        return jobs