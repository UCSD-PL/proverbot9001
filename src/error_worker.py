import argparse
from models.tactic_predictor import TacticPredictor
from search_results import SearchResult
from worker import Worker


class ErrorWorker(Worker):
    predictor: TacticPredictor
    def __init__(self, args: argparse.Namespace,
                 predictor: TacticPredictor) -> None:
        super().__init__(args)
        self.predictor = predictor

    # maybe change this to return the error scrape results
    def run_job(self, job: ReportJob, restart: bool = True) -> None:
        assert self.coq
        self.run_into_job(job, restart, self.args.careful)
        job_project, job_file, job_module, job_lemma = job
        try:
            # call error scrape function here
            pass
        except coq_serapy.CoqAnomaly:
            if self.args.hardfail:
                raise
            self.restart_coq()
            self.reset_file_state()
            self.enter_file(job_file)
            if restart:
                eprint("Hit an anomaly, restarting job", guard=self.args.verbose >= 2)
                return self.run_job(job, restart=False)
        except Exception:
            eprint(f"FAILED in file {job_file}, lemma {job_lemma}")
            raise
        # Pop the actual Qed/Defined/Save
        ending_command = self.remaining_commands.pop(0)
        coq_serapy.admit_proof(self.coq, job_lemma, ending_command)
