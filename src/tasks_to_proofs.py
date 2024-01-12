import sys
import argparse
import json
import coq_serapy

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("tasks_file")
  parser.add_argument("output_file")
  args = parser.parse_args()

  if args.output_file == "-":
    output_file = sys.stdout
  else:
    output_file = open(args.output_file, 'w')

  seen = set()
  with open(args.tasks_file, 'r') as f:
    for line in f:
      task_dict = json.loads(line)
      lemma_name = coq_serapy.lemma_name_from_statement(
        task_dict["proof_statement"])
      full_lemma_name = task_dict["module_prefix"] + lemma_name
      if full_lemma_name not in seen:
        print(full_lemma_name, file=output_file)
      seen.add(full_lemma_name)

if __name__ == "__main__":
  main()
