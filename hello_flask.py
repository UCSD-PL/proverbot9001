import json
import multiprocessing
import os
import random
import sys

from bs4 import BeautifulSoup
from flask import Flask, redirect, render_template, request, session, url_for

import coq_serapy
sys.path.insert(0, './src')
from src.search_file import main

app = Flask(__name__)

def prove_and_print(theorem_lemma, random_id, search_type):
    # if(not(theorem_lemma.split("\n")[-2:] == ["Proof.""Admitted."])):
    #     theorem_lemma += "\nProof.\nAdmitted."
    # print(theorem_lemma)
    trialfile = "trial" + random_id + ".v"
    f = open(trialfile, "w")
    f.write(theorem_lemma)
    f.close()

    with coq_serapy.SerapiContext(
            # How you want the underlying sertop binary to be run. If not sure,
            # use this.
            ["sertop", "--implicit"],
            # A top level module for the code to reside in. Empty string or
            # None leaves in the default top module.
            None,
            # A prelude directory in which to start the binary
            ".") as coq:

        coq.quiet = True
        proof_commands = coq_serapy.load_commands(trialfile)
        try:
            cmds_left, cmds_run = coq.run_into_next_proof(
            proof_commands)
            _, _ = coq.finish_proof(cmds_left)
            print("Valid Coq!")
        except Exception as e: 
            print("Something went wrong!")
            return 1, str(e)

    admitted_index = theorem_lemma.find("Admitted.")
    proof_index = theorem_lemma.rfind("Proof.", 0, admitted_index)
    clipped_theorem = theorem_lemma[proof_index + 8 : admitted_index]

    proof_prefix = ''.join(list(map(lambda x: ' ' if (x == "\n" or x == '\r') else x, clipped_theorem)))
    proof_prefix = '"' + proof_prefix + '"'


    if not (proof_prefix == '""'):
       main(["--weightsfile", "data/polyarg-weights.dat",  "--search-prefix", '"' + str(proof_prefix) + '"',"--search-type",  str(search_type), str(trialfile), "--no-generate-report"])
    else:
        main(["--weightsfile", "data/polyarg-weights.dat","--search-type", str(search_type), str(trialfile), "--no-generate-report", "-vvv"])

    # TODO : Show user that the search is still going on, just to make sure that nothing has gone wrong.

    with open("search-report/trial" + random_id + "-proofs.txt", "r") as file:
        for line in file:
            pass
        last_line = line
        last_line = last_line[last_line.find('{"status'):][:-2]
        parsedjson = json.loads(last_line)
        with open("proved_theorem" + random_id + ".v", "w") as f:
            if proof_prefix:
                splits = theorem_lemma.split("\n")
                for line in splits[:splits.index('Proof.\r')]:
                    f.write(line)
                    f.write("\n")
            else:
                for line in theorem_lemma.split("\n")[:-2]:
                    f.write(line)
                    f.write("\n")
            commands = parsedjson["commands"]
            for i in range(len(commands)):
                f.write(commands[i]["tactic"] + "\n")
        f.close()
    file.close()

    os.system("mv search-report/trial" + random_id + "*.svg static/")
    os.system("alectryon --frontend coq --backend webpage proved_theorem" + 
    random_id + ".v -o proved_theorem" + random_id + ".html")
    
    with open("proved_theorem" + random_id + ".html") as fp:
        soup = BeautifulSoup(fp, "lxml")
        for link in soup.findAll('link'):
            link['href'] = link['href'].replace("alectryon.css", "{{url_for('static', filename='alectryon.css')}}")
            link['href'] = link['href'].replace("pygments.css", "{{url_for('static', filename='pygments.css')}}")
        for script in soup.findAll('script'):
            script['src'] = script['src'].replace("alectryon.js", "{{url_for('static', filename='alectryon.js')}}")
        for div in soup.find_all("div", {'class':'alectryon-banner'}): 
                div.decompose()
        soup.head.append(soup.new_tag("script", src="{{url_for('static',filename='d3.min.js')}}"))
        soup.head.append(soup.new_tag("link", rel="stylesheet", href="{{url_for('static', filename='d3-min.css')}}"))
        soup.body.append(soup.new_tag("script", src="{{url_for('static', filename='d3-tree.js')}}"))
        with open("modified_html" + random_id + ".html", "w") as fp2:
            fp2.write(soup.prettify())
        fp2.close()
    fp.close()
    os.system("mv modified_html" + random_id + ".html templates/")
    return 0, "no error"

def get_choices():
    choices = ["dfs", "beam-bfs", "best-first"]
    return choices


@app.route('/')
def my_form():
    choices = get_choices()
    return render_template('user_input.html', choices=choices,err_msg='')

@app.route('/', methods=['POST'])
def my_form_post():
    multiprocessing.set_start_method('spawn')
    theorem_lemma = request.form['theorem_lemma']
    random_id = random.randrange(1000000)
    search_type = request.form['search_type']
    if (search_type in get_choices()):
        code, err_msg = prove_and_print(theorem_lemma, str(random_id), search_type)
    else:
        # the search type gets concatenated into a bash command
        # suppose an adversary passed "; <MALICIOUS COMMAND> #" in the form
        # then we are just running arbitrary bash scripts from the user. terrifying!
        # at the very least we can check the input
        code = 1
        err_msg = "invalid search type"
    if code == 1:
        return render_template('user_input.html', theorem_lemma=theorem_lemma, err_msg=err_msg)
    return render_template("modified_html" + str(random_id) + ".html")


@app.route('/rev_rev_list/')
def rrl():
    return render_template('rev_rev.html')

@app.route('/zero_eq_five_imp_false/')
def zeqfif():
    return render_template('zeqfif.html')

@app.route('/one_plus_n_Sn/')
def opnsn():
    return render_template('opnsn.html')

@app.route('/app_nil_r/')
def app_nil_r():
    return render_template('App_nil_r.html')

@app.route('/false_false/')
def false_false():
    return render_template('false_false.html')
