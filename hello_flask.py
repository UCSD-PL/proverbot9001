import json
import os
import random

from bs4 import BeautifulSoup
from flask import Flask, redirect, render_template, request, url_for

import coq_serapy

app = Flask(__name__)

def prove_and_print(theorem_lemma, random_id):
    # if(not(theorem_lemma.split("\n")[-2:] == ["Proof.""Admitted."])):
    #     theorem_lemma += "\nProof.\nAdmitted."
    # print(theorem_lemma)
    trialfile = "trial" + random_id + ".v"
    f = open(trialfile, "w")
    f.write(theorem_lemma)
    f.close()

    with coq_serapy.SerapiContext(
            ["sertop", "--implicit"], None, ".") as coq:

        coq.quiet = True
        proof_commands = coq_serapy.load_commands(trialfile)
        try:
            cmds_left, cmds_run = coq.run_into_next_proof(
            proof_commands)
            _, _ = coq.finish_proof(cmds_left)
        except Exception as e: 
            return 1, str(e)

    admitted_index = theorem_lemma.find("Admitted.")
    proof_index = theorem_lemma.rfind("Proof.", 0, admitted_index)
    clipped_theorem = theorem_lemma[proof_index + 8 : admitted_index]

    proof_prefix = ''.join(list(map(lambda x: ' ' if (x == "\n" or x == '\r') else x, clipped_theorem)))
    proof_prefix = '"' + proof_prefix + '"'

    cmdtorun = "./src/search_file.py --weightsfile data/polyarg-weights.dat " + trialfile + " --no-generate-report"

    if not (proof_prefix == '""'):
        cmdtorun = "./src/search_file.py --weightsfile data/polyarg-weights.dat --search-prefix " + proof_prefix + " " + trialfile + " --no-generate-report"

    os.system(cmdtorun)
    # TODO : Show user that the search is still going on, just to make sure that nothing has gone wrong.

    theorem_successfully_proved = True

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
            if (commands[(len(commands)) - 1]["tactic"] == "Admitted."):
                theorem_successfully_proved = False
        f.close()
    file.close()

    with open("trial" + random_id + "Zd-json_graph.txt", "r") as json_graph:
        d3_tree = json_graph.read()
    json_graph.close()

    with open("static/d3-interactive.js", "r") as d3_interact:
        with open("static/d3-tree" + random_id + ".js", "w") as d3_tree_random_id:
            d3_tree_random_id.write("var treeData = " + d3_tree + ";")
            d3_tree_random_id.write(d3_interact.read())
        d3_tree_random_id.close()
    d3_interact.close()

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
        if (soup.title is not None):
            new_title = soup.new_tag("title")
            new_title.string = "Proofster Results"
            soup.title.replace_with(new_title)
        back_button = soup.new_tag("a", href="/", **{"class":"button"})
        back_button.string = "Go back"
        soup.head.append(back_button)
        soup.head.append(soup.new_tag("script", src="{{url_for('static',filename='d3.min.js')}}"))
        soup.head.append(soup.new_tag("link", rel="stylesheet", href="{{url_for('static', filename='d3-min.css')}}"))
        soup.head.append(soup.new_tag("link", rel="stylesheet", href="https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css"))
        soup.head.append(soup.new_tag("link", rel="stylesheet", href="{{url_for('static', filename='footer.css')}}"))
        soup.body.append(soup.new_tag("script", src="{{url_for('static', filename='d3-tree" + str(random_id) + ".js')}}"))
        soup.body.insert_before("{% include 'title.html' %}")
        if not theorem_successfully_proved:
            theorem_synthesis_failed = soup.new_tag("div")
            theorem_synthesis_failed['class']="notification is-danger"
            theorem_synthesis_failed['style'] = "font-size: 22px; display: flex; justify-content: center;"
            theorem_synthesis_failed.string = "Sorry, I couldn't synthesize a proof of this theorem for you."
            soup.body.insert_before(theorem_synthesis_failed)
        soup.body.append("{% include 'footer.html' %}")
        with open("modified_html" + random_id + ".html", "w") as fp2:
            fp2.write(soup.prettify())
        fp2.close()
    fp.close()

    os.system("mv modified_html" + random_id + ".html templates/")
    os.system("rm -rf trial" + random_id + "* proved_theorem" + random_id + "* search-report/trial" + random_id + "*")
    return 0, "no error"


@app.route('/')
def my_form():
    return render_template('user_input.html')

@app.route('/', methods=['POST'])
def my_form_post():
    theorem_lemma = request.form['theorem_lemma']
    if (theorem_lemma == ""):
        return render_template('user_input.html', theorem_lemma="", err_msg="Please enter a theorem to be proved.")
    random_id = random.randrange(1000000)
    code, err_msg = prove_and_print(theorem_lemma, str(random_id))
    if (code == 1):
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
