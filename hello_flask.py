import coq_serapy
from flask import Flask, request, render_template, session, redirect, url_for
import os
import random
import json
import glob
from bs4 import BeautifulSoup


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
        except:
            # TODO : Show the user that the input was incorrect and give
            # an option to re-enter corrected input. 
            print("Something went wrong!")
            return 1
    # # os.system("rm -rf search-report/trial")

    # print("THEOREM LEMMA: " + theorem_lemma)

    proof_index = theorem_lemma.find("Proof.")
    admitted_index = theorem_lemma.find("Admitted.")
    clipped_theorem = theorem_lemma[proof_index + 8 : admitted_index]

    proof_prefix = ''.join(list(map(lambda x: ' ' if (x == "\n" or x == '\r') else x, clipped_theorem)))
    proof_prefix = '"' + proof_prefix + '"'

    cmdtorun = "./src/search_file.py --weightsfile data/polyarg-weights.dat --search-type " + search_type + " " + trialfile + " --no-generate-report"

    if proof_prefix:
        cmdtorun = "./src/search_file.py --weightsfile data/polyarg-weights.dat --search-prefix " + proof_prefix + " --search-type " + search_type + " " + trialfile + " --no-generate-report"

    os.system(cmdtorun)
    # TODO : Show user that the search is still going on, just to make sure that nothing has gone wrong.

    with open("search-report/trial" + random_id + "-proofs.txt", "r") as file:
        for line in file:
            pass
        last_line = line
        last_line = last_line[last_line.find('{"status'):][:-2]
        print(theorem_lemma.split("\n")) 
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
        # original_tag = soup.body
        # home_dir = os.path.expanduser('~')
        # svgfile = glob.glob('trial' + random_id + '*.svg', 
        #     root_dir="static/")[0]
        # new_tag = soup.new_tag("img", src="{{url_for('static', filename=\'" + svgfile + "\')}}",
        #  width="100%", height="auto")
        # original_tag.append(new_tag)
        soup.body.append(soup.new_tag("script", src="{{url_for('static', filename='d3-tree.js')}}"))
        # TODO : Make the search tree collapsible and expandable. Hovering over the nodes can show proof until that point.
        with open("modified_html" + random_id + ".html", "w") as fp2:
            fp2.write(soup.prettify())
        fp2.close()
    fp.close()
    os.system("mv modified_html" + random_id + ".html templates/")
    return 0

def get_choices():
    choices = ["dfs", "beam-bfs", "best-first"]
    return choices
def get_err_msg():
    return "Invalid Coq! Please fix any typos or missing imports before retrying."

@app.route('/')
def my_form():
    choices = get_choices()
    return render_template('user_input.html', choices=choices,err_msg='')

@app.route('/', methods=['POST'])
def my_form_post():
    theorem_lemma = request.form['theorem_lemma']
    random_id = random.randrange(1000000)
    search_type = request.form['search_type']
    code = prove_and_print(theorem_lemma, str(random_id), search_type)
    if code == 1:
        choices = get_choices()
        err_msg = get_err_msg()
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